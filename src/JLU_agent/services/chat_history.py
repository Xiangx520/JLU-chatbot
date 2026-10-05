"""保存完整聊天记录，与 Agent 的压缩上下文分开管理。"""

import json
import sqlite3
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

from JLU_agent.config import agent_config as config


class ChatHistoryService:
    def __init__(self, browser_id: str, path: Path | None = None):
        self.browser_id = browser_id
        self.path = path if path is not None else config.CHAT_HISTORY_DB_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS conversations (
                    thread_id TEXT PRIMARY KEY,
                    browser_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    thread_id TEXT NOT NULL REFERENCES conversations(thread_id),
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    sources TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS conversations_browser
                    ON conversations(browser_id, updated_at);
                CREATE INDEX IF NOT EXISTS messages_thread ON chat_messages(thread_id, id);
            """)

    def _connect(self):
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="microseconds")

    def list_conversations(self) -> list[dict]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT * FROM conversations WHERE browser_id = ? "
                "ORDER BY updated_at DESC, thread_id",
                (self.browser_id,),
            )
            return [dict(row) for row in rows]

    def get_messages(self, thread_id: str) -> list[dict]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT m.role, m.content, m.sources FROM chat_messages m "
                "JOIN conversations c ON c.thread_id = m.thread_id "
                "WHERE m.thread_id = ? AND c.browser_id = ? ORDER BY m.id",
                (thread_id, self.browser_id),
            )
            return [
                {"role": row["role"], "content": row["content"],
                 "reference": json.loads(row["sources"])}
                for row in rows
            ]

    def _create_conversation(self, conn, thread_id: str, question: str, now: str):
        title = " ".join(question.split())[:30] or "历史对话"
        return conn.execute(
            "INSERT INTO conversations VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(thread_id) DO NOTHING",
            (thread_id, self.browser_id, title, now, now),
        ).rowcount

    def _append_message(self, conn, thread_id: str, role: str, content: str,
                        references: list[dict], now: str):
        inserted = conn.execute(
            "INSERT INTO chat_messages (thread_id, role, content, sources) "
            "SELECT thread_id, ?, ?, ? FROM conversations "
            "WHERE thread_id = ? AND browser_id = ?",
            (role, content, json.dumps(references, ensure_ascii=False),
             thread_id, self.browser_id),
        ).rowcount
        if not inserted:
            raise ValueError("当前浏览器无法写入该对话。")
        conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE thread_id = ? AND browser_id = ?",
            (now, thread_id, self.browser_id),
        )

    def save_user_message(self, thread_id: str, content: str) -> None:
        now = self._now()
        with closing(self._connect()) as conn, conn:
            self._create_conversation(conn, thread_id, content, now)
            self._append_message(conn, thread_id, "user", content, [], now)

    def save_assistant_message(self, thread_id: str, content: str,
                               references: list[dict]) -> None:
        with closing(self._connect()) as conn, conn:
            self._append_message(conn, thread_id, "assistant", content, references, self._now())

    def delete_conversation(self, thread_id: str) -> None:
        """只删除当前浏览器所属对话，同时清理 Agent 的检查点。"""
        with closing(self._connect()) as conn, conn:
            conn.execute("BEGIN IMMEDIATE")
            owned = conn.execute(
                "SELECT 1 FROM conversations WHERE thread_id = ? AND browser_id = ?",
                (thread_id, self.browser_id),
            ).fetchone()
            if not owned:
                raise ValueError("该对话不存在或不属于当前浏览器。")
            # 先清理上下文；失败时保留历史记录，方便重新尝试删除。
            if config.CHECKPOINT_DB_PATH.exists():
                with closing(sqlite3.connect(config.CHECKPOINT_DB_PATH)) as checkpoint_conn:
                    SqliteSaver(checkpoint_conn).delete_thread(thread_id)
            conn.execute("DELETE FROM chat_messages WHERE thread_id = ?", (thread_id,))
            conn.execute(
                "DELETE FROM conversations WHERE thread_id = ? AND browser_id = ?",
                (thread_id, self.browser_id),
            )

    def import_conversation(self, thread_id: str, messages: list[dict]) -> None:
        """仅导入当前页面还保留的记录，已存在的会话不重复写入。"""
        if not messages:
            return
        question = next((m["content"] for m in messages if m["role"] == "user"), "历史对话")
        now = self._now()
        with closing(self._connect()) as conn, conn:
            if not self._create_conversation(conn, thread_id, question, now):
                return
            for message in messages:
                self._append_message(
                    conn, thread_id, message["role"], message["content"],
                    message.get("reference", []), now,
                )
