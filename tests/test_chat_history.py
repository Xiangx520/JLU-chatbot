"""完整聊天记录的持久化、排序和浏览器归属测试。"""

import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from langgraph.checkpoint.base import empty_checkpoint
from langgraph.checkpoint.sqlite import SqliteSaver

from JLU_agent.config import agent_config as config
from JLU_agent.services.chat_history import ChatHistoryService


class ChatHistoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "history.db"
        checkpoint_patch = patch.object(config, "CHECKPOINT_DB_PATH", Path(self.temp.name) / "checkpoint.db")
        checkpoint_patch.start()
        self.addCleanup(checkpoint_patch.stop)
        self.first = ChatHistoryService("browser-one", self.path)
        self.second = ChatHistoryService("browser-two", self.path)

    def test_empty_store_and_reopening_preserve_full_messages(self):
        self.assertEqual(self.first.list_conversations(), [])
        self.assertEqual(self.first.get_messages("missing"), [])
        self.first.save_user_message("thread", "  吉林大学\n校园问题" + "字" * 40)
        references = [{"title": "学校官网", "url": "https://www.jlu.edu.cn"}]
        self.first.save_assistant_message("thread", "完整回答", references)
        for i in range(30):
            self.first.save_user_message("thread", f"问题 {i}")
            self.first.save_assistant_message("thread", f"回答 {i}", [])
        restored = ChatHistoryService("browser-one", self.path)
        messages = restored.get_messages("thread")
        self.assertEqual(len(messages), 62)
        self.assertEqual(messages[1]["reference"], references)
        self.assertEqual(messages[1]["content"], "完整回答")
        conversation = restored.list_conversations()[0]
        self.assertEqual(len(conversation["title"]), 30)
        self.assertNotIn("\n", conversation["title"])
        self.assertTrue(conversation["updated_at"].endswith("+08:00"))

    def test_latest_update_orders_conversations_without_changing_title(self):
        with patch.object(self.first, "_now", side_effect=[
            "2026-10-05T10:00:00+08:00", "2026-10-05T11:00:00+08:00",
            "2026-10-05T12:00:00+08:00",
        ]):
            self.first.save_user_message("first", "第一个问题")
            self.first.save_user_message("second", "第二个问题")
            self.first.save_assistant_message("first", "回答", [])
        rows = self.first.list_conversations()
        self.assertEqual([row["thread_id"] for row in rows], ["first", "second"])
        self.assertEqual(rows[0]["title"], "第一个问题")
        self.assertEqual(rows[0]["created_at"], "2026-10-05T10:00:00+08:00")

    def test_browsers_cannot_read_or_append_each_others_conversations(self):
        self.first.save_user_message("first", "浏览器一的问题")
        self.second.save_user_message("second", "浏览器二的问题")
        self.assertEqual(self.second.get_messages("first"), [])
        self.assertEqual([row["thread_id"] for row in self.first.list_conversations()], ["first"])
        for action in (
            lambda: self.second.save_user_message("first", "注入问题"),
            lambda: self.second.save_assistant_message("first", "注入回答", []),
        ):
            with self.assertRaises(ValueError):
                action()
        self.assertEqual(len(self.first.get_messages("first")), 1)

    def test_import_is_once_and_preserves_sources(self):
        messages = [
            {"role": "user", "content": "旧问题"},
            {"role": "assistant", "content": "旧回答", "reference": [
                {"title": "来源", "url": "https://www.jlu.edu.cn"}
            ]},
        ]
        self.first.import_conversation("legacy", messages)
        self.first.import_conversation("legacy", messages)
        restored = self.first.get_messages("legacy")
        self.assertEqual(len(restored), 2)
        self.assertEqual(restored[1], messages[1])
        self.second.import_conversation("legacy", messages)
        self.assertEqual(self.second.list_conversations(), [])
        self.first.import_conversation("blank", [])
        self.assertEqual(len(self.first.list_conversations()), 1)

    def test_user_message_failure_rolls_back_new_conversation(self):
        with patch.object(self.first, "_append_message", side_effect=sqlite3.OperationalError("写入失败")):
            with self.assertRaises(sqlite3.OperationalError):
                self.first.save_user_message("broken", "问题")
        self.assertEqual(self.first.list_conversations(), [])


    def test_delete_removes_messages_checkpoints_and_pending_writes(self):
        self.first.save_user_message("deleted", "待删除问题")
        self.first.save_assistant_message("deleted", "待删除回答", [])
        self.first.save_user_message("kept", "保留问题")
        with SqliteSaver.from_conn_string(str(config.CHECKPOINT_DB_PATH)) as saver:
            for thread_id in ("deleted", "kept"):
                saved = saver.put(
                    {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}},
                    empty_checkpoint(), {"source": "input", "step": 0, "parents": {}}, {},
                )
                saver.put_writes(saved, [("messages", "待执行消息")], "task")
            self.first.delete_conversation("deleted")
            self.assertIsNone(saver.get_tuple({"configurable": {"thread_id": "deleted"}}))
            self.assertIsNotNone(saver.get_tuple({"configurable": {"thread_id": "kept"}}))
        self.assertEqual(self.first.get_messages("deleted"), [])
        self.assertEqual([row["thread_id"] for row in self.first.list_conversations()], ["kept"])
        with closing(sqlite3.connect(self.path)) as conn:
            self.assertEqual(conn.execute(
                "SELECT COUNT(*) FROM chat_messages WHERE thread_id = 'deleted'"
            ).fetchone()[0], 0)
        with closing(sqlite3.connect(config.CHECKPOINT_DB_PATH)) as conn:
            self.assertEqual(conn.execute(
                "SELECT COUNT(*) FROM writes WHERE thread_id = 'deleted'"
            ).fetchone()[0], 0)

    def test_delete_checks_browser_ownership_before_clearing_context(self):
        self.first.save_user_message("private", "私有对话")
        with patch.object(SqliteSaver, "delete_thread") as delete:
            with self.assertRaises(ValueError):
                self.second.delete_conversation("private")
            delete.assert_not_called()
        self.assertEqual(len(self.first.get_messages("private")), 1)

    def test_delete_without_checkpoint_and_failed_context_cleanup(self):
        self.first.save_user_message("no-context", "问题")
        self.first.delete_conversation("no-context")
        self.assertFalse(config.CHECKPOINT_DB_PATH.exists())
        self.first.save_user_message("failed", "问题")
        config.CHECKPOINT_DB_PATH.touch()
        with patch.object(SqliteSaver, "delete_thread", side_effect=sqlite3.OperationalError("清理失败")):
            with self.assertRaises(sqlite3.OperationalError):
                self.first.delete_conversation("failed")
        self.assertEqual(len(self.first.get_messages("failed")), 1)


if __name__ == "__main__":
    unittest.main()
