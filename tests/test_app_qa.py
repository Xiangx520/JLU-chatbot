"""使用真实临时历史数据库和模拟 Agent 检查聊天页交互。"""

import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from streamlit.testing.v1 import AppTest

from JLU_agent.agents import jlu_chat_agent
from JLU_agent.config import agent_config as config
from JLU_agent.schemas.structured_output import AnswerInfo, Reference
from JLU_agent.services.chat_history import ChatHistoryService
from JLU_agent.ui import browser_identity


PAGE = Path(__file__).resolve().parents[1] / "streamlit_app" / "pages" / "app_qa.py"


class ChatPageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "history.db"
        checkpoint_patch = patch.object(config, "CHECKPOINT_DB_PATH", Path(self.temp.name) / "checkpoint.db")
        checkpoint_patch.start()
        self.addCleanup(checkpoint_patch.stop)
        path_patch = patch.object(config, "CHAT_HISTORY_DB_PATH", self.path)
        path_patch.start()
        self.addCleanup(path_patch.stop)
        identity_patch = patch.object(
            browser_identity, "get_browser_identity",
            return_value={"browser_id": "browser-one", "error": None},
        )
        self.identity = identity_patch.start()
        self.addCleanup(identity_patch.stop)
        self.agent = Mock()
        self.agent.stream_chat_with_sources.return_value = AnswerInfo(answer="回答")
        constructor_patch = patch.object(jlu_chat_agent, "JLUChatAgent", return_value=self.agent)
        self.constructor = constructor_patch.start()
        self.addCleanup(constructor_patch.stop)
        self.service = ChatHistoryService("browser-one", self.path)
        self.app = AppTest.from_file(str(PAGE))

    def ask(self, question, app=None):
        app = app or self.app
        app.chat_input[0].set_value(question).run()
        self.assertFalse(app.exception)

    def new_button(self):
        return next(button for button in self.app.button if button.label == "新建对话")

    def test_loading_and_unavailable_storage_block_chat(self):
        self.identity.return_value = None
        self.app.run()
        self.assertFalse(self.app.chat_input)
        self.assertTrue(self.app.info)
        self.identity.return_value = {"browser_id": None, "error": "本地存储不可用"}
        self.app.run()
        self.assertEqual(self.app.error[0].value, "本地存储不可用")
        self.assertFalse(self.app.chat_input)
        self.constructor.assert_not_called()
        self.assertEqual(self.service.list_conversations(), [])

    def test_new_conversations_are_drafts_and_use_distinct_thread_ids(self):
        self.app.run()
        first_id = self.app.session_state["qa_thread_id"]
        self.assertEqual(self.service.list_conversations(), [])
        self.ask("第一个问题")
        self.new_button().click().run()
        new_id = self.app.session_state["qa_thread_id"]
        self.assertNotEqual(first_id, new_id)
        self.assertFalse(self.app.chat_message)
        self.assertEqual(len(self.service.list_conversations()), 1)
        self.app.run()
        self.assertEqual(self.app.session_state["qa_thread_id"], new_id)
        self.ask("第二个问题")
        self.assertEqual([call.args[:2] for call in self.agent.stream_chat_with_sources.call_args_list],
                         [("第一个问题", first_id), ("第二个问题", new_id)])
        self.assertEqual(len(self.service.list_conversations()), 2)
        self.constructor.assert_called_once_with()

    def test_restore_and_refresh_keep_original_thread_and_sources(self):
        self.service.save_user_message("older", "旧问题")
        sources = [{"title": "学校官网", "url": "https://www.jlu.edu.cn"}]
        self.service.save_assistant_message("older", "旧回答", sources)
        self.service.save_user_message("latest", "最近问题")
        self.app.run()
        self.assertEqual(self.app.session_state["qa_thread_id"], "latest")
        self.constructor.assert_not_called()
        self.app.button(key="history_older").click().run()
        self.assertEqual(self.app.session_state["qa_thread_id"], "older")
        self.assertEqual(self.app.session_state["qa_messages"][1]["reference"], sources)
        self.assertEqual(self.app.get("link_button")[0].proto.label, "学校官网")
        self.app.run()
        self.assertEqual(self.app.session_state["qa_thread_id"], "older")
        self.agent.stream_chat_with_sources.return_value = AnswerInfo(answer="后续回答")
        self.ask("继续追问")
        self.assertEqual(self.agent.stream_chat_with_sources.call_args.args[:2], ("继续追问", "older"))
        refreshed = AppTest.from_file(str(PAGE)).run()
        self.assertEqual(refreshed.session_state["qa_thread_id"], "older")
        self.assertEqual(len(refreshed.chat_message), 4)
        self.assertEqual(refreshed.session_state["qa_messages"][1]["reference"], sources)
        self.assertEqual(self.agent.stream_chat_with_sources.call_count, 1)

    def test_answer_saves_current_sources_and_trims_question(self):
        self.agent.stream_chat_with_sources.return_value = AnswerInfo(answer="网页回答", reference=[
            Reference(title="学校官网", url="https://www.jlu.edu.cn")
        ])
        self.app.run()
        self.ask("  最新通知  ")
        thread_id = self.app.session_state["qa_thread_id"]
        messages = self.service.get_messages(thread_id)
        self.assertEqual(messages[0]["content"], "最新通知")
        self.assertEqual(messages[1]["reference"], [{"title": "学校官网", "url": "https://www.jlu.edu.cn"}])
        self.assertEqual(self.app.get("link_button")[0].proto.label, "学校官网")

    def test_interrupted_stream_retains_question_without_partial_answer(self):
        def interrupted(question, thread_id, on_text):
            on_text("部分回答")
            raise ConnectionError("连接中断")
        self.agent.stream_chat_with_sources.side_effect = interrupted
        self.app.run()
        self.ask("你好")
        messages = self.service.get_messages(self.app.session_state["qa_thread_id"])
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "user")
        self.assertIn("回答生成失败", self.app.error[0].value)
        self.assertEqual(len(self.app.chat_message), 1)
        self.assertTrue(any((button.key or "").startswith("history_") for button in self.app.button))
        self.app.run()
        self.assertEqual(self.agent.stream_chat_with_sources.call_count, 1)

    def test_user_save_failure_does_not_call_model(self):
        self.app.run()
        with patch.object(ChatHistoryService, "save_user_message", side_effect=sqlite3.OperationalError("写入失败")):
            self.ask("你好")
        self.constructor.assert_not_called()
        self.assertEqual(self.service.list_conversations(), [])
        self.assertIn("问题未保存", self.app.error[0].value)

    def test_assistant_save_failure_does_not_retry_model(self):
        self.app.run()
        with patch.object(ChatHistoryService, "save_assistant_message", side_effect=sqlite3.OperationalError("写入失败")):
            self.ask("你好")
        self.assertIn("未保存到历史记录", self.app.error[0].value)
        self.assertEqual(len(self.service.get_messages(self.app.session_state["qa_thread_id"])), 1)
        self.app.run()
        self.assertEqual(self.agent.stream_chat_with_sources.call_count, 1)

    def test_import_current_records_once_without_initializing_agent(self):
        self.app.session_state["qa_thread_id"] = "legacy"
        self.app.session_state["qa_messages"] = [
            {"role": "user", "content": "旧问题"},
            {"role": "assistant", "content": "旧回答", "reference": []},
        ]
        self.app.run()
        self.app.run()
        refreshed = AppTest.from_file(str(PAGE)).run()
        self.assertEqual(len(self.service.get_messages("legacy")), 2)
        self.assertEqual(refreshed.session_state["qa_thread_id"], "legacy")
        self.constructor.assert_not_called()

    def test_browser_identity_change_does_not_transfer_history(self):
        self.service.save_user_message("private", "浏览器一的问题")
        self.app.run()
        self.identity.return_value = {"browser_id": "browser-two", "error": None}
        self.app.run()
        self.assertFalse(self.app.chat_message)
        self.assertEqual(ChatHistoryService("browser-two", self.path).list_conversations(), [])
        self.assertNotEqual(self.app.session_state["qa_thread_id"], "private")
        self.constructor.assert_not_called()


    def confirm_delete(self, thread_id):
        self.app.button(key=f"delete_{thread_id}").click().run()
        return next(button for button in self.app.button if button.label == "确认删除")

    def test_delete_requires_confirmation_and_can_be_cancelled(self):
        self.service.save_user_message("saved", "待删除问题")
        self.app.run()
        self.confirm_delete("saved")
        self.assertEqual(len(self.service.list_conversations()), 1)
        self.assertTrue(self.app.warning)
        next(button for button in self.app.button if button.label == "取消").click().run()
        self.assertFalse(self.app.warning)
        self.assertEqual(len(self.service.list_conversations()), 1)

    def test_delete_current_conversation_starts_empty_draft(self):
        self.service.save_user_message("saved", "待删除问题")
        self.app.run()
        self.confirm_delete("saved").click().run()
        self.assertFalse(self.app.exception)
        self.assertFalse(self.app.chat_message)
        self.assertEqual(self.service.list_conversations(), [])
        self.assertNotEqual(self.app.session_state["qa_thread_id"], "saved")
        self.assertIn("已删除", self.app.success[0].value)
        refreshed = AppTest.from_file(str(PAGE)).run()
        self.assertFalse(refreshed.chat_message)
        self.constructor.assert_not_called()

    def test_delete_other_conversation_keeps_current_selection(self):
        self.service.save_user_message("older", "旧问题")
        self.service.save_user_message("latest", "最新问题")
        self.app.run()
        self.confirm_delete("older").click().run()
        self.assertFalse(self.app.exception)
        self.assertEqual(self.app.session_state["qa_thread_id"], "latest")
        self.assertEqual(len(self.app.chat_message), 1)
        self.assertEqual([row["thread_id"] for row in self.service.list_conversations()], ["latest"])

    def test_failed_deletion_retains_history_and_selection(self):
        self.service.save_user_message("saved", "问题")
        self.app.run()
        confirm = self.confirm_delete("saved")
        with patch.object(ChatHistoryService, "delete_conversation", side_effect=sqlite3.OperationalError("删除失败")):
            confirm.click().run()
        self.assertFalse(self.app.exception)
        self.assertIn("删除对话失败", self.app.error[0].value)
        self.assertFalse(self.app.success)
        self.assertEqual(self.app.session_state["qa_thread_id"], "saved")
        self.assertEqual(len(self.service.get_messages("saved")), 1)

    def test_deletion_in_another_tab_does_not_recreate_old_thread(self):
        self.service.save_user_message("saved", "旧问题")
        self.app.run()
        self.service.delete_conversation("saved")
        self.app.run()
        self.assertNotEqual(self.app.session_state["qa_thread_id"], "saved")
        self.ask("新问题")
        self.assertEqual(self.service.get_messages("saved"), [])

    def test_stale_history_button_cannot_restore_deleted_conversation(self):
        self.service.save_user_message("older", "旧问题")
        self.service.save_user_message("latest", "最新问题")
        self.app.run()
        self.service.delete_conversation("older")
        self.app.button(key="history_older").click().run()
        self.assertFalse(self.app.exception)
        self.assertEqual(self.app.session_state["qa_thread_id"], "latest")
        self.assertIn("已删除", self.app.error[0].value)
        self.assertEqual(self.service.get_messages("older"), [])


if __name__ == "__main__":
    unittest.main()
