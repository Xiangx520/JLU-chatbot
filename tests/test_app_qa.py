"""用模拟 Streamlit 检查聊天页的网页来源展示。"""

import runpy
import sys
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from JLU_agent.schemas.structured_output import AnswerInfo, Reference


PAGE = Path(__file__).resolve().parents[1] / "streamlit_app" / "pages" / "app_qa.py"


class ChatPageTests(unittest.TestCase):
    def make_streamlit(self, messages, question=None, response=None):
        agent = Mock()
        agent.chat_with_sources.return_value = response
        ui = SimpleNamespace(
            session_state={
                "qa_messages": messages,
                "qa_thread_id": "test-thread",
                "jlu_chat_agent": agent,
            },
            title=Mock(),
            chat_message=Mock(side_effect=lambda _: nullcontext()),
            write=Mock(),
            caption=Mock(),
            link_button=Mock(),
            chat_input=Mock(return_value=question),
            spinner=Mock(side_effect=lambda _: nullcontext()),
        )
        return ui, agent

    def run_page(self, ui):
        with patch.dict(sys.modules, {"streamlit": ui}):
            runpy.run_path(str(PAGE))

    def test_history_shows_links_only_for_messages_with_web_sources(self):
        ui, agent = self.make_streamlit([
            {"role": "assistant", "content": "知识库回答", "reference": []},
            {"role": "assistant", "content": "网页回答", "reference": [
                {"title": "吉林大学官网", "url": "https://www.jlu.edu.cn/news"}
            ]},
            {"role": "assistant", "content": "后来没有联网"},
        ])

        self.run_page(ui)

        ui.write.assert_any_call("知识库回答")
        ui.write.assert_any_call("后来没有联网")
        ui.caption.assert_called_once_with("搜索来源")
        ui.link_button.assert_called_once_with(
            "吉林大学官网", "https://www.jlu.edu.cn/news"
        )
        agent.chat_with_sources.assert_not_called()

    def test_new_answer_saves_and_displays_only_current_sources(self):
        response = AnswerInfo(answer="网页回答", reference=[
            Reference(title="学校官网", url="https://www.jlu.edu.cn")
        ])
        ui, agent = self.make_streamlit([], "  最新通知  ", response)

        self.run_page(ui)

        agent.chat_with_sources.assert_called_once_with("最新通知", "test-thread")
        self.assertEqual(ui.session_state["qa_messages"][-1], {
            "role": "assistant",
            "content": "网页回答",
            "reference": [{"title": "学校官网", "url": "https://www.jlu.edu.cn"}],
        })
        ui.link_button.assert_called_once_with("学校官网", "https://www.jlu.edu.cn")

    def test_new_answer_without_web_sources_has_no_links(self):
        ui, _ = self.make_streamlit(
            [], "你好", AnswerInfo(answer="你好，同学！")
        )

        self.run_page(ui)

        self.assertEqual(ui.session_state["qa_messages"][-1]["reference"], [])
        ui.caption.assert_not_called()
        ui.link_button.assert_not_called()


if __name__ == "__main__":
    unittest.main()
