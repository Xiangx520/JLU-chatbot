"""使用本地模拟模型验证对话流程，不请求真实 API。"""

import importlib
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import Mock, patch

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatResult
from pydantic import Field

from JLU_agent.agents import jlu_chat_agent as agent_module
from JLU_agent.config import agent_config as config
from JLU_agent.config import chroma_config
from JLU_agent.schemas.agent_prompts import CHAT_MODEL_SYSTEM_PROMPT
from JLU_agent.schemas import agent_prompts
from JLU_agent.services.RAG import file_ls, vector_store
from JLU_agent.tools import knowledge_tools


class RecordingChatModel(FakeMessagesListChatModel):
    """记录模型实际收到的消息，用于检查提示词和上下文。"""

    received_messages: list[list[BaseMessage]] = Field(default_factory=list)
    bound_tool_names: list[str] = Field(default_factory=list)

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        """记录注册的工具；实际工具执行仍交给真实的 Agent 图。"""
        self.bound_tool_names = [tool.name for tool in tools]
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.received_messages.append(list(messages))
        return super()._generate(
            messages, stop=stop, run_manager=run_manager, **kwargs
        )


class JLUChatAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        key_patch = patch.object(config, "get_deepseek_api_key", return_value="test-key")
        key_patch.start()
        self.addCleanup(key_patch.stop)
        tavily_key_patch = patch.object(
            config, "get_tavily_api_key", return_value="test-tavily-key"
        )
        tavily_key_patch.start()
        self.addCleanup(tavily_key_patch.stop)

        self.tavily = Mock()
        self.tavily.invoke.return_value = {"results": []}
        tavily_patch = patch.object(
            knowledge_tools, "TavilySearch", return_value=self.tavily
        )
        tavily_patch.start()
        self.addCleanup(tavily_patch.stop)

        tracing_patch = patch.dict(
            os.environ,
            {"LANGSMITH_TRACING": "false", "LANGCHAIN_TRACING_V2": "false"},
        )
        tracing_patch.start()
        self.addCleanup(tracing_patch.stop)

        temp_dir = TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        checkpoint_dir = Path(temp_dir.name)
        for name, value in (
            ("CHECKPOINT_DIR", checkpoint_dir),
            ("CHECKPOINT_DB_PATH", checkpoint_dir / "checkpoint.db"),
        ):
            config_patch = patch.object(config, name, value)
            config_patch.start()
            self.addCleanup(config_patch.stop)

        self.file_ls_service = Mock(spec=file_ls.FileLoaderAndSearchService)
        self.file_ls_service.search.return_value = []
        service_patch = patch.object(
            agent_module, "FileLoaderAndSearchService", return_value=self.file_ls_service
        )
        self.service_constructor = service_patch.start()
        self.addCleanup(service_patch.stop)

        self.rewrite_model = Mock()
        self.rewrite_model.invoke.return_value = AIMessage(
            content="吉林大学始建于哪一年？"
        )
        rewrite_patch = patch.object(
            knowledge_tools, "init_chat_model", return_value=self.rewrite_model
        )
        self.rewrite_initializer = rewrite_patch.start()
        self.addCleanup(rewrite_patch.stop)

    def make_agent(
        self, responses: list[BaseMessage]
    ) -> tuple[agent_module.JLUChatAgent, RecordingChatModel]:
        model = RecordingChatModel(responses=responses)
        with patch.object(agent_module, "init_chat_model", return_value=model):
            agent = agent_module.JLUChatAgent()
        self.addCleanup(agent.agent.checkpointer.conn.close)
        return agent, model

    def test_chat_returns_text_and_passes_system_prompt(self) -> None:
        agent, model = self.make_agent([AIMessage(content="你好，同学！")])

        self.assertEqual(agent.chat("  你好  ", "thread-1"), "你好，同学！")
        messages = model.received_messages[0]
        self.assertEqual([message.type for message in messages], ["system", "human"])
        self.assertEqual(messages[0].content, CHAT_MODEL_SYSTEM_PROMPT)
        self.assertEqual(messages[1].content, "你好")
        self.assertEqual(
            model.bound_tool_names, ["search_knowledge_base", "search_tavily_web"]
        )
        self.service_constructor.assert_called_once_with()
        self.file_ls_service.search.assert_not_called()
        self.tavily.invoke.assert_not_called()
        self.rewrite_initializer.assert_not_called()
        self.rewrite_model.invoke.assert_not_called()

    def test_same_thread_keeps_history(self) -> None:
        agent, model = self.make_agent(
            [AIMessage(content="第一条回答"), AIMessage(content="第二条回答")]
        )

        agent.chat("第一个问题", "thread-1")
        agent.chat("第二个问题", "thread-1")
        self.assertEqual(
            [message.content for message in model.received_messages[1]],
            [CHAT_MODEL_SYSTEM_PROMPT, "第一个问题", "第一条回答", "第二个问题"],
        )
        self.service_constructor.assert_called_once_with()

    def test_different_threads_do_not_share_history(self) -> None:
        agent, model = self.make_agent(
            [AIMessage(content="第一条回答"), AIMessage(content="第二条回答")]
        )

        self.assertEqual(agent.chat("第一个问题", "thread-1"), "第一条回答")
        self.assertEqual(agent.chat("第二个问题", "thread-2"), "第二条回答")
        self.assertEqual(len(model.received_messages), 2)
        self.assertEqual(
            [message.content for message in model.received_messages[1]],
            [CHAT_MODEL_SYSTEM_PROMPT, "第二个问题"],
        )

    def test_invalid_input_does_not_call_model(self) -> None:
        agent, model = self.make_agent([AIMessage(content="不会被调用")])

        for message in ("", "  \n\t"):
            with self.subTest(message=message), self.assertRaises(ValueError):
                agent.chat(message, "thread-1")
        for message in (None, 123, ["你好"]):
            with self.subTest(message=message), self.assertRaises(TypeError):
                agent.chat(message, "thread-1")
        self.assertEqual(model.received_messages, [])
        self.file_ls_service.search.assert_not_called()

    def test_text_blocks_are_returned_as_a_string(self) -> None:
        reply = AIMessage(
            content=[
                {"type": "text", "text": "你好，"},
                {"type": "text", "text": "同学！"},
            ]
        )
        agent, _ = self.make_agent([reply])
        self.assertEqual(agent.chat("你好", "thread-1"), "你好，同学！")

    def test_empty_answer_raises_error(self) -> None:
        agent, _ = self.make_agent([AIMessage(content=" \n\t")])
        with self.assertRaisesRegex(RuntimeError, "没有返回有效的回答"):
            agent.chat("你好", "thread-1")

    def test_call_error_is_preserved(self) -> None:
        agent, _ = self.make_agent([AIMessage(content="不会返回")])
        error = ConnectionError("模拟网络连接失败")
        with patch.object(agent.agent, "invoke", side_effect=error):
            with self.assertRaises(ConnectionError) as context:
                agent.chat("你好", "thread-1")
        self.assertIs(context.exception, error)

    def knowledge_request(self) -> AIMessage:
        return AIMessage(
            content="",
            tool_calls=[{
                "name": "search_knowledge_base",
                "args": {"query": "吉林大学始建于哪一年？"},
                "id": "knowledge-call-1",
                "type": "tool_call",
            }],
        )

    def test_agent_calls_tool_and_receives_knowledge(self) -> None:
        self.file_ls_service.search.return_value = [(
            Document(
                page_content="吉林大学始建于1946年。",
                metadata={"source": "introduction.txt"},
            ),
            0.75,
        )]
        self.rewrite_model.invoke.return_value = AIMessage(
            content="吉林大学的建校年份是什么？"
        )
        answer = "吉林大学始建于1946年。（来源：introduction.txt）"
        agent, model = self.make_agent([
            self.knowledge_request(), AIMessage(content=answer),
        ])

        self.assertEqual(agent.chat("吉林大学始建于哪一年？", "rag-thread"), answer)
        self.file_ls_service.search.assert_called_once_with("吉林大学的建校年份是什么？")
        self.rewrite_model.invoke.assert_called_once_with(
            agent_prompts.REWRITE_PROMPT.format(query="吉林大学始建于哪一年？")
        )
        self.assertEqual(model.received_messages[0][-1].content, "吉林大学始建于哪一年？")
        self.assertEqual(len(model.received_messages), 2)
        tool_reply = model.received_messages[1][-1]
        self.assertIsInstance(tool_reply, ToolMessage)
        self.assertEqual(tool_reply.tool_call_id, "knowledge-call-1")
        self.assertIn("1946", tool_reply.content)
        self.assertIn("来源：introduction.txt", tool_reply.content)

    def test_empty_search_result_reaches_model(self) -> None:
        answer = "知识库中暂无足够资料。"
        agent, model = self.make_agent([
            self.knowledge_request(), AIMessage(content=answer),
        ])

        result = agent.chat_with_sources("吉林大学始建于哪一年？", "empty-thread")
        self.assertEqual(result.answer, answer)
        self.assertEqual(result.reference, [])
        self.assertIn("知识库中暂无足够资料", model.received_messages[1][-1].content)

    def test_web_fallback_sources_belong_only_to_current_turn(self) -> None:
        self.tavily.invoke.return_value = {"results": [
            {
                "title": "吉林大学官网",
                "url": "https://www.jlu.edu.cn/news",
                "content": "最新通知",
            },
        ]}
        web_request = AIMessage(content="", tool_calls=[{
            "name": "search_tavily_web",
            "args": {"query": "吉林大学最新通知"},
            "id": "web-call-1",
            "type": "tool_call",
        }])
        agent, model = self.make_agent([
            self.knowledge_request(),
            web_request,
            AIMessage(content="官网发布了最新通知。[来源](https://www.jlu.edu.cn/news)"),
            AIMessage(content="不客气。"),
        ])

        result = agent.chat_with_sources("吉林大学最新通知", "web-thread")
        self.assertEqual(
            [(source.title, source.url) for source in result.reference],
            [("吉林大学官网", "https://www.jlu.edu.cn/news")],
        )
        self.assertIn("知识库中暂无足够资料", model.received_messages[1][-1].content)
        self.assertIsInstance(model.received_messages[2][-1], ToolMessage)
        self.assertEqual(model.received_messages[2][-1].name, "search_tavily_web")
        self.tavily.invoke.assert_called_once_with({"query": "吉林大学最新通知"})

        next_result = agent.chat_with_sources("谢谢", "web-thread")
        self.assertEqual(next_result.answer, "不客气。")
        self.assertEqual(next_result.reference, [])
        self.assertEqual(len(model.received_messages), 4)

    def test_search_error_is_preserved(self) -> None:
        error = ConnectionError("模拟知识库检索失败")
        self.file_ls_service.search.side_effect = error
        agent, model = self.make_agent([self.knowledge_request()])

        with self.assertRaises(ConnectionError) as context:
            agent.chat("吉林大学始建于哪一年？", "error-thread")
        self.assertIs(context.exception, error)
        self.assertEqual(len(model.received_messages), 1)

    def test_summarization_still_runs_with_tools(self) -> None:
        self.file_ls_service.search.return_value = [(
            Document(
                page_content="吉林大学始建于1946年。",
                metadata={"source": "introduction.txt"},
            ),
            0.75,
        )]
        with patch.object(config, "SUMMARIZE_TRIGGER_MESSAGES", 6), patch.object(
            config, "SUMMARIZE_KEEP_MESSAGES", 2
        ):
            agent, model = self.make_agent([
                self.knowledge_request(),
                AIMessage(content="1946年。（来源：introduction.txt）"),
                AIMessage(content="不客气！"),
                AIMessage(content="已查明始建于1946年，来源 introduction.txt。"),
                AIMessage(content="再见！"),
            ])
        agent.chat("吉林大学始建于哪一年？", "summary-thread")
        agent.chat("谢谢", "summary-thread")
        self.assertEqual(agent.chat("再见", "summary-thread"), "再见！")
        self.assertEqual(len(model.received_messages), 5)
        summary_input = str(model.received_messages[3][0].content)
        self.assertIn("对应知识库文件名或网页标题与链接", summary_input)
        self.assertIn("introduction.txt", summary_input)
        self.assertIn("已查明始建于1946年", str(model.received_messages[4]))

    def test_import_does_not_load_configuration(self) -> None:
        with (
            patch.object(config, "get_deepseek_api_key", side_effect=AssertionError),
            patch.object(chroma_config, "get_dashscope_api_key", side_effect=AssertionError),
            patch.object(vector_store, "Chroma", side_effect=AssertionError),
            patch.object(agent_module.sqlite3, "connect", side_effect=AssertionError),
        ):
            importlib.reload(knowledge_tools)
            importlib.reload(agent_module)


class AgentConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        temp_dir = TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        self.env_file = Path(temp_dir.name) / ".env"

        env_patch = patch.dict(os.environ, {}, clear=True)
        env_patch.start()
        self.addCleanup(env_patch.stop)

        file_patch = patch.object(config, "ENV_FILE", self.env_file)
        file_patch.start()
        self.addCleanup(file_patch.stop)

    def test_reads_env_file_without_exporting_other_settings(self) -> None:
        self.env_file.write_text(
            "DEEPSEEK_API_KEY=file-key\nLANGSMITH_TRACING=true\n", encoding="utf-8"
        )
        self.assertEqual(config.get_deepseek_api_key(), "file-key")
        self.assertNotIn("DEEPSEEK_API_KEY", os.environ)
        self.assertNotIn("LANGSMITH_TRACING", os.environ)

    def test_environment_variable_has_priority(self) -> None:
        self.env_file.write_text("DEEPSEEK_API_KEY=file-key\n", encoding="utf-8")
        os.environ["DEEPSEEK_API_KEY"] = " environment-key "
        self.assertEqual(config.get_deepseek_api_key(), "environment-key")

    def test_missing_key_fails_at_initialization(self) -> None:
        with self.assertRaisesRegex(ValueError, "未配置 DEEPSEEK_API_KEY"):
            agent_module.JLUChatAgent()

    def test_blank_key_in_env_file_is_rejected(self) -> None:
        self.env_file.write_text("DEEPSEEK_API_KEY=\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "未配置 DEEPSEEK_API_KEY"):
            config.get_deepseek_api_key()

    def test_blank_environment_variable_does_not_fall_back_to_file(self) -> None:
        self.env_file.write_text("DEEPSEEK_API_KEY=file-key\n", encoding="utf-8")
        os.environ["DEEPSEEK_API_KEY"] = "  "
        with self.assertRaisesRegex(ValueError, "未配置 DEEPSEEK_API_KEY"):
            config.get_deepseek_api_key()

    def test_tavily_key_uses_file_and_environment_priority(self) -> None:
        self.env_file.write_text(
            "TAVILY_API_KEY=file-key\nOTHER_SECRET=untouched\n", encoding="utf-8"
        )
        self.assertEqual(config.get_tavily_api_key(), "file-key")
        self.assertNotIn("TAVILY_API_KEY", os.environ)
        self.assertNotIn("OTHER_SECRET", os.environ)

        os.environ["TAVILY_API_KEY"] = " environment-key "
        self.assertEqual(config.get_tavily_api_key(), "environment-key")

    def test_missing_or_blank_tavily_key_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "未配置 TAVILY_API_KEY"):
            config.get_tavily_api_key()

        self.env_file.write_text("TAVILY_API_KEY=\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "未配置 TAVILY_API_KEY"):
            config.get_tavily_api_key()

        self.env_file.write_text("TAVILY_API_KEY=file-key\n", encoding="utf-8")
        os.environ["TAVILY_API_KEY"] = "  "
        with self.assertRaisesRegex(ValueError, "未配置 TAVILY_API_KEY"):
            config.get_tavily_api_key()

    def test_missing_tavily_key_fails_at_agent_initialization(self) -> None:
        self.env_file.write_text("DEEPSEEK_API_KEY=test-key\n", encoding="utf-8")
        with (
            patch.object(agent_module, "init_chat_model", return_value=Mock()),
            patch.object(agent_module, "FileLoaderAndSearchService", return_value=Mock()),
        ):
            with self.assertRaisesRegex(ValueError, "未配置 TAVILY_API_KEY"):
                agent_module.JLUChatAgent()


if __name__ == "__main__":
    unittest.main()
