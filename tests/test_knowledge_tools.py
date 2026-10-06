"""使用模拟知识库验证工具封装，不连接外部服务。"""

import json
import unittest
from unittest.mock import Mock, patch

from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.tools import ToolException

from JLU_agent.config import agent_config
from JLU_agent.schemas import agent_prompts
from JLU_agent.services.RAG.file_ls import FileLoaderAndSearchService
from JLU_agent.tools import knowledge_tools

create_tools = knowledge_tools.create_tools


class KnowledgeToolsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = Mock(spec=FileLoaderAndSearchService)
        self.service.search.return_value = []

        key_patch = patch.object(agent_config, "get_deepseek_api_key", return_value="test-key")
        key_patch.start()
        self.addCleanup(key_patch.stop)
        tavily_key_patch = patch.object(
            agent_config, "get_tavily_api_key", return_value="test-tavily-key"
        )
        tavily_key_patch.start()
        self.addCleanup(tavily_key_patch.stop)

        self.tavily = Mock()
        self.tavily.invoke.return_value = {"results": []}
        tavily_patch = patch.object(
            knowledge_tools, "TavilySearch", return_value=self.tavily
        )
        self.tavily_constructor = tavily_patch.start()
        self.addCleanup(tavily_patch.stop)

        self.rewrite_model = Mock()
        self.rewrite_model.invoke.return_value = AIMessage(content="吉林大学历史")
        model_patch = patch.object(
            knowledge_tools, "init_chat_model", return_value=self.rewrite_model
        )
        self.model_initializer = model_patch.start()
        self.addCleanup(model_patch.stop)

    def test_factory_and_tool_format_multiple_sources(self) -> None:
        self.service.search.return_value = [
            (Document(page_content="片段一", metadata={"source": "introduction.txt"}), 0.75),
            (Document(page_content="片段二", metadata={"source": "history.txt"}), 0.4),
        ]
        tools = create_tools(self.service)
        self.assertEqual(
            [tool.name for tool in tools],
            ["search_knowledge_base", "search_tavily_web"],
        )
        self.tavily_constructor.assert_called_once_with(
            tavily_api_key="test-tavily-key",
            max_results=agent_config.TAVILY_MAX_RESULTS,
            search_depth=agent_config.TAVILY_SEARCH_DEPTH,
            topic="general",
            include_answer=False,
            include_raw_content=False,
            include_images=False,
            handle_tool_error=False,
        )
        self.service.search.assert_not_called()
        self.tavily.invoke.assert_not_called()
        self.rewrite_model.invoke.assert_not_called()
        self.model_initializer.assert_not_called()

        result = tools[0].invoke({"query": "吉林大学历史"})
        self.assertIn("[1]\n来源：introduction.txt\n正文：片段一\n得分：0.75", result)
        self.assertIn("[2]\n来源：history.txt\n正文：片段二\n得分：0.4", result)
        self.service.search.assert_called_once_with("吉林大学历史")
        self.rewrite_model.invoke.assert_called_once_with(
            agent_prompts.REWRITE_PROMPT.format(query="吉林大学历史")
        )
        self.model_initializer.assert_called_once_with(
            model=agent_config.REWRITE_MODEL_NAME,
            api_key="test-key",
            base_url=agent_config.CHAT_MODEL_BASE_URL,
            timeout=agent_config.CHAT_MODEL_TIMEOUT,
            max_retries=agent_config.CHAT_MODEL_MAX_RETRIES,
        )
        tools[0].invoke({"query": "吉林大学历史"})
        self.assertEqual(self.model_initializer.call_count, 2)
        self.assertEqual(self.rewrite_model.invoke.call_count, 2)

    def test_rewritten_question_is_used_only_for_search(self) -> None:
        self.rewrite_model.invoke.return_value = AIMessage(
            content="吉林大学在什么地方设有校区？"
        )
        create_tools(self.service)[0].invoke({"query": "  它在哪有校区？  "})
        self.service.search.assert_called_once_with("吉林大学在什么地方设有校区？")
        self.rewrite_model.invoke.assert_called_once_with(
            agent_prompts.REWRITE_PROMPT.format(query="  它在哪有校区？  ")
        )

    def test_blank_rewrite_uses_original_query(self) -> None:
        self.rewrite_model.invoke.return_value = AIMessage(content=" \n\t")
        with self.assertLogs(knowledge_tools.logger, level="WARNING") as log:
            create_tools(self.service)[0].invoke({"query": "吉林大学历史"})
        self.service.search.assert_called_once_with("吉林大学历史")
        self.assertNotIn("吉林大学历史", " ".join(log.output))

    def test_rewrite_error_uses_original_query_without_logging_details(self) -> None:
        self.rewrite_model.invoke.side_effect = TimeoutError("sensitive-details")
        with self.assertLogs(knowledge_tools.logger, level="WARNING") as log:
            create_tools(self.service)[0].invoke({"query": "吉林大学历史"})
        self.service.search.assert_called_once_with("吉林大学历史")
        self.assertIn("TimeoutError", " ".join(log.output))
        self.assertNotIn("sensitive-details", " ".join(log.output))
        self.assertNotIn("吉林大学历史", " ".join(log.output))

    def test_no_documents_or_blank_documents_have_clear_message(self) -> None:
        tool = create_tools(self.service)[0]
        for documents in ([], [(Document(page_content=" \n\t"), 0.5)]):
            self.service.search.return_value = documents
            self.assertIn("知识库中暂无足够资料", tool.invoke({"query": "问题"}))

    def test_missing_source_is_not_invented(self) -> None:
        self.service.search.return_value = [(Document(page_content="正文"), 0.5)]
        result = create_tools(self.service)[0].invoke({"query": "问题"})
        self.assertIn("来源：未知来源", result)

    def test_zero_normalized_score_is_still_shown(self) -> None:
        self.service.search.return_value = [(
            Document(page_content="候选片段", metadata={"source": "history.txt"}),
            0.0,
        )]
        result = create_tools(self.service)[0].invoke({"query": "问题"})
        self.assertIn("来源：history.txt\n正文：候选片段\n得分：0.0", result)

    def test_web_search_keeps_valid_results_and_skips_unusable_links(self) -> None:
        self.tavily.invoke.return_value = {"results": [
            {"title": "学校官网", "url": "https://www.jlu.edu.cn/news", "content": "招生通知"},
            {"title": "重复", "url": "https://www.jlu.edu.cn/news", "content": "重复摘要"},
            {"title": "无效协议", "url": "javascript:alert(1)", "content": "不能显示"},
            {"title": "缺摘要", "url": "https://example.com", "content": " "},
            {"title": "无效地址", "url": "https://", "content": "不能显示"},
            {"title": "无效 IPv6", "url": "https://[", "content": "不能显示"},
            {"title": "第二条", "url": "http://example.com", "content": "其他内容"},
        ]}
        result = json.loads(create_tools(self.service)[1].invoke({"query": "  招生通知  "}))
        self.tavily.invoke.assert_called_once_with({"query": "招生通知"})
        self.assertEqual(result, {"results": [
            {"title": "学校官网", "url": "https://www.jlu.edu.cn/news", "content": "招生通知"},
            {"title": "重复", "url": "https://www.jlu.edu.cn/news", "content": "重复摘要"},
            {"title": "第二条", "url": "http://example.com", "content": "其他内容"},
        ], "message": ""})
        self.service.search.assert_not_called()

    def test_web_search_empty_and_blank_query(self) -> None:
        tool = create_tools(self.service)[1]
        self.assertEqual(
            json.loads(tool.invoke({"query": "问题"})),
            {"results": [], "message": "未找到相关网页资料。"},
        )
        with self.assertRaisesRegex(ValueError, "搜索问题不能为空"):
            tool.invoke({"query": " \n "})
        self.tavily.invoke.assert_called_once_with({"query": "问题"})

    def test_tavily_no_results_exception_is_not_reported_as_outage(self) -> None:
        self.tavily.invoke.side_effect = ToolException(
            "No search results found for 'private-query'"
        )
        result = create_tools(self.service)[1].invoke({"query": "private-query"})
        self.assertEqual(
            json.loads(result), {"results": [], "message": "未找到相关网页资料。"}
        )
        self.assertNotIn("private-query", result)

    def test_web_search_wrapped_service_error_raises_without_details(self) -> None:
        tool = create_tools(self.service)[1]
        self.tavily.invoke.return_value = {"error": ValueError("secret-response")}
        with self.assertRaisesRegex(RuntimeError, "网页搜索失败") as caught:
            tool.invoke({"query": "private-query"})
        self.assertNotIn("secret-response", str(caught.exception))
        self.assertNotIn("private-query", str(caught.exception))

    def test_web_search_direct_error_propagates(self) -> None:
        error = TimeoutError("模拟网络超时")
        self.tavily.invoke.side_effect = error
        with self.assertRaises(TimeoutError) as caught:
            create_tools(self.service)[1].invoke({"query": "问题"})
        self.assertIs(caught.exception, error)


if __name__ == "__main__":
    unittest.main()
