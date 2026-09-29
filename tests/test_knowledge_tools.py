"""使用模拟 Chroma 验证检索与工具封装，不连接实际知识库。"""

import unittest
from unittest.mock import Mock, patch

from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from JLU_agent.config import agent_config
from JLU_agent.config import chroma_config as config
from JLU_agent.schemas import agent_prompts
from JLU_agent.services.RAG.file_ls import FileLoaderAndSearchService
from JLU_agent.services.RAG.vector_store import VectorStoreService
from JLU_agent.tools import knowledge_tools

create_tools = knowledge_tools.create_tools


class KnowledgeToolsTests(unittest.TestCase):
    def setUp(self) -> None:
        # 跳过真实构造过程，避免读取密钥或打开项目知识库。
        self.vector_store = object.__new__(VectorStoreService)
        self.vector_store.chroma = Mock()
        self.vector_store.chroma.similarity_search_with_score.return_value = []
        self.service = Mock(spec=FileLoaderAndSearchService)
        self.service.search.return_value = []

        key_patch = patch.object(agent_config, "get_deepseek_api_key", return_value="test-key")
        key_patch.start()
        self.addCleanup(key_patch.stop)

        self.rewrite_model = Mock()
        self.rewrite_model.invoke.return_value = AIMessage(content="吉林大学历史")
        model_patch = patch.object(
            knowledge_tools, "init_chat_model", return_value=self.rewrite_model
        )
        self.model_initializer = model_patch.start()
        self.addCleanup(model_patch.stop)

    def test_search_uses_configured_limit_and_preserves_documents(self) -> None:
        documents = [(
            Document(
                page_content="吉林大学始建于1946年。",
                metadata={"source": "introduction.txt"},
            ),
            0.1,
        )]
        self.vector_store.chroma.similarity_search_with_score.return_value = documents
        with patch.object(config, "K", 3):
            result = self.vector_store.search("  吉林大学始建于哪一年？  ")
        self.assertIs(result, documents)
        self.vector_store.chroma.similarity_search_with_score.assert_called_once_with(
            "吉林大学始建于哪一年？", k=3
        )

    def test_invalid_queries_do_not_search(self) -> None:
        for query in (None, 123):
            with self.subTest(query=query), self.assertRaises(TypeError):
                self.vector_store.search(query)
        for query in ("", " \n\t"):
            with self.subTest(query=query), self.assertRaises(ValueError):
                self.vector_store.search(query)
        self.vector_store.chroma.similarity_search_with_score.assert_not_called()

    def test_factory_and_tool_format_multiple_sources(self) -> None:
        self.service.search.return_value = [
            (Document(page_content="片段一", metadata={"source": "introduction.txt"}), 0.75),
            (Document(page_content="片段二", metadata={"source": "history.txt"}), 0.4),
        ]
        tools = create_tools(self.service)
        self.assertEqual([tool.name for tool in tools], ["search_knowledge_base"])
        self.service.search.assert_not_called()
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


if __name__ == "__main__":
    unittest.main()
