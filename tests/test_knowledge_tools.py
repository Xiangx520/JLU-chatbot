"""使用模拟 Chroma 验证检索与工具封装，不连接实际知识库。"""

import unittest
from unittest.mock import Mock, patch

from langchain_core.documents import Document

from JLU_agent.config import chroma_config as config
from JLU_agent.services.RAG.vector_store import VectorStoreService
from JLU_agent.tools.knowledge_tools import create_tools


class KnowledgeToolsTests(unittest.TestCase):
    def setUp(self) -> None:
        # 跳过真实构造过程，避免读取密钥或打开项目知识库。
        self.service = object.__new__(VectorStoreService)
        self.service.chroma = Mock()
        self.service.chroma.similarity_search.return_value = []

    def test_search_uses_configured_limit_and_preserves_documents(self) -> None:
        documents = [Document(
            page_content="吉林大学始建于1946年。",
            metadata={"source": "introduction.txt"},
        )]
        self.service.chroma.similarity_search.return_value = documents
        with patch.object(config, "K", 3):
            result = self.service.search("  吉林大学始建于哪一年？  ")
        self.assertIs(result, documents)
        self.service.chroma.similarity_search.assert_called_once_with(
            "吉林大学始建于哪一年？", k=3
        )

    def test_invalid_queries_do_not_search(self) -> None:
        for query in (None, 123):
            with self.subTest(query=query), self.assertRaises(TypeError):
                self.service.search(query)
        for query in ("", " \n\t"):
            with self.subTest(query=query), self.assertRaises(ValueError):
                self.service.search(query)
        self.service.chroma.similarity_search.assert_not_called()

    def test_factory_and_tool_format_multiple_sources(self) -> None:
        self.service.chroma.similarity_search.return_value = [
            Document(page_content="片段一", metadata={"source": "introduction.txt"}),
            Document(page_content="片段二", metadata={"source": "history.txt"}),
        ]
        tools = create_tools(self.service)
        self.assertEqual([tool.name for tool in tools], ["search_knowledge_base"])
        self.service.chroma.similarity_search.assert_not_called()

        result = tools[0].invoke({"query": "吉林大学历史"})
        self.assertIn("[1]\n来源：introduction.txt\n正文：片段一", result)
        self.assertIn("[2]\n来源：history.txt\n正文：片段二", result)
        self.service.chroma.similarity_search.assert_called_once_with("吉林大学历史", k=config.K)

    def test_no_documents_or_blank_documents_have_clear_message(self) -> None:
        tool = create_tools(self.service)[0]
        for documents in ([], [Document(page_content=" \n\t")]):
            self.service.chroma.similarity_search.return_value = documents
            self.assertIn("知识库中暂无足够资料", tool.invoke({"query": "问题"}))

    def test_missing_source_is_not_invented(self) -> None:
        self.service.chroma.similarity_search.return_value = [Document(page_content="正文")]
        result = create_tools(self.service)[0].invoke({"query": "问题"})
        self.assertIn("来源：未知来源", result)


if __name__ == "__main__":
    unittest.main()
