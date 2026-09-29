"""用模拟检索结果验证双库融合，不连接真实知识库。"""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from langchain_core.documents import Document

from JLU_agent.services.RAG.docs_rerank import rerank
from JLU_agent.services.RAG.file_ls import FileLoaderAndSearchService


class RerankTests(unittest.TestCase):
    def test_chroma_smaller_distance_ranks_higher(self) -> None:
        near = Document(id="near", page_content="相关内容")
        far = Document(id="far", page_content="较远内容")

        results = rerank([(far, 0.9), (near, 0.1)], [])

        self.assertEqual([doc.id for doc, _ in results], ["near", "far"])
        self.assertGreater(results[0][1], results[1][1])

    def test_bm25_higher_score_ranks_higher(self) -> None:
        high = Document(id="high", page_content="相关内容")
        low = Document(id="low", page_content="较少匹配")

        results = rerank([], [(low, 1.0), (high, 4.0)])

        self.assertEqual([doc.id for doc, _ in results], ["high", "low"])

    def test_shared_document_is_returned_once_with_combined_score(self) -> None:
        shared = Document(id="shared", page_content="共同结果")
        vector_only = Document(id="vector", page_content="向量结果")
        bm25_only = Document(id="bm25", page_content="关键词结果")

        results = rerank(
            [(shared, 0.1), (vector_only, 0.8)],
            [(shared, 5.0), (bm25_only, 1.0)],
        )

        self.assertEqual(len(results), 3)
        self.assertEqual(results[0][0].id, "shared")
        self.assertEqual({doc.id for doc, _ in results}, {"shared", "vector", "bm25"})
        self.assertIs(type(results[0][1]), float)

    def test_empty_results_and_search_service_calls_both_stores(self) -> None:
        self.assertEqual(rerank([], []), [])

        service = object.__new__(FileLoaderAndSearchService)
        near = Document(id="near", page_content="吉林大学历史")
        service.vectorStoreService = SimpleNamespace(search=Mock(return_value=[(near, 0.1)]))
        service.indexStoreService = SimpleNamespace(search=Mock(return_value=[]))

        results = service.search("吉林大学")

        service.vectorStoreService.search.assert_called_once_with("吉林大学")
        service.indexStoreService.search.assert_called_once_with("吉林大学")
        self.assertEqual([doc.id for doc, _ in results], ["near"])


if __name__ == "__main__":
    unittest.main()
