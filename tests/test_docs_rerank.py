"""用模拟检索结果和模型验证 RRF 与交叉编码重排。"""

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from langchain_core.documents import Document

from JLU_agent.services.RAG.docs_reranker import (
    cross_encoder_rerank,
    get_cross_encoder,
    reciprocal_rank_fusion,
    rerank,
)
from JLU_agent.services.RAG.file_ls import FileLoaderAndSearchService


class RerankTests(unittest.TestCase):
    def test_rrf_uses_rank_and_deduplicates_ids(self) -> None:
        shared = Document(id="shared", page_content="共同结果")
        vector_only = Document(id="vector", page_content="向量结果")
        bm25_only = Document(id="bm25", page_content="关键词结果")

        results = reciprocal_rank_fusion([
            [(shared, 100.0), (vector_only, -5.0)],
            [(bm25_only, 0.1), (shared, 999.0)],
        ])

        self.assertEqual(len(results), 3)
        self.assertEqual([doc.id for doc in results], ["shared", "bm25", "vector"])

    def test_cross_encoder_keeps_negative_scores_and_metadata(self) -> None:
        documents = [
            Document(id=str(index), page_content=f"正文{index}", metadata={"source": f"来源{index}.txt"})
            for index in range(4)
        ]
        model = Mock()
        model.predict.return_value = [-2.0, 0.0, 4.0, 1.0]

        with patch("JLU_agent.services.RAG.docs_reranker.get_cross_encoder", return_value=model):
            results = cross_encoder_rerank("吉林大学", documents)

        model.predict.assert_called_once_with(
            [("吉林大学", document.page_content) for document in documents]
        )
        self.assertEqual([(doc.id, score) for doc, score in results], [
            ("2", 4.0), ("3", 1.0), ("1", 0.0)
        ])
        self.assertEqual(results[0][0].metadata["source"], "来源2.txt")
        self.assertEqual(cross_encoder_rerank("吉林大学", [], top_k=3), [])

    def test_single_negative_result_is_not_dropped(self) -> None:
        document = Document(id="one", page_content="资料")
        model = Mock()
        model.predict.return_value = [-3.5]
        with patch("JLU_agent.services.RAG.docs_reranker.get_cross_encoder", return_value=model):
            self.assertEqual(cross_encoder_rerank("问题", [document]), [(document, -3.5)])

    def test_all_six_candidates_reach_model_before_top_three(self) -> None:
        vector_results = [
            (Document(id=f"vector-{index}", page_content=f"向量{index}"), float(index))
            for index in range(3)
        ]
        bm25_results = [
            (Document(id=f"bm25-{index}", page_content=f"索引{index}"), float(index))
            for index in range(3)
        ]
        model = Mock()
        model.predict.return_value = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0]

        with patch("JLU_agent.services.RAG.docs_reranker.get_cross_encoder", return_value=model):
            results = rerank("问题", vector_results, bm25_results)

        self.assertEqual(len(model.predict.call_args.args[0]), 6)
        self.assertEqual(len(results), 3)
        self.assertEqual([score for _, score in results], [2.0, 1.0, 0.0])

    def test_model_is_loaded_once(self) -> None:
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))
        fake_encoder = Mock(return_value=object())
        fake_sentence_transformers = SimpleNamespace(CrossEncoder=fake_encoder)
        get_cross_encoder.cache_clear()
        try:
            with patch.dict(sys.modules, {
                "torch": fake_torch,
                "sentence_transformers": fake_sentence_transformers,
            }):
                self.assertIs(get_cross_encoder(), get_cross_encoder())
        finally:
            get_cross_encoder.cache_clear()

        fake_encoder.assert_called_once_with(
            model_name_or_path="Qwen/Qwen3-Reranker-0.6B", device="cuda"
        )

    def test_model_errors_reach_caller(self) -> None:
        document = Document(id="one", page_content="资料")
        model = Mock()
        model.predict.side_effect = RuntimeError("推理失败")
        with patch("JLU_agent.services.RAG.docs_reranker.get_cross_encoder", return_value=model):
            with self.assertRaisesRegex(RuntimeError, "推理失败"):
                rerank("问题", [(document, 0.1)], [])

    def test_empty_results_and_search_service_calls_both_stores(self) -> None:
        with patch("JLU_agent.services.RAG.docs_reranker.get_cross_encoder") as load_model:
            self.assertEqual(rerank("吉林大学", [], []), [])
            load_model.assert_not_called()

        service = object.__new__(FileLoaderAndSearchService)
        near = Document(id="near", page_content="吉林大学历史")
        service.vectorStoreService = SimpleNamespace(search=Mock(return_value=[(near, 0.1)]))
        service.indexStoreService = SimpleNamespace(search=Mock(return_value=[]))
        model = Mock()
        model.predict.return_value = [2.0]
        with patch("JLU_agent.services.RAG.docs_reranker.get_cross_encoder", return_value=model):
            results = service.search("吉林大学")

        service.vectorStoreService.search.assert_called_once_with("吉林大学")
        service.indexStoreService.search.assert_called_once_with("吉林大学")
        model.predict.assert_called_once_with([("吉林大学", "吉林大学历史")])
        self.assertEqual([doc.id for doc, _ in results], ["near"])


if __name__ == "__main__":
    unittest.main()
