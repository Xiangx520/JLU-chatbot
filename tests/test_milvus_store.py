"""模拟 Milvus 客户端，验证持久化接口和混合检索参数。"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from langchain_core.documents import Document
from pymilvus import FunctionType

from JLU_agent.config import rag_config as config
from JLU_agent.services.RAG import milvus_store
from JLU_agent.services.RAG.milvus_store import MilvusStoreService, OUTPUT_FIELDS


class MilvusConfigTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.env_file = Path(temp.name) / ".env"
        for patcher in [patch.dict(os.environ, {}, clear=True), patch.object(config, "ENV_FILE", self.env_file)]:
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_defaults_and_env_file_without_export(self):
        self.assertEqual(config.get_milvus_connection_args(), {
            "uri": "http://localhost:19530", "db_name": "default", "timeout": 30.0,
        })
        self.assertEqual(config.get_milvus_collection_name(), "jlu_knowledge")
        self.env_file.write_text(
            "MILVUS_URI=https://milvus.example.com\nMILVUS_TOKEN=file-token\n"
            "MILVUS_DB_NAME=campus\nMILVUS_COLLECTION_NAME=documents\nDASHSCOPE_API_KEY=file-key\n",
            encoding="utf-8",
        )
        self.assertEqual(config.get_milvus_connection_args()["token"], "file-token")
        self.assertEqual(config.get_milvus_connection_args()["db_name"], "campus")
        self.assertEqual(config.get_milvus_collection_name(), "documents")
        self.assertEqual(config.get_dashscope_api_key(), "file-key")
        self.assertNotIn("MILVUS_TOKEN", os.environ)
        self.assertNotIn("DASHSCOPE_API_KEY", os.environ)

    def test_environment_precedence_including_empty_token(self):
        self.env_file.write_text("MILVUS_TOKEN=file-token\nMILVUS_URI=http://file:19530\n", encoding="utf-8")
        os.environ.update(MILVUS_URI=" http://environment:19530 ", MILVUS_TOKEN="")
        args = config.get_milvus_connection_args()
        self.assertEqual(args["uri"], "http://environment:19530")
        self.assertNotIn("token", args)

    def test_bad_configuration_and_missing_key(self):
        for uri in ["", "./milvus.db", "https://", "http://localhost:bad", "http://["]:
            with self.subTest(uri=uri), patch.dict(os.environ, {"MILVUS_URI": uri}):
                with self.assertRaisesRegex(ValueError, "MILVUS_URI"):
                    config.get_milvus_connection_args()
        for key in ["MILVUS_DB_NAME", "MILVUS_COLLECTION_NAME"]:
            with patch.dict(os.environ, {key: "invalid name"}):
                with self.assertRaisesRegex(ValueError, key):
                    if key == "MILVUS_DB_NAME":
                        config.get_milvus_connection_args()
                    else:
                        config.get_milvus_collection_name()
        with self.assertRaisesRegex(ValueError, "DASHSCOPE_API_KEY"):
            config.get_dashscope_api_key()


class MilvusStoreTests(unittest.TestCase):
    def setUp(self):
        self.backend = Mock()
        self.backend.fields = OUTPUT_FIELDS + ["dense", "sparse"]
        self.backend.client.has_collection.return_value = True
        self.embeddings = Mock()
        patcher = patch.object(milvus_store, "Milvus", return_value=self.backend)
        self.constructor = patcher.start()
        self.addCleanup(patcher.stop)
        self.service = MilvusStoreService(
            embeddings=self.embeddings,
            connection_args={"uri": "http://localhost:19530"},
            collection_name="test_knowledge",
        )

    def test_constructor_uses_one_collection_and_chinese_bm25(self):
        args = self.constructor.call_args.kwargs
        self.assertFalse(args["auto_id"])
        self.assertFalse(args["drop_old"])
        self.assertNotIn("timeout", args)
        self.assertEqual(args["consistency_level"], "Strong")
        self.assertEqual(args["vector_field"], ["dense", "sparse"])
        self.assertEqual(args["builtin_function"].analyzer_params, {"type": "chinese"})
        self.assertEqual(args["builtin_function"].function.type, FunctionType.BM25)
        self.assertEqual(args["index_params"][0]["metric_type"], "COSINE")
        self.assertEqual(args["index_params"][1]["params"], {"bm25_k1": 1.5, "bm25_b": 0.75})

    def test_first_insert_explicit_ids_and_existing_upsert(self):
        chunks = [Document(id="stable-0", page_content="吉林大学")]
        self.backend.client.has_collection.return_value = False
        self.service.upload_chunks(chunks)
        self.backend.add_documents.assert_called_once_with(chunks, ids=["stable-0"], timeout=config.MILVUS_TIMEOUT)
        self.backend.upsert.assert_not_called()
        self.backend.client.has_collection.return_value = True
        self.service.upload_chunks(chunks)
        self.backend.upsert.assert_called_once_with(documents=chunks, ids=["stable-0"], timeout=config.MILVUS_TIMEOUT)

    def test_invalid_chunks_never_embed(self):
        for chunks in [[], [Document(page_content="no id")], [Document(id="same", page_content="a"), Document(id="same", page_content="b")]]:
            with self.subTest(chunks=chunks), self.assertRaises(ValueError):
                self.service.upload_chunks(chunks)
        self.backend.add_documents.assert_not_called()
        self.backend.upsert.assert_not_called()

    def test_empty_collection_without_embedding_and_multi_session_reload(self):
        self.backend.client.has_collection.return_value = False
        self.assertEqual(self.service.get_chunks(), [])
        self.assertEqual(self.service.search("吉林大学"), [])
        self.backend.client.query_iterator.assert_not_called()
        self.backend.similarity_search_with_score.assert_not_called()
        self.backend.fields = []
        self.backend.client.has_collection.return_value = True
        refreshed = Mock(fields=OUTPUT_FIELDS)
        refreshed.similarity_search_with_score.return_value = []
        self.constructor.return_value = refreshed
        self.service.search("吉林大学")
        self.assertIs(self.service.store, refreshed)
        self.assertEqual(self.constructor.call_count, 2)

    def test_paginated_query_preserves_metadata_and_escapes_filename(self):
        filename = '校区"\\材料.txt'
        rows = [dict(pk=f"id-{i}", text=f"正文{i}", source=filename, id=f"id-{i}",
                     chunk_index=i, create_time="2026-10-07", content_hash="hash") for i in range(3)]
        iterator = Mock()
        iterator.next.side_effect = [rows[:2], rows[2:], []]
        self.backend.client.query_iterator.return_value = iterator
        documents = self.service.get_chunks(filename)
        self.assertEqual([doc.id for doc in documents], ["id-0", "id-1", "id-2"])
        self.assertEqual(documents[2].metadata["chunk_index"], 2)
        self.assertEqual(documents[0].metadata["source"], filename)
        args = self.backend.client.query_iterator.call_args.kwargs
        self.assertEqual(json.loads(args["filter"].split(" == ", 1)[1]), filename)
        self.assertEqual(args["output_fields"], OUTPUT_FIELDS)
        self.assertEqual(args["consistency_level"], "Strong")
        iterator.close.assert_called_once_with()

    def test_iterator_closed_on_failure_and_outages_propagate(self):
        iterator = Mock()
        iterator.next.side_effect = ConnectionError("查询中断")
        self.backend.client.query_iterator.return_value = iterator
        with self.assertRaises(ConnectionError):
            self.service.get_chunks()
        iterator.close.assert_called_once_with()
        self.backend.client.has_collection.side_effect = ConnectionError("服务不可用")
        with self.assertRaises(ConnectionError):
            self.service.search("吉林大学")

    def test_search_uses_server_rrf_and_restores_document_id(self):
        raw_doc = Document(page_content="校园信息", metadata={"pk": "stable", "source": "history.txt", "chunk_index": 0})
        self.backend.similarity_search_with_score.return_value = [(raw_doc, 0.1)]
        results = self.service.search("  吉林大学  ")
        args = self.backend.similarity_search_with_score.call_args
        self.assertEqual(args.args, ("吉林大学",))
        self.assertEqual(args.kwargs["k"], 6)
        self.assertEqual(args.kwargs["fetch_k"], 3)
        self.assertEqual(args.kwargs["reranker"].type, FunctionType.RERANK)
        self.assertEqual(args.kwargs["reranker"].params, {"reranker": "rrf", "k": 60})
        self.assertEqual(results[0][0].id, "stable")
        self.assertEqual(results[0][0].metadata, {"source": "history.txt", "chunk_index": 0})
        self.assertIn("pk", raw_doc.metadata)

    def test_invalid_queries_never_search(self):
        for query in [None, 123]:
            with self.assertRaises(TypeError):
                self.service.search(query)
        for query in ["", " \n\t"]:
            with self.assertRaises(ValueError):
                self.service.search(query)
        self.backend.similarity_search_with_score.assert_not_called()

    def test_delete_and_write_failures_propagate(self):
        self.service.delete_chunks([])
        self.backend.delete.assert_not_called()
        self.service.delete_chunks(["stable"])
        self.backend.delete.assert_called_once_with(ids=["stable"], timeout=config.MILVUS_TIMEOUT)
        self.backend.upsert.side_effect = ConnectionError("写入失败")
        with self.assertRaises(ConnectionError):
            self.service.upload_chunks([Document(id="stable", page_content="正文")])

    def test_connection_failure_is_not_an_empty_library(self):
        self.constructor.side_effect = ConnectionError("Milvus 服务不可用")
        with self.assertRaisesRegex(ConnectionError, "服务不可用"):
            MilvusStoreService(
                embeddings=self.embeddings,
                connection_args={"uri": "http://localhost:19530"},
                collection_name="test_knowledge",
            )


if __name__ == "__main__":
    unittest.main()
