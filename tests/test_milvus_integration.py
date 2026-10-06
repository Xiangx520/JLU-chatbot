"""显式启用的真实 Milvus 测试；只操作随机测试 Collection。"""

import os
import shutil
import subprocess
import time
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from langchain_core.embeddings import Embeddings

from JLU_agent.config import rag_config as config
from JLU_agent.services.RAG.file_ls import FileLoaderAndSearchService
from JLU_agent.services.RAG.milvus_store import MilvusStoreService
from JLU_agent.services.RAG.text_splitter import TextSplitterService


class LocalEmbeddings(Embeddings):
    """固定的本地语义类别向量，用于验证存储流程而非衡量模型效果。"""

    def __init__(self):
        self.document_calls = 0

    @staticmethod
    def _vector(text):
        return [float("历史" in text), float("校区" in text), float("食堂" in text), 0.1]

    def embed_documents(self, texts):
        self.document_calls += 1
        return [self._vector(text) for text in texts]

    def embed_query(self, text):
        return self._vector(text)


@unittest.skipUnless(os.getenv("RUN_MILVUS_INTEGRATION") == "1", "设置 RUN_MILVUS_INTEGRATION=1 连接真实测试服务")
class MilvusIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.collection_name = f"test_jlu_{uuid4().hex}"
        self.embeddings = LocalEmbeddings()
        self.connection_args = {
            "uri": os.getenv("MILVUS_TEST_URI", "http://localhost:19530"),
            "db_name": os.getenv("MILVUS_TEST_DB_NAME", "default"),
            "timeout": 5.0,
        }
        if token := os.getenv("MILVUS_TEST_TOKEN"):
            self.connection_args["token"] = token
        self.store = self.make_store()
        self.addCleanup(self.cleanup_collection)
        self.service = self.make_service(self.store)

    def make_store(self):
        return MilvusStoreService(
            embeddings=self.embeddings,
            connection_args=self.connection_args,
            collection_name=self.collection_name,
        )

    @staticmethod
    def make_service(store):
        service = object.__new__(FileLoaderAndSearchService)
        service.textSplitterService = TextSplitterService()
        service.milvusStoreService = store
        return service

    def cleanup_collection(self):
        # Collection 名只由本测试生成，永远不使用应用的知识库名。
        client = self.store.store.client
        try:
            if client.has_collection(self.collection_name):
                client.drop_collection(self.collection_name)
        finally:
            client.close()

    def test_upload_chinese_bm25_dense_hybrid_and_idempotent_upsert(self):
        self.assertEqual(self.store.search("校区"), [])
        self.service.upload_by_doc("吉林大学历史始于一九四六年。", "history.txt")
        self.service.upload_by_doc("吉林大学校区交通指南，校区位于长春。", "campus.txt")
        self.service.upload_by_doc("吉林大学食堂提供早餐。", "canteen.txt")
        backend = self.store.store
        sparse = backend.client.search(
            self.collection_name,
            data=["校区"], anns_field="sparse", limit=3,
            search_params={"metric_type": "BM25", "params": {}},
            output_fields=["source", "text"], consistency_level="Strong",
        )[0]
        self.assertTrue(sparse)
        self.assertEqual(sparse[0]["entity"]["source"], "campus.txt")
        self.assertGreater(sparse[0]["distance"], 0)
        dense = backend.client.search(
            self.collection_name,
            data=[self.embeddings.embed_query("校区地点")], anns_field="dense", limit=3,
            search_params={"metric_type": "COSINE", "params": {}},
            output_fields=["source"], consistency_level="Strong",
        )[0]
        self.assertEqual(dense[0]["entity"]["source"], "campus.txt")
        hybrid = self.store.search("校区")
        self.assertTrue(hybrid)
        self.assertEqual(hybrid[0][0].metadata["source"], "campus.txt")
        self.assertTrue(all(doc.id for doc, _ in hybrid))
        chunks = self.store.get_chunks("campus.txt")
        self.store.upload_chunks(chunks)
        self.store.upload_chunks(chunks)
        self.assertEqual(len(self.store.get_chunks()), 3)

    def test_multiple_sessions_pagination_replacement_delete_and_reopen(self):
        # 两个实例均在首次建库前创建，检验后创建的 Collection 能被已有会话发现。
        other_store = self.make_store()
        self.addCleanup(other_store.store.client.close)
        other = self.make_service(other_store)
        filename = '校区"\\资料.txt'
        self.service.upload_by_doc("吉林大学校区交通指南。" * 300, filename)
        with patch.object(config, "QUERY_BATCH_SIZE", 2):
            chunks = other.get_document_chunks(filename)
        self.assertGreater(len(chunks), 2)
        self.assertEqual([doc.metadata["chunk_index"] for doc in chunks], list(range(len(chunks))))
        self.assertTrue(all(doc.metadata["source"] == filename for doc in chunks))
        old_ids = {doc.id for doc in chunks}
        calls = self.embeddings.document_calls
        self.service.upload_by_doc("吉林大学校区交通指南。" * 300, filename)
        self.assertEqual(self.embeddings.document_calls, calls)
        self.assertIn(filename, other.upload_by_doc("吉林大学校区交通指南。" * 300, "copy.txt"))
        self.service.upload_by_doc("新版校区交通指南", filename)
        replacement = other.get_document_chunks(filename)
        self.assertEqual(len(replacement), 1)
        self.assertFalse(old_ids & {doc.id for doc in replacement})
        reopened = self.make_store()
        self.addCleanup(reopened.store.client.close)
        self.assertEqual([doc.id for doc in reopened.get_chunks(filename)], [replacement[0].id])
        other.delete_document(filename)
        self.assertEqual(self.service.list_documents(), [])
        self.assertEqual(self.store.search("校区"), [])
        self.service.upload_by_doc("新版校区交通指南", filename)
        self.assertEqual(len(other.get_document_chunks(filename)), 1)

    @unittest.skipUnless(os.getenv("RUN_MILVUS_RESTART") == "1", "设置 RUN_MILVUS_RESTART=1 允许重启本地 Compose 服务")
    def test_service_restart_preserves_test_collection(self):
        self.assertIn(self.connection_args["uri"], {"http://localhost:19530", "http://127.0.0.1:19530"})
        self.assertIsNotNone(shutil.which("docker"), "重启测试需要 Docker Compose")
        self.service.upload_by_doc("吉林大学历史资料", "history.txt")
        expected_ids = [doc.id for doc in self.store.get_chunks()]
        self.store.store.client.flush(self.collection_name)
        subprocess.run(
            ["docker", "compose", "restart", "standalone"],
            cwd=Path(__file__).resolve().parents[1], check=True, timeout=60,
        )
        deadline = time.monotonic() + 90
        while True:
            try:
                reopened = self.make_store()
                break
            except Exception:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(2)
        self.addCleanup(reopened.store.client.close)
        self.assertEqual([doc.id for doc in reopened.get_chunks()], expected_ids)
        self.assertTrue(reopened.search("历史"))


if __name__ == "__main__":
    unittest.main()
