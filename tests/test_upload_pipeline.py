"""在临时目录和模拟服务中检查手动上传到两个知识库的流程。"""

from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import streamlit as st
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from JLU_agent.config import chroma_config as config
from JLU_agent.services.RAG import file_ls, index_store
from JLU_agent.services.RAG.text_splitter import TextSplitterService
from JLU_agent.services.RAG.vector_store import VectorStoreService


class LocalEmbeddings(Embeddings):
    """固定维度的本地向量，确保测试不触发外部模型。"""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), 1.0, 0.0] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class RecordingStore:
    """模拟按 Document.id 覆盖写入的存储，并可在写入后失败一次。"""

    def __init__(self) -> None:
        self.documents: dict[str, Document] = {}
        self.calls = 0
        self.fail_after_write = False

    def upload(self, chunks: list[Document]) -> None:
        self.calls += 1
        for chunk in chunks:
            if not chunk.id:
                raise AssertionError("上传的切片必须有稳定 ID")
            self.documents[chunk.id] = chunk
        if self.fail_after_write:
            self.fail_after_write = False
            raise OSError("模拟存储完成后报告失败")


    upload_into_chroma = upload

    def get_chunks(self, filename=None):
        return [doc for doc in self.documents.values()
                if filename is None or doc.metadata["source"] == filename]

    def delete_chunks(self, ids):
        for doc_id in ids:
            self.documents.pop(doc_id, None)


class UploadPipelineTests(unittest.TestCase):
    def setUp(self):
        self.service = object.__new__(file_ls.FileLoaderAndSearchService)
        self.service.textSplitterService = TextSplitterService()
        self.store = RecordingStore()
        self.service.vectorStoreService = self.store
        self.service.indexStoreService = Mock()

    def test_upload_duplicate_and_delete_then_reupload(self):
        service = self.service
        self.assertIn("已上传", service.upload_by_doc("吉林大学历史", "history.txt"))
        self.assertIn("内容未变化", service.upload_by_doc("吉林大学历史", "history.txt"))
        self.assertIn("history.txt", service.upload_by_doc("吉林大学历史", "copy.txt"))
        self.assertEqual(self.store.calls, 1)
        self.assertEqual(service.list_documents()[0]["chunk_count"], 1)
        service.delete_document("history.txt")
        self.assertEqual(service.list_documents(), [])
        self.assertIn("已上传", service.upload_by_doc("吉林大学历史", "history.txt"))
        self.assertEqual(self.store.calls, 2)

    def test_shorter_replacement_removes_old_chunks_and_keeps_other_file(self):
        service = self.service
        service.upload_by_doc("吉林大学历史。" * 300, "history.txt")
        service.upload_by_doc("校园食堂菜单", "canteen.txt")
        old_ids = {doc.id for doc in service.get_document_chunks("history.txt")}
        self.assertGreater(len(old_ids), 1)
        self.assertIn("已替换", service.upload_by_doc("新版吉林大学历史", "history.txt"))
        self.assertEqual(len(service.get_document_chunks("history.txt")), 1)
        self.assertFalse(old_ids & set(self.store.documents))
        self.assertEqual(len(service.get_document_chunks("canteen.txt")), 1)
        rebuilt = self.service.indexStoreService.rebuild.call_args.args[0]
        self.assertEqual({doc.id for doc in rebuilt}, set(self.store.documents))

    def test_replacement_with_duplicate_content_preserves_original(self):
        self.service.upload_by_doc("吉林大学历史", "history.txt")
        self.service.upload_by_doc("校园食堂菜单", "canteen.txt")
        before = dict(self.store.documents)
        result = self.service.upload_by_doc("校园食堂菜单", "history.txt")
        self.assertIn("canteen.txt", result)
        self.assertEqual(self.store.documents, before)

    def test_legacy_duplicates_and_multiple_versions(self):
        splitter = self.service.textSplitterService
        versions = splitter.split_text("旧版历史", "history.txt")
        versions += splitter.split_text("新版历史", "history.txt")
        for doc in versions:
            doc.metadata.pop("content_hash")
        self.store.upload(versions)
        self.assertEqual(len(self.service.list_documents()), 1)
        self.assertIn("history.txt", self.service.upload_by_doc("新版历史", "copy.txt"))
        self.service.upload_by_doc("最新版历史", "history.txt")
        self.assertEqual(len(self.service.get_document_chunks("history.txt")), 1)
        self.assertEqual(self.service.get_document_chunks("history.txt")[0].page_content, "最新版历史")

    def test_retry_after_vector_and_index_failure(self):
        self.store.fail_after_write = True
        with self.assertRaises(OSError):
            self.service.upload_by_doc("吉林大学历史", "history.txt")
        self.service.upload_by_doc("吉林大学历史", "history.txt")
        self.assertEqual(self.store.calls, 1)
        self.service.indexStoreService.rebuild.side_effect = OSError("磁盘写入失败")
        with self.assertRaises(OSError):
            self.service.upload_by_doc("新版历史", "history.txt")
        self.service.indexStoreService.rebuild.side_effect = None
        self.service.upload_by_doc("新版历史", "history.txt")
        self.assertEqual(self.store.calls, 2)
        self.assertEqual(len(self.store.documents), 1)

    def test_partial_vector_write_retry_fills_missing_chunks(self):
        text = "吉林大学历史。" * 300
        chunks = self.service.textSplitterService.split_text(text, "history.txt")
        self.store.upload(chunks[:1])
        self.service.upload_by_doc(text, "history.txt")
        self.assertEqual(set(self.store.documents), {doc.id for doc in chunks})

    def test_preview_order_and_latest_time(self):
        chunks = self.service.textSplitterService.split_text("吉林大学历史。" * 300, "history.txt")
        chunks[-1].metadata["create_time"] = "2099-01-01 00:00:00"
        self.store.upload(list(reversed(chunks)))
        self.assertEqual(self.service.get_document_chunks("history.txt"), chunks)
        self.assertEqual(self.service.list_documents()[0]["create_time"], "2099-01-01 00:00:00")
        self.assertEqual(self.service.get_document_chunks("missing.txt"), [])

    def test_splitter_has_stable_ids_and_content_hash(self):
        first = self.service.textSplitterService.split_text("吉林大学历史。" * 200, "history.txt")
        again = self.service.textSplitterService.split_text("吉林大学历史。" * 200, "history.txt")
        self.assertEqual([doc.id for doc in first], [doc.id for doc in again])
        for number, doc in enumerate(first):
            self.assertEqual(doc.metadata["chunk_index"], number)
            self.assertTrue(doc.id.startswith(doc.metadata["content_hash"]))


class IndexStorePersistenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        path_patch = patch.object(config, "INDEX_PATH", Path(self.temp.name) / "index_db")
        path_patch.start()
        self.addCleanup(path_patch.stop)
        segmenter = SimpleNamespace(cut=lambda text: list(text))
        segmenter_patch = patch.object(index_store.pkuseg, "pkuseg", return_value=segmenter)
        segmenter_patch.start()
        self.addCleanup(segmenter_patch.stop)

    def test_missing_or_empty_index_directory_starts_without_retriever(self) -> None:
        self.assertIsNone(index_store.IndexStoreService().retriever)
        config.INDEX_PATH.mkdir()
        self.assertIsNone(index_store.IndexStoreService().retriever)

    def test_stale_service_instances_keep_both_documents(self) -> None:
        first = index_store.IndexStoreService()
        second = index_store.IndexStoreService()
        first_doc = Document(
            id="file-one-0",
            page_content="吉林大学历史",
            metadata={"source": "first.txt", "chunk_index": 0, "create_time": "2026-01-01"},
        )
        second_doc = Document(
            id="file-two-0",
            page_content="吉林大学图书馆",
            metadata={"source": "second.txt", "chunk_index": 0, "create_time": "2026-01-02"},
        )

        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            first.upload_into_bm5([first_doc])
            second.upload_into_bm5([second_doc])
            restored = index_store.IndexStoreService()
            restored.search("吉林大学")

        self.assertEqual(
            {item["id"] for item in restored.retriever.corpus},
            {"file-one-0", "file-two-0"},
        )
        self.assertEqual(
            {item["id"]: item["source"] for item in restored.retriever.corpus},
            {"file-one-0": "first.txt", "file-two-0": "second.txt"},
        )

    def test_reuploading_same_chunk_does_not_duplicate_it(self) -> None:
        service = index_store.IndexStoreService()
        doc = Document(
            id="history-0", page_content="吉林大学历史", metadata={"source": "history.txt"}
        )
        service.upload_into_bm5([doc])
        service.upload_into_bm5([doc])

        restored = index_store.IndexStoreService()
        restored.search("吉林大学")
        self.assertEqual([item["id"] for item in restored.retriever.corpus], ["history-0"])

    def test_search_returns_scored_documents_and_skips_zero_scores(self) -> None:
        service = index_store.IndexStoreService()
        service.upload_into_bm5([
            Document(
                id="history-0",
                page_content="吉林大学历史",
                metadata={
                    "source": "history.txt",
                    "chunk_index": 0,
                    "create_time": "2026-09-29",
                },
            ),
            Document(
                id="canteen-0",
                page_content="食堂菜单",
                metadata={"source": "canteen.txt", "chunk_index": 0},
            ),
        ])

        with patch.object(config, "K", 3):  # 索引只有两块，不能直接请求 3 条。
            results = service.search("  吉林大学  ")
        self.assertEqual(len(results), 1)
        document, score = results[0]
        self.assertEqual(document.id, "history-0")
        self.assertEqual(document.page_content, "吉林大学历史")
        self.assertEqual(document.metadata["source"], "history.txt")
        self.assertEqual(document.metadata["chunk_index"], 0)
        self.assertEqual(document.metadata["create_time"], "2026-09-29")
        self.assertIs(type(score), float)
        self.assertGreater(score, 0)
        self.assertEqual(service.search("火星"), [])

    def test_search_reads_uploads_from_another_service_instance(self) -> None:
        first = index_store.IndexStoreService()
        second = index_store.IndexStoreService()
        self.assertEqual(second.search("吉林大学"), [])

        first.upload_into_bm5([
            Document(
                id="history-0",
                page_content="吉林大学历史",
                metadata={"source": "history.txt"},
            )
        ])
        results = second.search("吉林大学")
        self.assertEqual([document.id for document, _ in results], ["history-0"])

    def test_search_validates_query_and_handles_empty_index(self) -> None:
        service = index_store.IndexStoreService()
        for query in (None, 123):
            with self.subTest(query=query), self.assertRaises(TypeError):
                service.search(query)
        for query in ("", " \n\t"):
            with self.subTest(query=query), self.assertRaises(ValueError):
                service.search(query)
        self.assertEqual(service.search("吉林大学"), [])
        config.INDEX_PATH.mkdir()
        self.assertEqual(service.search("吉林大学"), [])

    def test_failed_bm25_save_does_not_update_in_memory_retriever(self) -> None:
        service = index_store.IndexStoreService()
        first_doc = Document(
            id="first-0", page_content="吉林大学历史", metadata={"source": "first.txt"}
        )
        second_doc = Document(
            id="second-0", page_content="吉林大学图书馆", metadata={"source": "second.txt"}
        )
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            service.upload_into_bm5([first_doc])
            with patch.object(index_store.bm25s.BM25, "save", side_effect=OSError("磁盘写入失败")):
                with self.assertRaises(OSError):
                    service.upload_into_bm5([second_doc])

            self.assertEqual({doc["id"] for doc in service.retriever.corpus}, {"first-0"})
            service.upload_into_bm5([second_doc])
            restored = index_store.IndexStoreService()
            restored.search("吉林大学")
            self.assertEqual(
                {doc["id"] for doc in restored.retriever.corpus},
                {"first-0", "second-0"},
            )


class ChromaPersistenceTests(unittest.TestCase):
    def test_same_document_id_is_upserted_in_temporary_chroma(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = object.__new__(VectorStoreService)
            store.chroma = Chroma(
                collection_name="isolated_upload_retry",
                embedding_function=LocalEmbeddings(),
                persist_directory=temp_dir,
                client_settings=Settings(anonymized_telemetry=False),
            )
            doc = Document(
                id="same-content-0",
                page_content="吉林大学历史",
                metadata={"source": "history.txt", "chunk_index": 0},
            )

            try:
                store.upload_into_chroma([doc])
                store.upload_into_chroma([doc])

                self.assertEqual(store.chroma._collection.count(), 1)
                self.assertEqual(store.chroma._collection.get()["ids"], [doc.id])
                self.assertEqual(
                    store.chroma._collection.get()["metadatas"][0]["source"],
                    "history.txt",
                )
            finally:
                # Windows 会锁定 Chroma 的 mmap 文件，必须先关闭客户端再删临时目录。
                store.chroma._client.close()


class KnowledgeManagementPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        path_patch = patch.object(config, "INDEX_PATH", Path(self.temp.name) / "index_db")
        path_patch.start()
        self.addCleanup(path_patch.stop)
        segmenter_patch = patch.object(
            index_store.pkuseg, "pkuseg", return_value=SimpleNamespace(cut=lambda text: list(text))
        )
        segmenter_patch.start()
        self.addCleanup(segmenter_patch.stop)
        self.vector = object.__new__(VectorStoreService)
        self.vector.chroma = Chroma(
            collection_name="management_test",
            embedding_function=LocalEmbeddings(),
            persist_directory=str(Path(self.temp.name) / "chroma"),
            client_settings=Settings(anonymized_telemetry=False),
        )
        self.addCleanup(self.vector.chroma._client.close)
        self.first = self.make_service()
        self.second = self.make_service()

    def make_service(self):
        service = object.__new__(file_ls.FileLoaderAndSearchService)
        service.textSplitterService = TextSplitterService()
        service.vectorStoreService = self.vector
        service.indexStoreService = index_store.IndexStoreService()
        return service

    def assert_consistent(self):
        chunks = self.vector.get_chunks()
        index = self.second.indexStoreService
        index.search("吉林大学")
        self.assertEqual(
            {doc.id for doc in chunks},
            {item["id"] for item in index.retriever.corpus} if index.retriever else set(),
        )

    def test_replace_delete_and_other_instances_see_latest(self):
        self.first.upload_by_doc("吉林大学历史。" * 300, "history.txt")
        self.first.upload_by_doc("校园食堂菜单", "canteen.txt")
        self.assertEqual(len(self.second.list_documents()), 2)
        old_ids = {doc.id for doc in self.second.get_document_chunks("history.txt")}
        self.first.upload_by_doc("新版吉林大学历史", "history.txt")
        self.assert_consistent()
        self.assertFalse(old_ids & {doc.id for doc in self.vector.get_chunks()})
        self.first.delete_document("history.txt")
        self.assert_consistent()
        self.assertEqual([doc["filename"] for doc in self.second.list_documents()], ["canteen.txt"])
        self.assertEqual(self.second.indexStoreService.search("吉林大学"), [])
        self.assertTrue(all(doc.metadata["source"] != "history.txt"
                            for doc, _ in self.vector.search("吉林大学")))
        self.first.delete_document("canteen.txt")
        self.assert_consistent()
        self.assertEqual(self.second.indexStoreService.search("食堂"), [])
        self.assertEqual(self.vector.search("食堂"), [])
        self.first.upload_by_doc("校园食堂菜单", "canteen.txt")
        self.assert_consistent()

    def test_failed_save_can_be_rebuilt_or_retried_without_embedding(self):
        self.first.upload_by_doc("吉林大学历史", "history.txt")
        with patch.object(index_store.bm25s.BM25, "save", side_effect=OSError("磁盘写入失败")):
            with self.assertRaises(OSError):
                self.first.upload_by_doc("新版吉林大学历史", "history.txt")
        with patch.object(self.vector, "upload_into_chroma") as embed:
            self.second.upload_by_doc("新版吉林大学历史", "history.txt")
            embed.assert_not_called()
        self.assert_consistent()
        with patch.object(index_store.bm25s.BM25, "save", side_effect=OSError("磁盘写入失败")):
            with self.assertRaises(OSError):
                self.first.upload_by_doc("食堂菜单", "canteen.txt")
        self.second.rebuild_index()
        self.assert_consistent()

    def test_rebuild_does_not_need_a_readable_old_index(self):
        self.first.upload_by_doc("吉林大学历史", "history.txt")
        (config.INDEX_PATH / "params.json").write_text("broken", encoding="utf-8")
        restored = self.make_service()
        restored.rebuild_index()
        self.assert_consistent()


class KnowledgePageTests(unittest.TestCase):
    def setUp(self):
        from streamlit.testing.v1 import AppTest
        page = Path(__file__).resolve().parents[1] / "streamlit_app" / "pages" / "app_knowledge.py"
        self.app = AppTest.from_file(str(page))
        self.service = Mock()
        self.service.list_documents.return_value = [
            {"filename": "history.txt", "chunk_count": 1, "create_time": "2026-10-05"},
            {"filename": "canteen.txt", "chunk_count": 1, "create_time": "2026-10-05"},
        ]
        self.service.get_document_chunks.return_value = [Document(page_content="吉林大学历史")]
        self.service.upload_by_doc.return_value = "已上传文档。"
        self.app.session_state["file_ls_service"] = self.service

    def button(self, label):
        return next(button for button in self.app.button if button.label == label)

    def test_list_filter_and_preview(self):
        self.app.run()
        self.assertFalse(self.app.exception)
        self.assertEqual(len(self.app.dataframe[0].value), 2)
        self.app.text_input[0].set_value("HISTORY").run()
        self.assertEqual(len(self.app.dataframe[0].value), 1)
        self.app.selectbox[0].select("history.txt").run()
        self.assertEqual(self.app.code[0].value, "吉林大学历史")
        self.assertTrue(self.button("删除所选文档").disabled)
        self.service.delete_document.assert_not_called()

    def test_confirmed_deletion_refreshes_and_clears_selection(self):
        self.app.run()
        self.app.selectbox[0].select("history.txt").run()
        self.app.checkbox[0].check().run()
        def delete(filename):
            self.service.list_documents.return_value = [
                doc for doc in self.service.list_documents.return_value
                if doc["filename"] != filename
            ]
        self.service.delete_document.side_effect = delete
        self.button("删除所选文档").click().run()
        self.service.delete_document.assert_called_once_with("history.txt")
        self.assertFalse(self.app.exception)
        self.assertEqual(self.app.success[0].value, "已删除「history.txt」。")
        self.assertIsNone(self.app.selectbox[0].value)

    def test_failed_deletion_does_not_report_success(self):
        self.app.run()
        self.app.selectbox[0].select("history.txt").run()
        self.app.checkbox[0].check().run()
        self.service.delete_document.side_effect = OSError("磁盘写入失败")
        self.button("删除所选文档").click().run()
        self.assertFalse(self.app.exception)
        self.assertFalse(self.app.success)
        self.assertIn("操作未全部完成", self.app.error[0].value)
        self.assertEqual(self.app.selectbox[0].value, "history.txt")

    def test_rebuild_empty_library(self):
        self.service.list_documents.return_value = []
        self.app.run()
        self.assertIn("暂无文档", self.app.info[0].value)
        self.button("重建检索索引").click().run()
        self.service.rebuild_index.assert_called_once_with()
        self.assertEqual(self.app.success[0].value, "检索索引已重建。")
        self.assertFalse(self.app.exception)

    def test_upload_only_on_click_and_parse_failure(self):
        uploaded = SimpleNamespace(
            name="history.txt", size=20, getvalue=lambda: "吉林大学历史".encode("utf-8")
        )
        with patch.object(st, "file_uploader", return_value=uploaded):
            self.app.run()
            self.service.upload_by_doc.assert_not_called()
            self.button("上传到知识库").click().run()
        self.service.upload_by_doc.assert_called_once_with("吉林大学历史", "history.txt")
        self.assertFalse(self.app.exception)
        self.assertEqual(self.app.success[0].value, "已上传文档。")
        uploaded.getvalue = lambda: b""
        with patch.object(st, "file_uploader", return_value=uploaded):
            self.app.run()
        self.assertTrue(self.app.error)
        self.assertFalse(any(button.label == "上传到知识库" for button in self.app.button))

    def test_failed_upload_does_not_report_success(self):
        uploaded = SimpleNamespace(name="history.txt", size=20, getvalue=lambda: b"history")
        self.service.upload_by_doc.side_effect = OSError("磁盘写入失败")
        with patch.object(st, "file_uploader", return_value=uploaded):
            self.app.run()
            self.button("上传到知识库").click().run()
        self.assertFalse(self.app.success)
        self.assertIn("操作未全部完成", self.app.error[0].value)
        self.assertFalse(self.app.exception)


if __name__ == "__main__":
    unittest.main()
