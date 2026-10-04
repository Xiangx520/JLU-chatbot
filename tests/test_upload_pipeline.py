"""在临时目录和模拟服务中检查手动上传到两个知识库的流程。"""

from __future__ import annotations

import contextlib
import io
import runpy
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
from JLU_agent.services.RAG.md5_str_file import Md5Service
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


class UploadPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        md5_patch = patch.object(config, "MD5_PATH", Path(self.temp.name) / "md5.txt")
        md5_patch.start()
        self.addCleanup(md5_patch.stop)

        self.service = object.__new__(file_ls.FileLoaderAndSearchService)
        self.service.md5Service = Md5Service()
        self.service.textSplitterService = TextSplitterService()
        self.chroma = RecordingStore()
        self.index = RecordingStore()
        self.service.vectorStoreService = SimpleNamespace(
            upload_into_chroma=self.chroma.upload
        )
        self.service.indexStoreService = SimpleNamespace(
            upload_into_bm5=self.index.upload
        )

    def test_first_and_duplicate_upload(self) -> None:
        first = self.service.upload_by_doc("吉林大学始建于1946年。", "history.txt")
        self.assertIn("success", first)
        self.assertEqual(self.chroma.calls, 1)
        self.assertEqual(self.index.calls, 1)
        self.assertEqual(set(self.chroma.documents), set(self.index.documents))
        self.assertTrue(self.chroma.documents)
        self.assertEqual(len(config.MD5_PATH.read_text(encoding="utf-8").splitlines()), 1)

        second = self.service.upload_by_doc("吉林大学始建于1946年。", "history.txt")
        self.assertIn("pass", second)
        self.assertEqual((self.chroma.calls, self.index.calls), (1, 1))
        self.assertEqual(len(config.MD5_PATH.read_text(encoding="utf-8").splitlines()), 1)

    def test_splitter_attaches_shared_metadata_and_stable_ids(self) -> None:
        splitter = TextSplitterService()
        text = "吉林大学历史。" * 200
        first = splitter.split_text(text, "history.txt")
        again = splitter.split_text(text, "history.txt")

        self.assertGreater(len(first), 1)
        self.assertEqual([doc.id for doc in first], [doc.id for doc in again])
        self.assertEqual(len({doc.id for doc in first}), len(first))
        for index, doc in enumerate(first):
            self.assertEqual(doc.metadata["source"], "history.txt")
            self.assertTrue(doc.metadata["create_time"])
            self.assertEqual(doc.metadata["chunk_index"], index)
            self.assertEqual(doc.metadata["id"], doc.id)

    def test_retry_after_each_write_failure(self) -> None:
        text = "吉林大学图书馆位于校园内。"
        for failed_step in ("chroma", "index", "md5"):
            with self.subTest(failed_step=failed_step):
                with tempfile.TemporaryDirectory() as temp_dir:
                    with patch.object(config, "MD5_PATH", Path(temp_dir) / "md5.txt"):
                        chroma = RecordingStore()
                        index = RecordingStore()
                        service = object.__new__(file_ls.FileLoaderAndSearchService)
                        service.md5Service = Md5Service()
                        service.textSplitterService = TextSplitterService()
                        service.vectorStoreService = SimpleNamespace(
                            upload_into_chroma=chroma.upload
                        )
                        service.indexStoreService = SimpleNamespace(
                            upload_into_bm5=index.upload
                        )

                        if failed_step == "chroma":
                            chroma.fail_after_write = True
                        elif failed_step == "index":
                            index.fail_after_write = True
                        else:
                            real_save = service.md5Service.save_md5
                            first_call = True

                            def save_once_then_fail(md5_hex: str) -> None:
                                nonlocal first_call
                                if first_call:
                                    first_call = False
                                    raise OSError("模拟 MD5 记录失败")
                                real_save(md5_hex)

                            service.md5Service.save_md5 = save_once_then_fail

                        with self.assertRaises(OSError):
                            service.upload_by_doc(text, "library.txt")
                        self.assertFalse(service.md5Service.check_md5(
                            service.md5Service.get_string_md5(text)
                        ))

                        self.assertIn("success", service.upload_by_doc(text, "library.txt"))
                        self.assertEqual(len(chroma.documents), len(index.documents))
                        self.assertEqual(set(chroma.documents), set(index.documents))
                        self.assertTrue(chroma.documents)
                        self.assertEqual(
                            len(config.MD5_PATH.read_text(encoding="utf-8").splitlines()), 1
                        )


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
            self.assertEqual(
                {doc["id"] for doc in index_store.IndexStoreService().retriever.corpus},
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


class KnowledgePageTests(unittest.TestCase):
    @staticmethod
    def uploaded_file(name="history.txt", data="吉林大学历史".encode("utf-8")):
        return SimpleNamespace(
            name=name,
            type="text/plain",
            size=len(data),
            getvalue=lambda: data,
        )

    def run_page(self, uploaded_file, clicked=False, state=None):
        if state is None:
            state = {}
        service = Mock()
        service.upload_by_doc.return_value = "[success]"
        page = Path(__file__).resolve().parents[1] / "streamlit_app" / "pages" / "app_knowledge.py"
        with (
            patch.object(st, "session_state", state),
            patch.object(st, "title"),
            patch.object(st, "subheader"),
            patch.object(st, "write"),
            patch.object(st, "code") as preview,
            patch.object(st, "error") as error,
            patch.object(st, "button", return_value=clicked) as button,
            patch.object(st, "spinner", return_value=contextlib.nullcontext()),
            patch.object(st, "file_uploader", return_value=uploaded_file) as uploader,
            patch.object(file_ls, "FileLoaderAndSearchService", return_value=service) as constructor,
        ):
            runpy.run_path(str(page))
        return SimpleNamespace(
            preview=preview, error=error, button=button, uploader=uploader,
            constructor=constructor, service=service, state=state,
        )

    def test_selection_and_reruns_only_preview(self) -> None:
        state = {}
        for _ in range(2):
            result = self.run_page(self.uploaded_file(), state=state)
            result.preview.assert_called_once_with(
                "吉林大学历史", language=None, wrap_lines=True, height=300
            )
            result.button.assert_called_once_with("上传到知识库")
            result.constructor.assert_not_called()
            result.service.upload_by_doc.assert_not_called()
        self.assertNotIn("file_ls_service", state)
        self.assertEqual(result.uploader.call_args.kwargs["type"], ["txt", "md", "pdf", "docx"])
        self.assertIs(result.uploader.call_args.kwargs["accept_multiple_files"], False)

    def test_click_uploads_original_text_and_filename(self) -> None:
        text = "# 吉林大学\n\n历史\n"
        for name in ("history.txt", "history.md"):
            with self.subTest(name=name):
                result = self.run_page(self.uploaded_file(name, text.encode("utf-8-sig")), clicked=True)
                result.constructor.assert_called_once_with()
                result.service.upload_by_doc.assert_called_once_with(text, name)
                self.assertIs(result.state["file_ls_service"], result.service)

    def test_pdf_and_docx_upload_parsed_text(self) -> None:
        from JLU_agent.services.RAG.parse_file import FileParser

        for name in ("guide.pdf", "guide.docx"):
            with self.subTest(name=name), patch.object(FileParser, "parse_file", return_value="解析正文") as parser:
                result = self.run_page(self.uploaded_file(name, b"document bytes"), clicked=True)
                parser.assert_called_once_with(b"document bytes", name)
                result.service.upload_by_doc.assert_called_once_with("解析正文", name)

    def test_existing_service_is_reused_only_on_click(self) -> None:
        service = Mock()
        state = {"file_ls_service": service}
        self.run_page(self.uploaded_file(), state=state)
        service.upload_by_doc.assert_not_called()
        for _ in range(2):
            result = self.run_page(self.uploaded_file(), clicked=True, state=state)
            result.constructor.assert_not_called()
        self.assertEqual(service.upload_by_doc.call_count, 2)
        service.upload_by_doc.assert_called_with("吉林大学历史", "history.txt")
        self.run_page(self.uploaded_file(), state=state)
        self.assertEqual(service.upload_by_doc.call_count, 2)

    def test_parse_failures_do_not_show_button_or_upload(self) -> None:
        for name, data in (
            ("empty.txt", b""), ("blank.md", b" \n"), ("invalid.txt", b"\xff"),
            ("broken.pdf", b"%PDF-1.7\ncorrupt"), ("broken.docx", b"not a zip"),
            ("unsupported.doc", b"document"),
        ):
            with self.subTest(name=name):
                result = self.run_page(self.uploaded_file(name, data), clicked=True)
                result.error.assert_called_once()
                result.preview.assert_not_called()
                result.button.assert_not_called()
                result.constructor.assert_not_called()
                result.service.upload_by_doc.assert_not_called()

    def test_no_file_does_not_initialize_service(self) -> None:
        result = self.run_page(None)
        result.constructor.assert_not_called()
        result.button.assert_not_called()
        result.preview.assert_not_called()
        result.error.assert_not_called()


if __name__ == "__main__":
    unittest.main()
