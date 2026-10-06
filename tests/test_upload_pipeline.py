"""验证 Milvus 上传、幂等恢复和知识库管理页面。"""

from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import streamlit as st
from langchain_core.documents import Document

from JLU_agent.services.RAG import file_ls
from JLU_agent.services.RAG.text_splitter import TextSplitterService


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


    upload_chunks = upload

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
        self.service.milvusStoreService = self.store

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

    def test_replacement_with_duplicate_content_preserves_original(self):
        self.service.upload_by_doc("吉林大学历史", "history.txt")
        self.service.upload_by_doc("校园食堂菜单", "canteen.txt")
        before = dict(self.store.documents)
        result = self.service.upload_by_doc("校园食堂菜单", "history.txt")
        self.assertIn("canteen.txt", result)
        self.assertEqual(self.store.documents, before)

    def test_duplicate_detection_with_multiple_versions_after_interrupted_replace(self):
        splitter = self.service.textSplitterService
        versions = splitter.split_text("旧版历史", "history.txt")
        versions += splitter.split_text("新版历史", "history.txt")
        self.store.upload(versions)
        self.assertEqual(len(self.service.list_documents()), 1)
        self.assertIn("history.txt", self.service.upload_by_doc("新版历史", "copy.txt"))
        self.service.upload_by_doc("最新版历史", "history.txt")
        self.assertEqual(len(self.service.get_document_chunks("history.txt")), 1)
        self.assertEqual(self.service.get_document_chunks("history.txt")[0].page_content, "最新版历史")

    def test_retry_after_write_and_delete_failure(self):
        self.store.fail_after_write = True
        with self.assertRaises(OSError):
            self.service.upload_by_doc("吉林大学历史", "history.txt")
        self.service.upload_by_doc("吉林大学历史", "history.txt")
        self.assertEqual(self.store.calls, 1)
        with patch.object(self.store, "delete_chunks", side_effect=OSError("删除失败")):
            with self.assertRaises(OSError):
                self.service.upload_by_doc("新版历史", "history.txt")
        self.assertEqual(len(self.store.documents), 2)
        self.service.upload_by_doc("新版历史", "history.txt")
        self.assertEqual(self.store.calls, 2)
        self.assertEqual(len(self.store.documents), 1)

    def test_partial_write_retry_fills_only_missing_chunks(self):
        text = "吉林大学历史。" * 300
        chunks = self.service.textSplitterService.split_text(text, "history.txt")
        self.store.upload(chunks[:1])
        self.service.upload_by_doc(text, "history.txt")
        self.assertEqual(set(self.store.documents), {doc.id for doc in chunks})

    def test_incomplete_replacement_retains_old_chunks_until_retry(self):
        self.service.upload_by_doc("旧版历史", "history.txt")
        old_ids = set(self.store.documents)
        text = "吉林大学历史。" * 300
        chunks = self.service.textSplitterService.split_text(text, "history.txt")
        original_upload = self.store.upload_chunks
        with patch.object(self.store, "upload_chunks", side_effect=lambda docs: self.store.upload(docs[:1])):
            with self.assertRaisesRegex(RuntimeError, "尚未全部入库"):
                self.service.upload_by_doc(text, "history.txt")
        self.assertTrue(old_ids.issubset(self.store.documents))
        uploaded = []
        def record_upload(docs):
            uploaded.extend(docs)
            original_upload(docs)
        with patch.object(self.store, "upload_chunks", side_effect=record_upload):
            self.service.upload_by_doc(text, "history.txt")
        self.assertEqual([doc.id for doc in uploaded], [doc.id for doc in chunks[1:]])
        self.assertEqual(set(self.store.documents), {doc.id for doc in chunks})

    def test_invalid_upload_preserves_existing_data(self):
        self.service.upload_by_doc("旧版历史", "history.txt")
        before = dict(self.store.documents)
        for text, filename in [(" \n", "history.txt"), ("正文", ""), (None, "history.txt")]:
            with self.assertRaises(ValueError):
                self.service.upload_by_doc(text, filename)
        self.assertEqual(self.store.documents, before)

    def test_multiple_services_see_latest_upload_and_deletion(self):
        other = object.__new__(file_ls.FileLoaderAndSearchService)
        other.textSplitterService = TextSplitterService()
        other.milvusStoreService = self.store
        self.service.upload_by_doc("校园信息", "campus.txt")
        self.assertEqual(other.list_documents()[0]["filename"], "campus.txt")
        other.delete_document("campus.txt")
        self.assertEqual(self.service.list_documents(), [])

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

    def test_empty_library_has_no_rebuild_button(self):
        self.service.list_documents.return_value = []
        self.app.run()
        self.assertIn("暂无文档", self.app.info[0].value)
        self.assertFalse(any(button.label == "重建检索索引" for button in self.app.button))
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
