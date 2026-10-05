import hashlib

from langchain_core.documents import Document

from JLU_agent.services.RAG.index_store import IndexStoreService, UPLOAD_LOCK
from JLU_agent.services.RAG.text_splitter import TextSplitterService
from JLU_agent.services.RAG.vector_store import VectorStoreService
from JLU_agent.services.RAG.docs_reranker import rerank


class FileLoaderAndSearchService:
    """管理知识文档，并协调向量库和 BM25 检索。"""

    def __init__(self):
        self.textSplitterService = TextSplitterService()
        self.vectorStoreService = VectorStoreService()
        self.indexStoreService = IndexStoreService()

    def list_documents(self) -> list[dict]:
        with UPLOAD_LOCK:
            documents = {}
            for chunk in self.vectorStoreService.get_chunks():
                filename = chunk.metadata["source"]
                info = documents.setdefault(
                    filename, {"filename": filename, "chunk_count": 0, "create_time": ""}
                )
                info["chunk_count"] += 1
                info["create_time"] = max(
                    info["create_time"], chunk.metadata.get("create_time") or ""
                )
            return sorted(documents.values(), key=lambda doc: doc["filename"])

    def get_document_chunks(self, filename: str) -> list[Document]:
        with UPLOAD_LOCK:
            return sorted(
                self.vectorStoreService.get_chunks(filename),
                key=lambda chunk: (chunk.metadata.get("chunk_index") or 0, chunk.id),
            )

    def upload_by_doc(self, data: str, filename: str) -> str:
        with UPLOAD_LOCK:
            content_hash = hashlib.sha256(data.encode("utf-8")).hexdigest()
            existing = self.vectorStoreService.get_chunks()
            old_chunks = [doc for doc in existing if doc.metadata["source"] == filename]
            # 旧切片的 ID 已包含全文 SHA-256，无须恢复原文或重算向量。
            duplicate = next(
                (
                    doc for doc in existing
                    if (doc.metadata.get("content_hash") or doc.id.rsplit("-", 1)[0])
                    == content_hash and doc.metadata["source"] != filename
                ),
                None,
            )
            if duplicate is not None:
                self.rebuild_index()
                return f"内容已存在于「{duplicate.metadata['source']}」，未修改知识文档。"

            new_chunks = self.textSplitterService.split_text(data, filename)
            if not new_chunks:
                raise ValueError("文档正文不能为空。")
            new_ids = {doc.id for doc in new_chunks}
            existing_ids = {doc.id for doc in old_chunks}
            # 失败后重试补齐切片；完整重复不调用嵌入模型。
            if not new_ids.issubset(existing_ids):
                self.vectorStoreService.upload_into_chroma(new_chunks)
            self.vectorStoreService.delete_chunks(list(existing_ids - new_ids))
            self.rebuild_index()
            if existing_ids == new_ids:
                return f"「{filename}」内容未变化，检索索引已同步。"
            action = "替换" if old_chunks else "上传"
            return f"已{action}「{filename}」，共 {len(new_chunks)} 个切片。"

    def delete_document(self, filename: str) -> None:
        with UPLOAD_LOCK:
            chunks = self.vectorStoreService.get_chunks(filename)
            self.vectorStoreService.delete_chunks([doc.id for doc in chunks])
            self.rebuild_index()

    def rebuild_index(self) -> None:
        with UPLOAD_LOCK:
            self.indexStoreService.rebuild(self.vectorStoreService.get_chunks())

    def search(self, query: str) -> list[tuple[Document, float]]:
        with UPLOAD_LOCK:
            vector_results = self.vectorStoreService.search(query)
            bm25_results = self.indexStoreService.search(query)
        return rerank(query, vector_results, bm25_results)
