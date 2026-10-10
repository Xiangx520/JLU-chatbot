import hashlib

from langchain_core.documents import Document

from JLU_agent.services.RAG.milvus_store import MilvusStoreService, UPLOAD_LOCK
from JLU_agent.services.RAG.text_splitter import TextSplitterService


class FileLoaderAndSearchService:
    """管理知识文档，直接返回 Milvus RRF 融合后的混合检索结果。"""

    def __init__(self, *, store: MilvusStoreService | None = None):
        self.textSplitterService = TextSplitterService()
        self.milvusStoreService = store if store is not None else MilvusStoreService()

    def list_documents(self) -> list[dict]:
        with UPLOAD_LOCK:
            documents = {}
            for chunk in self.milvusStoreService.get_chunks():
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
                self.milvusStoreService.get_chunks(filename),
                key=lambda chunk: (chunk.metadata.get("chunk_index") or 0, chunk.id),
            )

    def upload_by_doc(self, data: str, filename: str) -> str:
        if not isinstance(data, str) or not data.strip():
            raise ValueError("文档正文不能为空。")
        if not isinstance(filename, str) or not filename.strip():
            raise ValueError("文件名不能为空。")
        with UPLOAD_LOCK:
            content_hash = hashlib.sha256(data.encode("utf-8")).hexdigest()
            existing = self.milvusStoreService.get_chunks()
            old_chunks = [doc for doc in existing if doc.metadata["source"] == filename]
            # 根据原文哈希去重，无须恢复原文或重算向量。
            duplicate = next(
                (
                    # 文档名不同内容相同
                    doc for doc in existing
                    if doc.metadata["content_hash"] == content_hash
                    and doc.metadata["source"] != filename
                ),
                None,
            )
            if duplicate is not None:
                return f"内容已存在于「{duplicate.metadata['source']}」，未修改知识文档。"

            new_chunks = self.textSplitterService.split_text(data, filename)
            if not new_chunks:
                raise ValueError("文档正文不能为空。")
            new_ids = {doc.id for doc in new_chunks}
            existing_ids = {doc.id for doc in old_chunks}
            # 失败后重试补齐切片；完整重复不调用嵌入模型。
            missing = [chunk for chunk in new_chunks if chunk.id not in existing_ids]
            if missing:
                self.milvusStoreService.upload_chunks(missing)
                saved_ids = {doc.id for doc in self.milvusStoreService.get_chunks(filename)}
                if not new_ids.issubset(saved_ids):
                    raise RuntimeError("新文档切片尚未全部入库，旧切片已保留，请重试上传。")
            self.milvusStoreService.delete_chunks(list(existing_ids - new_ids))
            if existing_ids == new_ids:
                return f"「{filename}」内容未变化。"
            action = "替换" if old_chunks else "上传"
            return f"已{action}「{filename}」，共 {len(new_chunks)} 个切片。"

    def delete_document(self, filename: str) -> None:
        with UPLOAD_LOCK:
            chunks = self.milvusStoreService.get_chunks(filename)
            self.milvusStoreService.delete_chunks([doc.id for doc in chunks])

    def search(self, query: str) -> list[tuple[Document, float]]:
        """保留 RRF 的结果顺序和得分，不加载或调用本地重排模型。"""
        with UPLOAD_LOCK:
            return self.milvusStoreService.search(query)
