"""在单个 Milvus Collection 中管理正文、向量和服务端 BM25。"""

import json
import threading

from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_milvus import BM25BuiltInFunction, Milvus
from pymilvus import DataType, Function, FunctionType

from JLU_agent.config import rag_config as config

# 跨页面会话共享；Milvus 的强一致性负责写入后的读取可见性。
UPLOAD_LOCK = threading.RLock()
OUTPUT_FIELDS = ["pk", "text", "source", "id", "chunk_index", "create_time", "content_hash"]


class MilvusStoreService:
    def __init__(
        self,
        *,
        embeddings: Embeddings | None = None,
        connection_args: dict | None = None,
        collection_name: str | None = None,
    ) -> None:
        self.connection_args = (
            config.get_milvus_connection_args() if connection_args is None else connection_args
        )
        self.collection_name = collection_name or config.get_milvus_collection_name()
        self.embeddings = embeddings if embeddings is not None else DashScopeEmbeddings(
            model=config.DASHSCOPE_EMBEDDING_MODEL,
            dashscope_api_key=config.get_dashscope_api_key(),
        )
        self.store = self._create_store()

    def _create_store(self) -> Milvus:
        return Milvus(
            embedding_function=self.embeddings,
            connection_args=self.connection_args,
            collection_name=self.collection_name,
            auto_id=False,
            drop_old=False,
            consistency_level="Strong",
            primary_field="pk",
            text_field="text",
            vector_field=["dense", "sparse"],
            builtin_function=BM25BuiltInFunction(
                input_field_names="text",
                output_field_names="sparse",
                analyzer_params={"type": "chinese"},
                function_name="knowledge_bm25",
            ),
            index_params=[
                {"index_type": "AUTOINDEX", "metric_type": "COSINE", "params": {}},
                {
                    "index_type": "SPARSE_INVERTED_INDEX",
                    "metric_type": "BM25",
                    "params": {"bm25_k1": config.K1, "bm25_b": config.B},
                },
            ],
            search_params=[
                {"metric_type": "COSINE", "params": {}},
                {"metric_type": "BM25", "params": {}},
            ],
            metadata_schema={
                key: {"dtype": DataType.VARCHAR, "max_length": 65535}
                for key in ["source", "id", "create_time", "content_hash"]
            } | {"chunk_index": {"dtype": DataType.INT64}},
            # langchain-milvus 0.4.0 在首次建库时会将构造器 timeout
            # 同时传入 insert 的显式参数和 kwargs；改用每次操作的 timeout。
        )

    def _existing_store(self) -> Milvus | None:
        if not self.store.client.has_collection(self.collection_name):
            return None
        # 另一个会话可能在本会话初始化之后完成首次建库；重新加载字段与索引配置。
        if not self.store.fields:
            self.store = self._create_store()
        return self.store

    def upload_chunks(self, chunks: list[Document]) -> None:
        if not chunks:
            raise ValueError("文档切片不能为空。")
        ids = [chunk.id for chunk in chunks]
        if any(not doc_id for doc_id in ids) or len(set(ids)) != len(ids):
            raise ValueError("文档切片必须包含唯一、非空的稳定 ID。")
        with UPLOAD_LOCK:
            store = self._existing_store()
            if store is None:
                # LangChain 不会自动采用 Document.id，必须显式提供 ids。
                self.store.add_documents(chunks, ids=ids, timeout=config.MILVUS_TIMEOUT)
            else:
                store.upsert(documents=chunks, ids=ids, timeout=config.MILVUS_TIMEOUT)

    def get_chunks(self, filename: str | None = None) -> list[Document]:
        with UPLOAD_LOCK:
            store = self._existing_store()
            if store is None:
                return []
            expression = "" if filename is None else f"source == {json.dumps(filename, ensure_ascii=False)}"
            iterator = store.client.query_iterator(
                collection_name=self.collection_name,
                filter=expression,
                output_fields=OUTPUT_FIELDS,
                batch_size=config.QUERY_BATCH_SIZE,
                consistency_level="Strong",
                timeout=config.MILVUS_TIMEOUT,
            )
            documents = []
            try:
                while batch := iterator.next():
                    for row in batch:
                        metadata = {key: value for key, value in row.items() if key not in {"pk", "text"}}
                        documents.append(Document(
                            id=str(row["pk"]), page_content=row["text"], metadata=metadata,
                        ))
            finally:
                iterator.close()
            return documents

    def delete_chunks(self, ids: list[str]) -> None:
        if not ids:
            return
        with UPLOAD_LOCK:
            store = self._existing_store()
            if store is not None:
                store.delete(ids=ids, timeout=config.MILVUS_TIMEOUT)

    def search(self, query: str) -> list[tuple[Document, float]]:
        if not isinstance(query, str):
            raise TypeError("检索问题必须是字符串。")
        query = query.strip()
        if not query:
            raise ValueError("检索问题不能为空。")
        with UPLOAD_LOCK:
            store = self._existing_store()
            if store is None:
                return []
            results = store.similarity_search_with_score(
                query,
                k=config.K * 2,
                fetch_k=config.K,
                timeout=config.MILVUS_TIMEOUT,
                reranker=Function(
                    name="knowledge_rrf",
                    input_field_names=[],
                    function_type=FunctionType.RERANK,
                    params={"reranker": "rrf", "k": config.RRF_K},
                ),
                consistency_level="Strong",
            )
            # LangChain 0.4 将主键保存在 metadata，而没有设置 Document.id。
            return [
                (Document(
                    id=str(document.metadata["pk"]),
                    page_content=document.page_content,
                    metadata={key: value for key, value in document.metadata.items() if key != "pk"},
                ), float(score))
                for document, score in results
            ]
