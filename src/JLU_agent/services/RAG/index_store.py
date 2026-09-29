"""将文档块分词并保存到 BM25 索引。"""

import threading

import bm25s
import pkuseg
from langchain_core.documents import Document

from JLU_agent.config import chroma_config as config





# 上传服务和索引服务共用这把锁，避免同一进程内同时写入。
UPLOAD_LOCK = threading.RLock()


class IndexStoreService:
    """管理 BM25 索引的加载和入库。"""

    def __init__(self) -> None:
        self.seg = pkuseg.pkuseg()          # 中文分词器
        self.retriever = None

        with UPLOAD_LOCK:
            if config.INDEX_PATH.is_dir() and any(config.INDEX_PATH.iterdir()):
                self.retriever = bm25s.BM25.load(
                    config.INDEX_PATH, load_corpus=True, show_progress=False
                )

    def upload_into_bm5(self, chunks: list[Document]):
        """合并新旧文档块，重新建立并保存 BM25 索引。"""
        if not chunks:
            raise ValueError("chunks 不能为空")

        with UPLOAD_LOCK:
            # 每次读取磁盘上的最新数据，避免另一个页面会话上传后被旧数据覆盖。
            corpus_by_id = {}       # 索引库只能处理字典类型的数据 这里先把文档字典转换类型
            if config.INDEX_PATH.is_dir() and any(config.INDEX_PATH.iterdir()):
                latest = bm25s.BM25.load(
                    config.INDEX_PATH, load_corpus=True, show_progress=False
                )
                corpus_by_id = {item["id"]: item for item in latest.corpus}

            # 相同 ID 的文档块会覆盖旧块，因此重试不会增加重复块。
            for doc in chunks:
                doc_id = doc.id or doc.metadata.get("id")
                if not doc_id:
                    raise ValueError("文档块缺少 id")
                corpus_by_id[doc_id] = {
                    "id": doc_id,
                    "content": doc.page_content,
                    "source": doc.metadata.get("source"),
                    "chunk_index": doc.metadata.get("chunk_index"),
                    "create_time": doc.metadata.get("create_time"),
                }

            corpus = list(corpus_by_id.values())
            tokens = [self.seg.cut(item["content"]) for item in corpus]
            retriever = bm25s.BM25(k1=config.K1, b=config.B, corpus=corpus)
            retriever.index(tokens, show_progress=False)

            config.INDEX_PATH.mkdir(parents=True, exist_ok=True)
            retriever.save(config.INDEX_PATH, corpus=corpus, show_progress=False)
            self.retriever = retriever
            return self.retriever

    def search(self, query: str) -> list[tuple[Document, float]]:
        """按关键词检索文档块，返回文档及其 BM25 分数。"""
        if not isinstance(query, str):
            raise TypeError("检索问题必须是字符串。")
        query = query.strip()
        if not query:
            raise ValueError("检索问题不能为空。")

        with UPLOAD_LOCK:
            # 重新加载索引检索器的实例 避免为空或加载了旧索引
            if not config.INDEX_PATH.is_dir() or not any(config.INDEX_PATH.iterdir()):
                self.retriever = None
                return []
            self.retriever = bm25s.BM25.load(
                config.INDEX_PATH, load_corpus=True, show_progress=False
            )

            # 对问题分词 方便检索
            query_tokens = self.seg.cut(query)
            # 索引取块数
            k = min(config.K, len(self.retriever.corpus))
            if not query_tokens or k == 0:
                return []

            # 得到索引结果 类型为list[tuple[documents, scores]],因为bm25支持一次传递多个问题 所以返回的答案是多个 实际使用只取第一个列表数据
            results = self.retriever.retrieve(
                [query_tokens], k=k, show_progress=False
            )
            documents_with_scores = []
            for item, raw_score in zip(results.documents[0], results.scores[0]):
                score = float(raw_score)
                if score <= 0:      # 避免返回了完全不相关的数据
                    continue
                # 将bm25保存的字典数据组装成document
                document = Document(
                    id=item["id"],
                    page_content=item["content"],
                    metadata={
                        "source": item["source"],
                        "chunk_index": item["chunk_index"],
                        "create_time": item["create_time"],
                    },
                )
                documents_with_scores.append((document, score))
            return documents_with_scores
