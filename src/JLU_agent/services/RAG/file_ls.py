from langchain_core.documents import Document

from JLU_agent.services.RAG.index_store import IndexStoreService, UPLOAD_LOCK
from JLU_agent.services.RAG.md5_str_file import Md5Service
from JLU_agent.services.RAG.text_splitter import TextSplitterService
from JLU_agent.services.RAG.vector_store import VectorStoreService
from JLU_agent.services.RAG.docs_reranker import rerank




class FileLoaderAndSearchService:
    """协调文档上传和双库检索。"""

    def __init__(self):
        self.md5Service = Md5Service()                      # MD5文件过滤器
        self.textSplitterService = TextSplitterService()    # 文本切割器
        self.vectorStoreService = VectorStoreService()      # 向量数据库
        self.indexStoreService = IndexStoreService()        # 索引数据库

    def upload_by_doc(self, data: str, filename: str) -> str:
        """依次写入 Chroma、BM25，全部成功后才记录文件摘要。"""
        # 多个 Streamlit 会话共享此锁，避免用旧索引覆盖其他会话刚上传的数据。
        with UPLOAD_LOCK:
            md5_hex = self.md5Service.get_string_md5(data)
            if self.md5Service.check_md5(md5_hex):
                return "[pass]the data has been uploaded!"

            # 文本切片
            chunks = self.textSplitterService.split_text(data, filename)
            # 知识入库
            self.vectorStoreService.upload_into_chroma(chunks)
            self.indexStoreService.upload_into_bm5(chunks)
            # 记录成功处理的结果
            self.md5Service.save_md5(md5_hex)
            return "[success]the data has been uploaded!"

    def search(self, query: str) -> list[tuple[Document, float]]:
        """融合向量检索和 BM25 检索的结果。"""
        # 稠密检索
        vector_results = self.vectorStoreService.search(query)
        # 稀疏检索
        bm25_results = self.indexStoreService.search(query)
        # 重排序
        reranked_results = rerank(query, vector_results, bm25_results)

        return reranked_results
