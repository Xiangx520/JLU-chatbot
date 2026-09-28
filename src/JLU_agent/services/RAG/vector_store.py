from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document


from JLU_agent.config import chroma_config as config





class VectorStoreService:

    def __init__(self):

        # 如果文件夹找不到则创建
        config.PERSIST_DIRECTORY.mkdir(parents=True, exist_ok=True)

        # 向量数据库实例
        self.chroma = Chroma(
            collection_name=config.COLLECTION_NAME,                     # 数据库表名
            embedding_function=DashScopeEmbeddings(                     # 向量模型
                model=config.DASHSCOPE_EMBEDDING_MODEL,
                dashscope_api_key=config.get_dashscope_api_key(),
            ),
            persist_directory=str(config.PERSIST_DIRECTORY)              # Chroma 接收字符串路径
        )


    def upload_into_chroma(self, chunks: list[Document]):
        """将传入的字符串进行向量化，存入向量数据库中"""

        self.chroma.add_documents(chunks)



    def search(self, query: str) -> list[Document]:
        """按问题检索已有知识，返回正文和来源，不重新写入数据。"""
        if not isinstance(query, str):
            raise TypeError("检索问题必须是字符串。")
        query = query.strip()
        if not query:
            raise ValueError("检索问题不能为空。")

        return self.chroma.similarity_search(query, k=config.K)
