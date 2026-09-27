from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import datetime


from JLU_agent.config import chroma_config as config
from JLU_agent.services.RAG.md5_str_file import Md5Service


class VectorStoreService:

    def __init__(self):
        # 加载md5处理器
        self.Md5Service=Md5Service()

        # 如果文件夹找不到则创建
        config.PERSIST_DIRECTORY.mkdir(parents=True, exist_ok=True)

        # 向量数据库实例
        self.chroma = Chroma(
            collection_name=config.COLLECTION_NAME,                                          # 数据库表名
            embedding_function=DashScopeEmbeddings(                                          # 向量模型
                model=config.DASHSCOPE_EMBEDDING_MODEL,
                dashscope_api_key=config.get_dashscope_api_key(),
            ),
            persist_directory=str(config.PERSIST_DIRECTORY)                                  # Chroma 接收字符串路径
        )

        # 文本分割器实例
        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,           # 分割文档的最大长度
            chunk_overlap=config.CHUNK_OVERLAP,     # 滑动窗口大小
            separators=config.SEPARATORS,           # 递归划分符号
            length_function=len                     # 长度统计依据
        )

    def search(self, query: str) -> list[Document]:
        """按问题检索已有知识，返回正文和来源，不重新写入数据。"""
        if not isinstance(query, str):
            raise TypeError("检索问题必须是字符串。")
        query = query.strip()
        if not query:
            raise ValueError("检索问题不能为空。")

        return self.chroma.similarity_search(query, k=config.K)

    def upload_by_str(self, data, filename):
        """将传入的字符串进行向量化，存入向量数据库中"""

        #去重判断
        md5_hex = Md5Service.get_string_md5(data)
        if Md5Service.check_md5(md5_hex): return "[pass]the data has been uploaded!"

        #文本切割
        chunks = self.spliter.split_text(data)

        #存入向量数据库
        metadata = {
            "source": filename,
            "create_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "operator": "admin"
        }

        self.chroma.add_texts(
            chunks,
            metadatas=[metadata for _ in chunks],
        )

        #记录处理后的数据
        Md5Service.save_md5(md5_hex)

        return "[success]the data has been uploaded!"
