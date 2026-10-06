import datetime
import hashlib

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from JLU_agent.config import rag_config as config


class TextSplitterService:
    """将文档正文切片，生成稳定 ID 与去重元数据。"""

    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=config.SEPARATORS,
            length_function=len,
        )

    def split_text(self, text: str, filename: str) -> list[Document]:
        # 一份原文只有一个元数据字典，分割器会将它复制到每个片段。
        chunks = self.splitter.create_documents(
            texts=[text],
            metadatas=[
                {
                    "source": filename,
                    "create_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
            ],
        )

        # 同一份内容重试时沿用 ID，Milvus 可幂等补写缺失片段。
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        for index, chunk in enumerate(chunks):
            chunk.id = f"{content_hash}-{index}"
            chunk.metadata["id"] = chunk.id
            chunk.metadata["chunk_index"] = index
            chunk.metadata["content_hash"] = content_hash

        return chunks
