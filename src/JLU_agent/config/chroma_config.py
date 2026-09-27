from pathlib import Path
import os
from dotenv import dotenv_values

#向量数据库相关配置



# 根据当前文件定位项目根目录，不依赖运行命令时所在的目录。
PROJECT_ROOT = Path(__file__).resolve().parents[3]
# .env文件路径
ENV_FILE = PROJECT_ROOT / ".env"
# 存储数据的文件路径
DATA_DIR = PROJECT_ROOT / "src" / "JLU_agent" / "repo"

# md5文本文件地址
MD5_PATH = DATA_DIR / "md5.txt"

# 向量模型
DASHSCOPE_EMBEDDING_MODEL = "qwen3.7-text-embedding-flash"

def get_dashscope_api_key() -> str:
    """读取 dashscope 密钥，系统环境变量优先于项目的 .env 文件。"""
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if api_key is None:
        # 只取需要的配置，不把 .env 中的其他变量写入进程环境。
        api_key = dotenv_values(ENV_FILE, encoding="utf-8").get("DASHSCOPE_API_KEY")

    if api_key is None or not api_key.strip():
        raise ValueError(
            "未配置 DASHSCOPE_API_KEY，请在系统环境变量或项目根目录的 .env 中设置。"
        )
    return api_key.strip()


# chroma数据库表名
COLLECTION_NAME = "str_rag"         # 目前只支持文本格式的数据向量化
# chroma数据库地址
PERSIST_DIRECTORY = DATA_DIR / "chroma_db"

K = 3  # 每次检索返回的最大文本片段数
DATA_PATH = ""
MD5_HEX_STORE = ""
ALLOW_KNOWLEDGE_FILE_TYPE = []

# 文本切割器相关参数
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
SEPARATORS = ["\n\n", "\n", ".", "!", "?", "。", "！", "？", " ", ""]
