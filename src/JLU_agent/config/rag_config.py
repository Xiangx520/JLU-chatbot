"""知识库模型、文本切分和 Milvus 连接配置。"""

import os
import re
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import dotenv_values

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ENV_FILE = PROJECT_ROOT / ".env"

DASHSCOPE_EMBEDDING_MODEL = "qwen3.7-text-embedding-flash"
RERANK_MODEL = "Qwen/Qwen3-Reranker-0.6B"
K = 3
RRF_K = 60
MILVUS_TIMEOUT = 30.0
QUERY_BATCH_SIZE = 1000
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
SEPARATORS = ["\n\n", "\n", ".", "!", "?", "。", "！", "？", " ", ""]
K1 = 1.5
B = 0.75


def _setting(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value is None:
        value = dotenv_values(ENV_FILE, encoding="utf-8").get(name)
    return default if value is None else value.strip()


def get_dashscope_api_key() -> str:
    api_key = _setting("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError(
            "未配置 DASHSCOPE_API_KEY，请在系统环境变量或项目根目录的 .env 中设置。"
        )
    return api_key


def _milvus_name(name: str, default: str) -> str:
    value = _setting(name, default)
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,254}", value):
        raise ValueError(f"{name} 必须以字母或下划线开头，仅包含字母、数字和下划线，最多 255 字符。")
    return value


def get_milvus_collection_name() -> str:
    return _milvus_name("MILVUS_COLLECTION_NAME", "jlu_knowledge")


def get_milvus_connection_args() -> dict:
    uri = _setting("MILVUS_URI", "http://localhost:19530")
    try:
        parsed = urlsplit(uri)
        valid = parsed.scheme in {"http", "https"} and bool(parsed.hostname)
        parsed.port  # 同时校验端口格式。
    except ValueError:
        valid = False
    if not valid:
        raise ValueError("MILVUS_URI 必须是有效的 http/https 服务地址，例如 http://localhost:19530。")
    args = {
        "uri": uri,
        "db_name": _milvus_name("MILVUS_DB_NAME", "default"),
        "timeout": MILVUS_TIMEOUT,
    }
    if token := _setting("MILVUS_TOKEN"):
        args["token"] = token
    return args
