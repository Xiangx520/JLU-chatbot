import os
from pathlib import Path

from dotenv import dotenv_values



# 根据当前文件定位项目根目录，不依赖运行命令时所在的目录。
PROJECT_ROOT = Path(__file__).resolve().parents[3]
# .env文件路径
ENV_FILE = PROJECT_ROOT / ".env"


# 聊天模型的相关配置信息
CHAT_MODEL_NAME = "deepseek-v4-pro"
CHAT_MODEL_BASE_URL = "https://api.deepseek.com"
CHAT_MODEL_TIMEOUT = 60
CHAT_MODEL_MAX_RETRIES = 2


# 会话总结相关配置信息
SUMMARIZE_MODEL_NAME = "deepseek-v4-pro"
SUMMARIZE_TRIGGER_MESSAGES = 20  # 触发总结的消息条数
SUMMARIZE_KEEP_MESSAGES = 6      # 保留原文的最近消息条数




def get_deepseek_api_key() -> str:
    """读取 DeepSeek 密钥，系统环境变量优先于项目的 .env 文件。"""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if api_key is None:
        # 只取需要的配置，不把 .env 中的其他变量写入进程环境。
        api_key = dotenv_values(ENV_FILE, encoding="utf-8").get("DEEPSEEK_API_KEY")

    if api_key is None or not api_key.strip():
        raise ValueError(
            "未配置 DEEPSEEK_API_KEY，请在系统环境变量或项目根目录的 .env 中设置。"
        )
    return api_key.strip()



# 会话记忆数据库的相关配置信息

# SQLite 短期记忆的存储目录和数据库文件路径。
CHECKPOINT_DIR = PROJECT_ROOT / "src" / "JLU_agent" / "repo" / "short-term_memory"
CHECKPOINT_DB_PATH = CHECKPOINT_DIR / "checkpoint.db"
CHAT_HISTORY_DB_PATH = CHECKPOINT_DIR / "chat_history.db"


# 检索关键词重写模型
REWRITE_MODEL_NAME = "deepseek-flash"


# Tavily联网工具
TAVILY_MAX_RESULTS = 5
TAVILY_SEARCH_DEPTH = "basic"


def get_tavily_api_key() -> str:
    """读取 Tavily 密钥，系统环境变量优先于项目的 .env 文件。"""
    api_key = os.getenv("TAVILY_API_KEY")
    if api_key is None:
        api_key = dotenv_values(ENV_FILE, encoding="utf-8").get("TAVILY_API_KEY")

    if api_key is None or not api_key.strip():
        raise ValueError(
            "未配置 TAVILY_API_KEY，请在系统环境变量或项目根目录的 .env 中设置。"
        )
    return api_key.strip()
