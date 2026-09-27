"""支持会话记忆和知识库工具的校园问答 Agent。"""

import sqlite3

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain.agents.middleware import SummarizationMiddleware

from JLU_agent.config import agent_config as config
from JLU_agent.schemas import agent_prompts as prompts
from JLU_agent.services.RAG.vector_store import VectorStoreService
from JLU_agent.tools.knowledge_tools import create_tools





class JLUChatAgent:
    """通过 thread_id 区分会话，由模型调用工具检索知识库。"""

    def __init__(self) -> None:
        # 初始化模型：实例创建时才读取密钥。
        model = init_chat_model(
            model=config.CHAT_MODEL_NAME,
            api_key=config.get_deepseek_api_key(),
            base_url=config.CHAT_MODEL_BASE_URL,
            timeout=config.CHAT_MODEL_TIMEOUT,
            max_retries=config.CHAT_MODEL_MAX_RETRIES,
        )

        # 每个 Agent 只创建一次检索服务，工具的定义和列表组装放在 tools 中。
        vector_store_service = VectorStoreService()
        tools = create_tools(vector_store_service)

        # 会话短期记忆管理,用sqlite存储会话记忆
        # 先确保目录存在，再使用配置中的数据库路径初始化 checkpointer。
        config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        checkpointer = SqliteSaver(
            sqlite3.connect(config.CHECKPOINT_DB_PATH, check_same_thread=False)
        )
        # 自动建表
        checkpointer.setup()

        # 自定义会话消息总结middleware
        summarize_middleware = SummarizationMiddleware(
            model=model,
            trigger=("messages", config.SUMMARIZE_TRIGGER_MESSAGES),   # 触发总结的消息条数
            keep=("messages", config.SUMMARIZE_KEEP_MESSAGES),         # 保留的最近消息条数
            summary_prompt=prompts.CHAT_SUMMARY_PROMPT,                # 使用中文总结
        )


        # 创建 Agent：系统提示词定义助手的身份和回答风格。
        self.agent = create_agent(
            model=model,
            tools=tools,
            system_prompt=prompts.CHAT_MODEL_SYSTEM_PROMPT,
            checkpointer=checkpointer,
            middleware=[summarize_middleware],
        )

    def chat(self, message: str, thread_id: str) -> str:
        """发送一个问题并返回回答；配置和接口异常由调用方处理。"""

        # 检查问题格式
        if not isinstance(message, str):
            raise TypeError("问题必须是字符串。")

        message = message.strip()       #洁净问题中的空格或换行符
        # 检查问题是否为空
        if not message:
            raise ValueError("问题不能为空，请输入问题后再试。")

        # 只提交当前问题；checkpointer 根据 thread_id 恢复历史，Agent 按需调用工具。
        result = self.agent.invoke(
            {"messages": [HumanMessage(message)]},
            config={
                "configurable":{
                    "thread_id": thread_id
                }
            }
        )

        # 提取最终回答：不向页面返回中间的工具调用消息。
        reply = result["messages"][-1]
        if not isinstance(reply, AIMessage):
            raise RuntimeError("模型没有返回助手消息，请重试。")

        answer = str(reply.text).strip()
        if not answer:
            raise RuntimeError("模型没有返回有效的回答，请重试。")
        return answer
