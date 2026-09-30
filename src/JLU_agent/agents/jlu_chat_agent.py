"""支持会话记忆、知识库和网页搜索的校园问答 Agent。"""

import json
import sqlite3

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain.agents.middleware import SummarizationMiddleware

from JLU_agent.config import agent_config as config
from JLU_agent.schemas import agent_prompts as prompts
from JLU_agent.schemas.structured_output import AnswerInfo, Reference
from JLU_agent.services.RAG.file_ls import FileLoaderAndSearchService
from JLU_agent.tools.knowledge_tools import create_tools





class JLUChatAgent:
    """通过 thread_id 区分会话，由模型按需调用知识库和网页工具。"""

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
        file_ls_service = FileLoaderAndSearchService()
        tools = create_tools(file_ls_service)

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
        """兼容原有文本接口；配置和接口异常由调用方处理。"""
        return self.chat_with_sources(message, thread_id).answer

    def chat_with_sources(self, message: str, thread_id: str) -> AnswerInfo:
        """返回回答和本轮网页搜索来源；未联网时来源列表为空。"""

        # 检查问题格式
        if not isinstance(message, str):
            raise TypeError("问题必须是字符串。")

        message = message.strip()       #过滤问题中的空格或换行符
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
        messages = result["messages"]
        reply = messages[-1]
        if not isinstance(reply, AIMessage):
            raise RuntimeError("模型没有返回助手消息，请重试。")

        answer = str(reply.text).strip()
        if not answer:
            raise RuntimeError("模型没有返回有效的回答，请重试。")

        # checkpointer 会附带历史消息；只读取最后一条用户消息之后的工具输出。
        last_user_index = max(
            (index for index, item in enumerate(messages) if isinstance(item, HumanMessage)),
            default=len(messages),
        )
        references = []
        seen_urls = set()
        for item in messages[last_user_index + 1 :]:
            if not isinstance(item, ToolMessage) or item.name != "search_tavily_web":
                continue
            try:
                payload = json.loads(item.content)
            except (TypeError, ValueError):
                continue
            if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
                continue
            for page in payload["results"]:
                if not isinstance(page, dict):
                    continue
                title, url = page.get("title"), page.get("url")
                if not isinstance(title, str) or not isinstance(url, str) or url in seen_urls:
                    continue
                references.append(Reference(title=title, url=url))
                seen_urls.add(url)
        return AnswerInfo(answer=answer, reference=references)
