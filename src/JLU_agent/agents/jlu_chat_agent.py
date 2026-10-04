"""支持会话记忆、知识库和网页搜索的校园问答 Agent。"""

import json
import sqlite3
from collections.abc import Callable

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
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

    def stream_chat_with_sources(
        self, message: str, thread_id: str, on_text: Callable[[str], None]
    ) -> AnswerInfo:
        """实时回调累计正文，完成后返回最终回答和本轮网页来源。"""

        # 检查问题格式
        if not isinstance(message, str):
            raise TypeError("问题必须是字符串。")

        message = message.strip()       #过滤问题中的空格或换行符
        # 检查问题是否为空
        if not message:
            raise ValueError("问题不能为空，请输入问题后再试。")

        # 只提交当前问题；checkpointer 根据 thread_id 恢复历史，Agent 按需调用工具。
        messages = []           # 接收模型输出的最终结果
        text = ""               # 接收模型实时的流结果
        model_step = None       # 记录是第几轮调用模型
        calling_tool = False    # 记录是否是工具调用
        for mode, data in self.agent.stream(    # 这个for不是表示循环 而是不断取出流式调用结果 其中mode表示此时stream的流模式
            {"messages": [HumanMessage(message)]},
            config={
                "configurable":{
                    "thread_id": thread_id
                }
            },
            stream_mode=["messages", "values"], # 流模式，规定返回内容，前者是生成chunks，后者是生成最终结果
        ):
            # 如果流模式等于'生成最终结果'，更新messages记录最终结果
            if mode == "values":
                messages = data["messages"]
                continue

            # 如果流模式等于'生成chunks'，不断向前端输出chunks
            chunk, metadata = data
            # 略过主agent以外的agent输出消息
            if metadata.get("langgraph_node") != "model" or not isinstance(chunk, AIMessage):
                continue

            step = metadata["langgraph_step"]   # 记录是第几轮调用模型
            # 如果当前的轮次和前一次不同 清空上一轮缓存的消息
            if step != model_step:
                if text:
                    on_text("")
                text = ""
                calling_tool = False
                model_step = step
            if chunk.tool_calls or (
                isinstance(chunk, AIMessageChunk) and chunk.tool_call_chunks
            ):
                # 如果是工具调用的信息 清空缓存信息
                if not calling_tool:
                    on_text("")
                text = ""
                calling_tool = True
            # 向前端更新chunks，调用st.empty()
            if not calling_tool and chunk.text:
                text += chunk.text
                on_text(text)

        # 提取最终回答：不向页面返回中间的工具调用消息。
        reply = messages[-1] if messages else None
        # 确认取到的消息是最后一条消息 检车是否是aiMsg或者是不是工具调用消息
        if not isinstance(reply, AIMessage) or reply.tool_calls:
            raise RuntimeError("模型没有返回助手消息，请重试。")

        # 过滤空格及空消息
        answer = str(reply.text).strip()
        if not answer:
            raise RuntimeError("模型没有返回有效的回答，请重试。")

        # checkpointer 会附带历史消息；只读取最后一条用户消息之后的工具输出。
        # 找到最后一条用户消息 之后的消息才有意义
        last_user_index = max(
            (index for index, item in enumerate(messages) if isinstance(item, HumanMessage)),
            default=len(messages),
        )
        references = []
        seen_urls = set()
        # 获取可能的索引消息
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
        # 封装最终返回数据
        return AnswerInfo(answer=answer, reference=references)
