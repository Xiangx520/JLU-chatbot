import streamlit as st
from uuid import uuid4

from JLU_agent.agents.jlu_chat_agent import JLUChatAgent




# 在对话框展示agent回答
def show_answer(content: str, references: list[dict[str, str]]) -> None:
    st.write(content)
    show_references(references)

# 在对话框展示知识来源的网站信息
def show_references(references: list[dict[str, str]]) -> None:
    # 如果没有调用tavily则不展示
    if references:
        st.caption("搜索来源")
        for reference in references:
            st.link_button(reference["title"], reference["url"])



st.title("💬 聊天问答")


#session_state 初始化

# 创建聊天记录列表。
if "qa_messages" not in st.session_state:
    st.session_state["qa_messages"] = []

# 创建聊天机器人对象
if "jlu_chat_agent" not in st.session_state:
    st.session_state["jlu_chat_agent"] = JLUChatAgent()

# 每个浏览器会话只生成一次thread_id
if "qa_thread_id" not in st.session_state:
    st.session_state["qa_thread_id"] = str(uuid4())



# 每次重新运行页面，先展示已有的聊天记录。
for message in st.session_state["qa_messages"]:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            show_answer(message["content"], message.get("reference", []))
        else:
            st.write(message["content"])



# 接收新问题。
question = st.chat_input("请输入你的问题")

if question and question.strip():   #过滤空白消息
    question = question.strip()

    # 保存并展示用户消息。
    st.session_state["qa_messages"].append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.write(question)

    # 调用 Agent，展示并保存回答。
    with st.chat_message("assistant"):
        answer_placeholder = st.empty()     # 创建一个可以反复替换内容的占位区域
        with st.spinner("正在检索并生成回答……"):
            response = st.session_state["jlu_chat_agent"].stream_chat_with_sources(
                question, st.session_state["qa_thread_id"], answer_placeholder.markdown     # 将占位区域的方法引用传给对话方法，流式生成结果后都会更新到占位区域
            )

        # 拿到response里的索引内容
        references = [reference.model_dump() for reference in response.reference]
        # 输出response里的最终正文内容
        answer_placeholder.markdown(response.answer)
        # 输出索引
        show_references(references)

    # 更新历史对话
    st.session_state["qa_messages"].append(
        {"role": "assistant", "content": response.answer, "reference": references}
    )


