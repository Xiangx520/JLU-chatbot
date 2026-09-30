import streamlit as st
from uuid import uuid4

from JLU_agent.agents.jlu_chat_agent import JLUChatAgent


def show_answer(content: str, references: list[dict[str, str]]) -> None:
    st.write(content)
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
        with st.spinner("模型正在思考……"):
            response = st.session_state["jlu_chat_agent"].chat_with_sources(
                question, st.session_state["qa_thread_id"]
            )

        references = [reference.model_dump() for reference in response.reference]
        show_answer(response.answer, references)

    st.session_state["qa_messages"].append(
        {"role": "assistant", "content": response.answer, "reference": references}
    )


