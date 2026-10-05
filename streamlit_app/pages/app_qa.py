from uuid import uuid4

import streamlit as st

from JLU_agent.agents.jlu_chat_agent import JLUChatAgent
from JLU_agent.services.chat_history import ChatHistoryService
from JLU_agent.ui.browser_identity import get_browser_identity


def show_references(references):
    if references:
        st.caption("搜索来源")
        for reference in references:
            st.link_button(reference["title"], reference["url"])


def show_messages(messages):
    for message in messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message["role"] == "assistant":
                show_references(message.get("reference", []))


def new_conversation():
    st.session_state["qa_thread_id"] = str(uuid4())
    st.session_state["qa_messages"] = []
    cancel_delete()


def cancel_delete():
    st.session_state.pop("qa_delete_thread", None)


def request_delete(thread_id):
    st.session_state["qa_delete_thread"] = thread_id


def delete_conversation(service, thread_id):
    try:
        service.delete_conversation(thread_id)
    except Exception as exc:
        st.session_state["qa_error"] = f"删除对话失败：{exc}"
        return
    cancel_delete()
    if st.session_state["qa_thread_id"] == thread_id:
        new_conversation()
    st.session_state["qa_notice"] = "对话及其上下文已删除。"


def restore_conversation(service, thread_id):
    try:
        messages = service.get_messages(thread_id)
        if not messages:
            raise ValueError("该对话已删除或不存在。")
        if not messages:
            raise ValueError("该对话已删除或不存在。")
    except Exception as exc:
        st.session_state["qa_error"] = f"恢复对话失败：{exc}"
        return
    st.session_state["qa_thread_id"] = thread_id
    st.session_state["qa_messages"] = messages
    cancel_delete()


def initialize_conversation(service):
    if st.session_state.get("qa_browser_id") == service.browser_id:
        return
    # 首次启用时保留当前页面记录；浏览器身份改变时不转移旧身份的记录。
    legacy = st.session_state.get("qa_messages", []) if "qa_browser_id" not in st.session_state else []
    if legacy:
        thread_id = st.session_state.get("qa_thread_id") or str(uuid4())
        service.import_conversation(thread_id, legacy)
        st.session_state["qa_thread_id"] = thread_id
    else:
        conversations = service.list_conversations()
        if conversations:
            st.session_state["qa_thread_id"] = conversations[0]["thread_id"]
        else:
            new_conversation()
    st.session_state["qa_browser_id"] = service.browser_id


def show_history(service, conversations):
    with st.sidebar:
        st.subheader("历史对话")
        st.button("新建对话", on_click=new_conversation)
        if not conversations:
            st.caption("发送第一个问题后，对话将自动保存。")
        for conversation in conversations:
            updated = conversation["updated_at"][:16].replace("T", " ")
            entry, delete = st.columns([4, 1])
            entry.button(
                f"{conversation['title']} · {updated}",
                key=f"history_{conversation['thread_id']}",
                on_click=restore_conversation,
                args=(service, conversation["thread_id"]),
                disabled=conversation["thread_id"] == st.session_state["qa_thread_id"],
                width="stretch",
            )
            delete.button(
                "删除", key=f"delete_{conversation['thread_id']}",
                on_click=request_delete, args=(conversation["thread_id"],),
            )
        pending = next(
            (conversation for conversation in conversations
             if conversation["thread_id"] == st.session_state.get("qa_delete_thread")),
            None,
        )
        if pending:
            st.warning(f"删除「{pending['title']}」？聊天记录和 Agent 上下文将永久删除。")
            st.button("确认删除", on_click=delete_conversation,
                      args=(service, pending["thread_id"]))
            st.button("取消", on_click=cancel_delete)


def answer_question(service, question):
    thread_id = st.session_state["qa_thread_id"]
    try:
        service.save_user_message(thread_id, question)
    except Exception as exc:
        st.error(f"问题未保存：{exc}。本次没有调用模型，请稍后重试。")
        return
    st.session_state["qa_messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        placeholder = st.empty()
        try:
            with st.spinner("正在检索并生成回答……"):
                if "jlu_chat_agent" not in st.session_state:
                    st.session_state["jlu_chat_agent"] = JLUChatAgent()
                response = st.session_state["jlu_chat_agent"].stream_chat_with_sources(
                    question, thread_id, placeholder.markdown
                )
        except Exception as exc:
            placeholder.empty()
            st.session_state["qa_error"] = f"回答生成失败：{exc}。问题已保存，未保存部分回答。"
            st.rerun()

        references = [reference.model_dump() for reference in response.reference]
        placeholder.markdown(response.answer)
        show_references(references)
        try:
            service.save_assistant_message(thread_id, response.answer, references)
        except Exception as exc:
            st.error(f"回答已生成，但未保存到历史记录：{exc}。请保留当前页面的回答。")
            return
    st.rerun()


def main():
    st.title("💬 聊天问答")
    identity = get_browser_identity()
    if identity is None:
        st.info("正在加载浏览器历史对话……")
        return
    if identity["error"]:
        st.error(identity["error"])
        return
    try:
        browser_id = identity["browser_id"]
        if st.session_state.get("qa_browser_id") != browser_id:
            st.session_state["qa_history_service"] = ChatHistoryService(browser_id)
        service = st.session_state["qa_history_service"]
        initialize_conversation(service)
        conversations = service.list_conversations()
        # 其他标签页删除了当前历史时，切换到新草稿，避免重新创建已删除的 ID。
        if st.session_state.get("qa_messages") and not any(
            conversation["thread_id"] == st.session_state["qa_thread_id"]
            for conversation in conversations
        ):
            new_conversation()
        st.session_state["qa_messages"] = service.get_messages(st.session_state["qa_thread_id"])
    except Exception as exc:
        st.error(f"读取历史对话失败：{exc}")
        return

    if error := st.session_state.pop("qa_error", None):
        st.error(error)
    if notice := st.session_state.pop("qa_notice", None):
        st.success(notice)
    show_history(service, conversations)
    show_messages(st.session_state["qa_messages"])
    question = st.chat_input("请输入你的问题")
    if question and question.strip():
        answer_question(service, question.strip())


main()
