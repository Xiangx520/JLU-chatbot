import streamlit as st

from JLU_agent.services.RAG.file_ls import FileLoaderAndSearchService
from JLU_agent.services.RAG.parse_file import FileParseError, FileParser


def run_operation(action, success_message=None):
    """只在页面边界展示存储错误，失败不报告成功。"""
    try:
        with st.spinner("正在处理知识库……"):
            result = action()
    except Exception as exc:
        st.error(f"操作未全部完成：{exc}。请检查 Milvus 服务并重试。")
        return False
    st.session_state["knowledge_notice"] = success_message or result
    st.session_state["knowledge_reset_selection"] = True
    st.rerun()
    return True


def show_upload(service):
    st.subheader("上传文档")
    st.caption("同名文件内容变化时直接替换，只保留最新内容；相同内容不会重复入库。")
    uploaded_file = st.file_uploader(
        "上传知识文档（TXT、MD、PDF、DOCX）",
        type=["txt", "md", "pdf", "docx"],
        accept_multiple_files=False,
    )
    if uploaded_file is None:
        return

    st.write(f"文件名：{uploaded_file.name} | 大小：{uploaded_file.size / 1024:.2f} KB")
    try:
        text = FileParser().parse_file(uploaded_file.getvalue(), uploaded_file.name)
    except FileParseError as exc:
        st.error(str(exc))
        return

    st.code(text, language=None, wrap_lines=True, height=300)
    if st.button("上传到知识库"):
        run_operation(lambda: service.upload_by_doc(text, uploaded_file.name))


def show_documents(service):
    st.subheader("已入库文档")
    keyword = st.text_input("搜索文件名")
    try:
        documents = service.list_documents()
    except Exception as exc:
        st.error(f"读取文档列表失败：{exc}")
        return

    documents = [doc for doc in documents if keyword.casefold() in doc["filename"].casefold()]
    filenames = [doc["filename"] for doc in documents]
    if st.session_state.get("knowledge_selected") not in filenames:
        st.session_state.pop("knowledge_selected", None)
    if not documents:
        st.info("没有匹配的文档。" if keyword else "知识库暂无文档，请先上传。")
        return

    st.dataframe(
        [
            {"文件名": doc["filename"], "切片数量": doc["chunk_count"],
             "最近入库时间": doc["create_time"]}
            for doc in documents
        ],
        hide_index=True,
        width="stretch",
    )
    filename = st.selectbox("选择文档", filenames, index=None, key="knowledge_selected")
    if filename is None:
        return

    st.caption("以下为实际入库片段，按切片顺序展示；相邻片段可能包含重叠内容。")
    try:
        chunks = service.get_document_chunks(filename)
    except Exception as exc:
        st.error(f"读取文档片段失败：{exc}")
        return
    for number, chunk in enumerate(chunks, 1):
        with st.expander(f"片段 {number}"):
            st.code(chunk.page_content, language=None, wrap_lines=True)

    confirmed = st.checkbox(f"确认删除「{filename}」及其全部切片", key=f"delete_{filename}")
    if st.button("删除所选文档", disabled=not confirmed):
        run_operation(lambda: service.delete_document(filename), f"已删除「{filename}」。")


def main():
    st.title("📚 知识库管理")
    if st.session_state.pop("knowledge_reset_selection", False):
        filename = st.session_state.pop("knowledge_selected", None)
        st.session_state.pop(f"delete_{filename}", None)
    if notice := st.session_state.pop("knowledge_notice", None):
        st.success(notice)
    try:
        if "file_ls_service" not in st.session_state:
            st.session_state["file_ls_service"] = FileLoaderAndSearchService()
    except Exception as exc:
        st.error(f"知识库初始化失败：{exc}")
        return

    service = st.session_state["file_ls_service"]
    show_upload(service)
    show_documents(service)


main()
