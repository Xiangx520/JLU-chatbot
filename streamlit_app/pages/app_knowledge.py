import streamlit as st
from JLU_agent.services.RAG.file_ls import FileLoaderAndSearchService
from JLU_agent.services.RAG.parse_file import FileParseError, FileParser


st.title("📚 知识加载")



# 文件暂存在内存中，选择文件只解析和预览。
uploaded_file = st.file_uploader(
    "上传知识文档（TXT、MD、PDF、DOCX）",
    type=["txt", "md", "pdf", "docx"],
    accept_multiple_files=False,
)

if uploaded_file is not None:

    # get the file info
    file_name = uploaded_file.name
    file_type = uploaded_file.type
    file_size = uploaded_file.size / 1024

    # show infos
    st.subheader(f"file name: {file_name}")
    st.write(f"type: {file_type} | size: {file_size:.2f} KB")

    try:
        text = FileParser().parse_file(uploaded_file.getvalue(), file_name)
    except FileParseError as exc:
        st.error(str(exc))
    else:
        st.subheader("文件内容预览")
        st.code(text, language=None, wrap_lines=True, height=300)

        if st.button("上传到知识库"):
            with st.spinner("正在上传到知识库..."):
                if "file_ls_service" not in st.session_state:
                    st.session_state["file_ls_service"] = FileLoaderAndSearchService()
                result = st.session_state["file_ls_service"].upload_by_doc(
                    text, file_name
                )
                st.write(result)
