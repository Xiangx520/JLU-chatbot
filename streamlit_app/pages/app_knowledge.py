import streamlit as st
from JLU_agent.services.RAG.file_ls import FileLoaderAndSearchService


st.title("📚 知识加载")



# 一次上传一个 TXT 文件，文件暂存在内存中。
uploaded_file = st.file_uploader(
    "上传 TXT 文件",
    type=["txt"],
    accept_multiple_files=False,
)

#创建知识库的服务
if "file_ls_service" not in st.session_state:
    st.session_state["file_ls_service"] = FileLoaderAndSearchService()


if uploaded_file is not None:

    # get the file info
    file_name = uploaded_file.name
    file_type = uploaded_file.type
    file_size = uploaded_file.size / 1024

    # show infos
    st.subheader(f"file name: {file_name}")
    st.write(f"type: {file_type} | size: {file_size:.2f} KB")

    # utf-8-sig 同时兼容普通 UTF-8 和带 BOM 的 UTF-8 文件。
    try:
        text = uploaded_file.getvalue().decode("utf-8-sig")
    except UnicodeDecodeError:
        st.error("无法读取文件，请将 TXT 文件保存为 UTF-8 编码后重新上传。")
    else:
        if not text.strip():
            st.warning("文件内容为空，请选择包含文本的 TXT 文件。")
        else:
            # 以只读纯文本展示内容，保留原文换行。
            st.subheader("文件内容预览")
            st.code(text, language=None, wrap_lines=True, height=300)

            # load the file into stores
            with st.spinner("uploading file..."):
                result = st.session_state["file_ls_service"].upload_by_doc(
                    text, file_name
                )
                st.write(result)
