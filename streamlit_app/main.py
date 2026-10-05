import streamlit as st

# 设置整个应用的布局，并默认展开侧边栏。
st.set_page_config(
    page_title="吉大校园助手",            #浏览器标题
    page_icon="🎓",
    layout="centered",                  #居中布局
    initial_sidebar_state="expanded",   #展示侧栏
)


def show_home():
    # 只有选择“主页”时，才执行这里的内容。
    st.title("吉大校园助手")

    # 介绍这个 Agent 项目。
    st.write(
        "吉大校园助手是一个面向吉林大学学生的 AI Agent 项目\n\n"
        "希望通过自然语言对话，为你提供便捷的问答体验。"
    )

    # 展示计划中的功能池。
    st.subheader("计划中的功能")
    st.write("**💬 聊天问答**：在聊天界面输入问题，与 AI 助手进行对话并获取回答。")
    st.write("**📚 知识库管理**：上传、查看、更新和删除校园知识文档。")

    # 说明当前开发进度。
    st.info("通过侧边栏进入聊天问答或知识库管理。")


# 注册页面，并在侧边栏显示导航选项。
page = st.navigation(
    [
        st.Page(show_home, title="主页", icon="🎓", default=True),                         #主页默认
        st.Page("pages/app_qa.py", title="聊天问答", icon="💬", url_path="app_qa"),   #注册聊天页面
        st.Page(                                                                          #数据上传
            "pages/app_knowledge.py",
            title="知识库管理",
            icon="📚",
            url_path="app_knowledge",
        ),
    ],
    position="sidebar",
)

# 执行当前选中的页面。
page.run()
