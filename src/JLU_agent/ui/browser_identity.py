"""通过浏览器本地存储，在刷新和新标签页之间保留历史归属。"""

from streamlit.components.v2 import component


_identity = component(
    "jlu_browser_identity",
    js="""
    export default function({ setStateValue }) {
        try {
            const key = "jlu_chat_browser_id";
            let browserId = localStorage.getItem(key);
            if (!browserId) {
                browserId = crypto.randomUUID();
                localStorage.setItem(key, browserId);
                browserId = localStorage.getItem(key);
                if (!browserId) throw new Error("storage unavailable");
            }
            setStateValue("identity", { browser_id: browserId, error: null });
        } catch (error) {
            setStateValue("identity", {
                browser_id: null,
                error: "浏览器本地存储不可用，无法识别历史对话。请允许站点存储后刷新。"
            });
        }
    }
    """,
)


def get_browser_identity() -> dict | None:
    result = _identity(key="qa_browser_identity", height=0, on_identity_change=lambda: None)
    return result.identity
