"""
Hippocampus 🧠 — 像人脑一样分级记忆的 AI 对话系统

使用方式:
    streamlit run app.py
"""
import os
import streamlit as st
from dotenv import load_dotenv

from hippocampus import config, db, ui

# 加载 .env 文件（如果有）
load_dotenv()


@st.cache_resource
def init_chromadb():
    """Streamlit 缓存的 ChromaDB 初始化。"""
    return db.get_collection()


def init_session_state():
    """初始化会话状态，从 .env 自动加载 API Key。"""
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "system", "content": "你是一个智能、友善的助手。请用中文回答用户的问题，回答要准确、简洁、有条理。"}
        ]
    if "retrieved_memories" not in st.session_state:
        st.session_state.retrieved_memories = []
    if "api_key_valid" not in st.session_state:
        st.session_state.api_key_valid = False

    env_key = os.getenv("DEEPSEEK_API_KEY", "")
    if env_key and not st.session_state.get("openai_api_key"):
        st.session_state.openai_api_key = env_key


def main():
    st.set_page_config(
        page_title="Hippocampus - 海马体记忆系统",
        page_icon="🧠",
        layout="wide"
    )

    init_session_state()

    try:
        collection = init_chromadb()
    except Exception as e:
        st.error(f"向量数据库初始化失败：{str(e)}")
        st.stop()

    api_key = ui.render_sidebar()

    if not st.session_state.api_key_valid:
        st.info("请先在侧边栏输入您的 DeepSeek API Key 以开始使用。")
        st.stop()

    ui.render_chat_interface()

    st.markdown("---")
    should_memorize = st.checkbox(
        "🧠 总结并记忆这次问答",
        help="勾选后，发送消息时会自动将本次问答总结为记忆存入向量数据库"
    )
    prompt = st.chat_input("请输入您的问题...")

    if prompt:
        ui.handle_user_input(prompt, should_memorize, collection, api_key)

    st.caption(f"💾 当前记忆库共 {collection.count()} 条记忆")


if __name__ == "__main__":
    main()
