"""Hippocampus 记忆系统 — Streamlit UI 组件"""

import sys

import streamlit as st

from hippocampus import db, llm


def render_sidebar():
    """渲染侧边栏（API Key 输入、使用说明、记忆统计）。"""
    with st.sidebar:
        st.title("🧠 Hippocampus")
        st.markdown("---")
        st.subheader("⚙️ 设置")

        api_key = st.text_input(
            "DeepSeek API Key",
            type="password",
            placeholder="sk-...",
            help="输入您的 DeepSeek API Key，或放在项目 .env 文件中自动加载。Key 不会保存在本地，仅在本次会话中使用。",
            value=st.session_state.get("openai_api_key", "")
        )

        if api_key:
            st.session_state.openai_api_key = api_key
            try:
                client = llm.get_client(api_key)
                client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[{"role": "user", "content": "ping"}],
                    max_tokens=1
                )
                st.session_state.api_key_valid = True
                st.success("✅ API Key 验证通过")
            except Exception:
                st.session_state.api_key_valid = False
                st.error("❌ API Key 无效，请检查后重试")
        else:
            st.session_state.api_key_valid = False
            st.info("请在上面输入您的 DeepSeek API Key")

        st.markdown("---")
        st.markdown("### 💡 使用说明")
        st.markdown("""
        1. **输入 API Key**：在侧边栏输入，或放在项目 `.env` 文件中自动加载
        2. **开始聊天**：在底部输入框提问，系统会自动检索相关历史记忆
        3. **总结并记忆**：勾选后发送，本次问答会自动存入记忆库供后续检索
        """)

        st.markdown("---")
        st.markdown("### 🗂️ 记忆统计")
        try:
            collection = db.get_collection()
            st.markdown(f"**记忆条数：{collection.count()}**")
        except Exception:
            st.markdown("**记忆条数：无法获取**")

        return api_key


def render_chat_interface():
    """渲染聊天历史与已加载的相关记忆。"""
    for msg in st.session_state.messages:
        if msg["role"] == "system":
            continue
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if st.session_state.retrieved_memories:
        with st.expander("📖 已加载的相关记忆（点击展开）", expanded=False):
            for i, mem in enumerate(st.session_state.retrieved_memories, 1):
                tag = mem.get("category", "")
                tag_display = f" `[{tag}]`" if tag else ""
                st.markdown(f"**记忆 #{i}**{tag_display}")
                st.markdown(f"摘要：{mem['summary']}")
                st.markdown(f"来源问题：_{mem['source_question']}_")
                if mem.get("timestamp"):
                    st.markdown(f"时间：{mem['timestamp']}")
                st.markdown("---")


def handle_user_input(prompt, should_memorize, collection, api_key):
    """处理用户输入：自动检索 → 分类 → LLM 回答 → 可选存储。"""
    if not prompt:
        return

    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    client = llm.get_client(api_key)

    # 对当前问题分类，用于分层检索
    question_category = llm.classify_text(client, prompt, mode="question")

    # 自动检索（分层检索 + 遗忘曲线）
    retrieved = []
    try:
        memories, err = db.search_memories(collection, prompt, category=question_category)
        if memories:
            retrieved = memories
        elif err and "empty" not in err:
            print(f"[Hippocampus] 自动记忆检索出错: {err}", file=sys.stderr)
    except Exception as e:
        print(f"[Hippocampus] 自动记忆检索异常: {e}", file=sys.stderr)

    st.session_state.retrieved_memories = retrieved

    if retrieved:
        mids = [m["id"] for m in retrieved if "id" in m]
        db.update_last_accessed(collection, mids)

    # 构建系统提示
    system_extra = ""
    if retrieved:
        lines = "\n".join(f"- {m['summary']}" for m in retrieved)
        system_extra = (
            "以下是用户过去的相关记忆：\n"
            f"{lines}\n\n"
            "请基于用户的当前问题和上述记忆，给出回答。如果记忆不相关，可以忽略。"
        )

    # 调用 LLM
    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            response_text = llm.get_llm_response(client, st.session_state.messages, system_extra)
            if response_text:
                st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})

                if should_memorize:
                    with st.status("🧠 正在记忆本次对话...", expanded=False) as status:
                        st.write("正在生成记忆摘要并分类...")
                        summary, category = llm.summarize_memory(client, prompt, response_text)
                        if summary:
                            st.write(f"记忆摘要：{summary}")
                            st.write(f"分类：{category}")
                            st.write("正在存入长期记忆库...")
                            db.store_memory(collection, summary, prompt, category)
                            status.update(label="✅ 记忆已保存", state="complete")
                        else:
                            status.update(label="❌ 记忆生成失败", state="error")
