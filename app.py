import streamlit as st
import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from openai import OpenAI
import hashlib
from datetime import datetime
import os
import sys

# =============================================================================
# 页面配置
# =============================================================================
st.set_page_config(
    page_title="Hippocampus - 海马体记忆系统",
    page_icon="🧠",
    layout="wide"
)

# =============================================================================
# 初始化 ChromaDB（使用 @st.cache_resource 确保只初始化一次）
# =============================================================================
@st.cache_resource
def init_chromadb():
    """
    初始化 ChromaDB 持久化客户端与记忆集合。
    使用本地 ONNX 嵌入模型（all-MiniLM-L6-v2）进行向量化，无需外部 API。
    """
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chromadb_data")
    os.makedirs(db_path, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=db_path)
    collection = chroma_client.get_or_create_collection(
        name="hippocampus_memories",
        embedding_function=DefaultEmbeddingFunction()
    )
    return collection


# =============================================================================
# 初始化 Streamlit 会话状态
# =============================================================================
def init_session_state():
    """初始化所有需要用到的会话状态变量"""
    if "messages" not in st.session_state:
        # 默认系统提示词
        st.session_state.messages = [
            {"role": "system", "content": "你是一个智能、友善的助手。请用中文回答用户的问题，回答要准确、简洁、有条理。"}
        ]
    if "retrieved_memories" not in st.session_state:
        # 存储最近一次检索到的记忆列表
        st.session_state.retrieved_memories = []
    if "api_key_valid" not in st.session_state:
        # 标记 API Key 是否已验证通过
        st.session_state.api_key_valid = False


# =============================================================================
# 辅助函数
# =============================================================================
def get_openai_client():
    """从会话状态中获取 OpenAI 兼容客户端（指向 DeepSeek API），如果不存在则创建"""
    api_key = st.session_state.get("openai_api_key", "")
    if not api_key:
        return None
    if st.session_state.get("openai_client") is None:
        st.session_state.openai_client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com/v1"
        )
    return st.session_state.openai_client


def generate_doc_id(question, timestamp):
    """根据问题和时间戳生成唯一的文档 ID"""
    raw = f"{question}_{timestamp}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


# =============================================================================
# 分层记忆树 — 分类常量与辅助函数
# =============================================================================
MEMORY_CATEGORIES = ["个人偏好", "项目工作", "技术知识", "日常闲聊", "其他"]


def _days_since(meta, now):
    """从 metadata 中计算距离上次访问的天数。"""
    date_str = meta.get("last_accessed") or meta.get("timestamp")
    if not date_str:
        return 0.0
    try:
        dt = datetime.fromisoformat(date_str)
        delta = now - dt
        return max(0.0, delta.total_seconds() / 86400.0)
    except (ValueError, TypeError):
        return 0.0


def classify_text(client, text, mode="memory"):
    """
    调用 LLM 将文本分类到预定义类别。
    mode="memory": 对记忆摘要分类；mode="question": 对用户问题分类。
    """
    try:
        label = "记忆内容" if mode == "memory" else "用户问题"
        prompt = (
            f"请将以下{label}归类到以下类别中：\n"
            f"{MEMORY_CATEGORIES}\n"
            f"只返回类别名称，不要解释。\n"
            f"{label}：{text}"
        )
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=20
        )
        category = response.choices[0].message.content.strip()
        if category in MEMORY_CATEGORIES:
            return category
        return "其他"
    except Exception:
        return "其他"


def update_last_accessed(collection, memory_ids):
    """更新指定记忆的 last_accessed 字段为当前时间。"""
    if not memory_ids:
        return
    try:
        now_str = datetime.now().isoformat()
        collection.update(
            ids=memory_ids,
            metadatas=[{"last_accessed": now_str}] * len(memory_ids)
        )
    except Exception as e:
        print(f"[Hippocampus] 更新访问时间失败: {e}", file=sys.stderr)


# =============================================================================
# 核心记忆功能
# =============================================================================
def summarize_memory(client, question, answer):
    """
    调用 LLM 将一问一答总结为不超过三句话的记忆摘要。
    参数:
        client: DeepSeek 客户端实例
        question: 用户的问题
        answer: AI 的回答
    返回:
        字符串形式的记忆摘要
    """
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一个专业的记忆压缩助手。请将下面这段问答对总结为一段不超过三句话的记忆摘要。"
                                              "保留核心事实、关键信息和用户意图，去除冗余内容。直接输出摘要，不要加前缀。"},
                {"role": "user", "content": f"用户问题：{question}\n\nAI回答：{answer}"}
            ],
            temperature=0.3,
            max_tokens=300
        )
        summary = response.choices[0].message.content.strip()
        return summary
    except Exception as e:
        st.error(f"记忆总结失败：{str(e)}")
        return None


def store_memory(collection, summary, question, category=""):
    """
    将记忆摘要存入 ChromaDB 向量数据库，包含分类标签与访问时间。
    参数:
        collection: ChromaDB 集合
        summary: 记忆摘要文本
        question: 原始用户问题（作为元数据保存）
        category: 分类标签（取自 MEMORY_CATEGORIES）
    """
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        last_accessed = datetime.now().isoformat()
        doc_id = generate_doc_id(question, timestamp)
        collection.add(
            documents=[summary],
            metadatas=[{
                "question": question,
                "timestamp": timestamp,
                "category": category,
                "last_accessed": last_accessed
            }],
            ids=[doc_id]
        )
        return True
    except Exception as e:
        st.error(f"记忆存储失败：{str(e)}")
        return False


def search_memories(collection, query, category=None, k=3):
    """
    分层检索：优先按 category 过滤，回退全库检索。
    对结果应用遗忘曲线重排序，丢弃 weight < 0.1 的记忆。

    参数:
        collection: ChromaDB 集合
        query: 用户的查询文本
        category: 可选的分类过滤
        k: 返回的记忆条数（默认为 3）
    返回:
        元组 (记忆列表, 错误信息或 None)
    """
    try:
        count = collection.count()
        if count == 0:
            return [], '记忆库中还没有任何记忆，先聊聊天并勾选「总结并记忆」来积累记忆吧。'

        # 第 1 步：尝试按类别过滤检索
        results = None
        if category:
            results = collection.query(
                query_texts=[query],
                n_results=min(k * 2, count),  # 多取一些，给遗忘曲线留裁剪空间
                where={"category": category}
            )
            if not results or not results["documents"] or len(results["documents"][0]) == 0:
                results = None

        # 第 2 步：无类别 / 类别无结果 → 全库检索
        if results is None:
            results = collection.query(
                query_texts=[query],
                n_results=min(k * 2, count)
            )

        # 第 3 步：解析结果，应用遗忘系数
        now = datetime.now()
        scored = []
        if results and results["documents"] and len(results["documents"]) > 0:
            for i in range(len(results["documents"][0])):
                doc_id = results["ids"][0][i]
                doc = results["documents"][0][i]
                meta = results["metadatas"][0][i] if results["metadatas"] else {}
                distance = results["distances"][0][i] if results.get("distances") else 0.0

                days = _days_since(meta, now)
                weight = 1.0 / (1.0 + days)

                if weight < 0.1:
                    continue

                similarity = 1.0 / (1.0 + distance)
                combined = similarity * weight

                scored.append({
                    "id": doc_id,
                    "summary": doc,
                    "source_question": meta.get("question", "未知来源"),
                    "timestamp": meta.get("timestamp", ""),
                    "category": meta.get("category", ""),
                    "score": combined
                })

            # 按综合得分降序排列，取 top-k
            scored.sort(key=lambda x: x["score"], reverse=True)
            scored = scored[:k]

        return scored, None

    except Exception as e:
        return [], f"记忆检索失败：{str(e)}"


# =============================================================================
# LLM 对话函数
# =============================================================================
def get_llm_response(client, messages, system_extra=""):
    """
    调用 DeepSeek Chat 模型获取回复。
    参数:
        client: DeepSeek 客户端（OpenAI 兼容）
        messages: 对话消息列表
        system_extra: 附加到系统提示词中的额外内容（如检索到的记忆）
    返回:
        字符串形式的 AI 回复
    """
    try:
        # 如果提供了额外的系统提示内容，注入到第一条 system 消息中
        if system_extra:
            enhanced_messages = messages.copy()
            for i, msg in enumerate(enhanced_messages):
                if msg["role"] == "system":
                    enhanced_messages[i] = {
                        "role": "system",
                        "content": msg["content"] + "\n\n" + system_extra
                    }
                    break
        else:
            enhanced_messages = messages

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=enhanced_messages,
            temperature=0.7,
            max_tokens=2000
        )
        return response.choices[0].message.content
    except Exception as e:
        # 临时：直接抛出原始错误，以便在终端查看完整报错
        raise e


# =============================================================================
# 侧边栏 UI
# =============================================================================
def render_sidebar():
    """渲染侧边栏，包含 API Key 输入和应用信息"""
    with st.sidebar:
        st.title("🧠 Hippocampus")
        st.markdown("---")
        st.subheader("⚙️ 设置")

        # API Key 输入框
        api_key = st.text_input(
            "DeepSeek API Key",
            type="password",
            placeholder="sk-...",
            help="输入您的 DeepSeek API Key。Key 不会保存在本地，仅在本次会话中使用。",
            value=st.session_state.get("openai_api_key", "")
        )

        if api_key:
            st.session_state.openai_api_key = api_key
            # 尝试验证 API Key
            try:
                test_client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
                test_client.chat.completions.create(
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
        1. **输入 API Key**：在侧边栏填入你的 DeepSeek Key
        2. **开始聊天**：在底部输入框提问，系统会自动检索相关历史记忆
        3. **总结并记忆**：勾选后发送，本次问答会自动存入记忆库供后续检索
        """)

        st.markdown("---")
        st.markdown("### 🗂️ 记忆统计")
        try:
            collection = init_chromadb()
            memory_count = collection.count()
            st.markdown(f"**记忆条数：{memory_count}**")
        except Exception:
            st.markdown("**记忆条数：无法获取**")

        return api_key


# =============================================================================
# 主聊天界面
# =============================================================================
def render_chat_interface():
    """渲染主聊天区域，包括消息历史和输入控件"""
    # 显示历史消息（跳过 system 消息）
    for msg in st.session_state.messages:
        if msg["role"] == "system":
            continue
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 如果最近检索过记忆，在界面上显示
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


# =============================================================================
# 消息发送与处理逻辑
# =============================================================================
def handle_user_input(prompt, should_memorize, collection, openai_client):
    """
    处理用户发送的消息：自动检索相关记忆、调用 LLM、记忆存储等。
    """
    if not prompt:
        return

    # 1. 显示用户消息
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. 对当前问题分类，用于分层检索
    question_category = classify_text(openai_client, prompt, mode="question")

    # 3. 自动检索相关记忆（分层检索 + 遗忘曲线）
    retrieved = []
    try:
        memories, error = search_memories(collection, prompt, category=question_category)
        if memories:
            retrieved = memories
        elif error and "记忆库中还没有任何记忆" not in error:
            print(f"[Hippocampus] 自动记忆检索出错: {error}", file=sys.stderr)
    except Exception as e:
        print(f"[Hippocampus] 自动记忆检索异常: {e}", file=sys.stderr)

    st.session_state.retrieved_memories = retrieved

    # 更新被注入记忆的访问时间
    if retrieved:
        memory_ids = [mem["id"] for mem in retrieved if "id" in mem]
        update_last_accessed(collection, memory_ids)

    # 4. 构建系统提示词（注入检索到的记忆作为背景知识）
    system_extra = ""
    if retrieved:
        memory_lines = "\n".join(f"- {mem['summary']}" for mem in retrieved)
        system_extra = (
            "以下是用户过去的相关记忆：\n"
            f"{memory_lines}\n\n"
            "请基于用户的当前问题和上述记忆，给出回答。如果记忆不相关，可以忽略。"
        )

    # 5. 调用 LLM 获取回复
    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            response_text = get_llm_response(openai_client, st.session_state.messages, system_extra)
            if response_text:
                st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})

                # 6. 如果用户勾选了"总结并记忆"，则进行记忆摘要、分类和存储
                if should_memorize:
                    with st.status("🧠 正在记忆本次对话...", expanded=False) as status:
                        st.write("正在生成记忆摘要...")
                        summary = summarize_memory(openai_client, prompt, response_text)
                        if summary:
                            st.write(f"记忆摘要：{summary}")
                            st.write("正在分类记忆...")
                            category = classify_text(openai_client, summary, mode="memory")
                            st.write(f"分类：{category}")
                            st.write("正在存入长期记忆库...")
                            success = store_memory(collection, summary, prompt, category)
                            if success:
                                status.update(label="✅ 记忆已保存", state="complete")
                            else:
                                status.update(label="❌ 记忆保存失败", state="error")
                        else:
                            status.update(label="❌ 记忆生成失败", state="error")


# =============================================================================
# 主程序入口
# =============================================================================
def main():
    """应用程序主入口：组装所有 UI 组件并协调交互逻辑"""
    # 初始化
    init_session_state()

    # 初始化 ChromaDB
    try:
        collection = init_chromadb()
    except Exception as e:
        st.error(f"向量数据库初始化失败：{str(e)}")
        st.stop()

    # 渲染侧边栏
    api_key = render_sidebar()

    # 如果 API Key 无效，阻止后续操作
    if not st.session_state.api_key_valid:
        st.info("请先在侧边栏输入您的 DeepSeek API Key 以开始使用。")
        st.stop()

    # 获取 OpenAI 客户端
    openai_client = get_openai_client()
    if openai_client is None:
        st.error("DeepSeek 客户端初始化失败，请检查 API Key。")
        st.stop()

    # ===== 主聊天区域（显示历史消息） =====
    render_chat_interface()

    # ===== 底部控制区：记忆开关 + 输入框 =====
    st.markdown("---")

    should_memorize = st.checkbox(
        "🧠 总结并记忆这次问答",
        help="勾选后，发送消息时会自动将本次问答总结为记忆存入向量数据库"
    )

    # 聊天输入框
    prompt = st.chat_input("请输入您的问题...")

    # ===== 处理用户发送的消息 =====
    if prompt:
        handle_user_input(prompt, should_memorize, collection, openai_client)

    # 显示记忆统计信息
    st.caption(f"💾 当前记忆库共 {collection.count()} 条记忆")


# =============================================================================
# 启动应用
# =============================================================================
if __name__ == "__main__":
    main()
