"""Hippocampus 记忆系统 — LLM 客户端与 API 调用"""

import json

from openai import OpenAI

from hippocampus.config import MEMORY_CATEGORIES


# 简单的进程级客户端缓存（按 api_key）
_client_cache: dict[str, OpenAI] = {}


def get_client(api_key: str) -> OpenAI:
    """获取或创建 OpenAI 兼容客户端（指向 DeepSeek API）。"""
    if api_key not in _client_cache:
        _client_cache[api_key] = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com/v1"
        )
    return _client_cache[api_key]


def get_llm_response(client, messages, system_extra=""):
    if system_extra:
        enhanced = messages.copy()
        for i, msg in enumerate(enhanced):
            if msg["role"] == "system":
                enhanced[i] = {
                    "role": "system",
                    "content": msg["content"] + "\n\n" + system_extra
                }
                break
    else:
        enhanced = messages

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=enhanced,
        temperature=0.7,
        max_tokens=2000
    )
    return response.choices[0].message.content


def summarize_memory(client, question, answer):
    """一次 API 调用完成记忆摘要 + 分类。返回 (summary, category)。"""
    try:
        cats = ", ".join(MEMORY_CATEGORIES)
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": (
                    "你是一个专业的记忆压缩与分类助手。请将下面的问答对处理为 JSON 格式输出：\n"
                    "1. summary：不超过三句话的记忆摘要，保留核心事实、关键信息和用户意图。\n"
                    f"2. category：从 [{cats}] 中选一个最合适的分类。\n"
                    "直接输出 JSON，不要加 Markdown 代码块标记或其他说明。"
                )},
                {"role": "user", "content": f"用户问题：{question}\n\nAI回答：{answer}"}
            ],
            temperature=0.3,
            max_tokens=400
        )
        raw = response.choices[0].message.content.strip()
        parsed = json.loads(raw)
        summary = parsed.get("summary", "").strip()
        category = parsed.get("category", "").strip()
        if not summary:
            return None, None
        if category not in MEMORY_CATEGORIES:
            category = "其他"
        return summary, category
    except (json.JSONDecodeError, KeyError, AttributeError):
        try:
            text = raw.strip()
            if text:
                return text, "其他"
        except Exception:
            pass
        try:
            text = response.choices[0].message.content.strip()
            if text:
                return text, "其他"
        except Exception:
            return None, None
        return None, None
    except Exception:
        return None, None


def classify_text(client, text, mode="memory"):
    """对文本进行分类（mode='memory' 或 'question'）。"""
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
        return category if category in MEMORY_CATEGORIES else "其他"
    except Exception:
        return "其他"
