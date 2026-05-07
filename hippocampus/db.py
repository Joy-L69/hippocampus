"""Hippocampus 记忆系统 — ChromaDB 数据库操作（无 Streamlit 依赖）"""

import os
import sys
import hashlib
from datetime import datetime

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction


def _project_root():
    """返回项目根目录（hippocampus/ 的父目录）。"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _generate_doc_id(question, timestamp):
    raw = f"{question}_{timestamp}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def _days_since(meta, now):
    date_str = meta.get("last_accessed") or meta.get("timestamp")
    if not date_str:
        return 0.0
    try:
        dt = datetime.fromisoformat(date_str)
        delta = now - dt
        return max(0.0, delta.total_seconds() / 86400.0)
    except (ValueError, TypeError):
        return 0.0


def get_collection():
    """获取 ChromaDB 记忆集合（纯函数，无 Streamlit 依赖）。"""
    db_path = os.path.join(_project_root(), "chromadb_data")
    os.makedirs(db_path, exist_ok=True)
    client = chromadb.PersistentClient(path=db_path)
    return client.get_or_create_collection(
        name="hippocampus_memories",
        embedding_function=DefaultEmbeddingFunction()
    )


def get_archive_collection():
    """获取归档集合。"""
    db_path = os.path.join(_project_root(), "chromadb_data")
    client = chromadb.PersistentClient(path=db_path)
    return client.get_or_create_collection(
        name="archive_collection",
        embedding_function=DefaultEmbeddingFunction()
    )


def store_memory(collection, summary, question, category=""):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    last_accessed = datetime.now().isoformat()
    doc_id = _generate_doc_id(question, timestamp)
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


def search_memories(collection, query, category=None, k=3):
    try:
        count = collection.count()
        if count == 0:
            return [], "empty"

        results = None
        if category:
            results = collection.query(
                query_texts=[query],
                n_results=min(k * 2, count),
                where={"category": category}
            )
            if not results or not results["documents"] or len(results["documents"][0]) == 0:
                results = None

        if results is None:
            results = collection.query(
                query_texts=[query],
                n_results=min(k * 2, count)
            )

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

            scored.sort(key=lambda x: x["score"], reverse=True)
            scored = scored[:k]

        return scored, None

    except Exception as e:
        return [], f"记忆检索失败：{str(e)}"


def update_last_accessed(collection, memory_ids):
    if not memory_ids:
        return
    try:
        now_str = datetime.now().isoformat()
        existing = collection.get(ids=memory_ids)
        merged = []
        for i, mem_id in enumerate(memory_ids):
            meta = dict(existing["metadatas"][i]) if existing and existing["metadatas"] and existing["metadatas"][i] else {}
            meta["last_accessed"] = now_str
            merged.append(meta)
        collection.update(ids=memory_ids, metadatas=merged)
    except Exception as e:
        print(f"[Hippocampus] 更新访问时间失败: {e}", file=sys.stderr)
