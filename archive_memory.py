"""
archive_memory.py — 记忆归档脚本

将 last_accessed 超过 30 天的记忆移至 archive_collection，
并从主集合中删除。

用法:
    python archive_memory.py
"""
from datetime import datetime

from hippocampus import db


def main():
    collection = db.get_collection()
    archive = db.get_archive_collection()

    all_data = collection.get()
    if not all_data or not all_data["ids"]:
        print("记忆库为空，无需归档。")
        return

    now = datetime.now()
    to_archive_ids = []
    to_archive_docs = []
    to_archive_metas = []

    for i, doc_id in enumerate(all_data["ids"]):
        meta = all_data["metadatas"][i] if all_data["metadatas"] else {}
        date_str = meta.get("last_accessed") or meta.get("timestamp")
        if not date_str:
            continue
        try:
            dt = datetime.fromisoformat(date_str)
            days_since = max(0.0, (now - dt).total_seconds() / 86400.0)
            if days_since > 30:
                to_archive_ids.append(doc_id)
                to_archive_docs.append(all_data["documents"][i])
                to_archive_metas.append(meta)
        except (ValueError, TypeError):
            continue

    if not to_archive_ids:
        print(f"没有超过 30 天未访问的记忆需要归档。当前记忆数：{len(all_data['ids'])}")
        return

    archive.add(
        ids=to_archive_ids,
        documents=to_archive_docs,
        metadatas=to_archive_metas
    )
    collection.delete(ids=to_archive_ids)

    print(f"已归档 {len(to_archive_ids)} 条记忆到 archive_collection。")
    print(f"主集合剩余 {collection.count()} 条记忆。")


if __name__ == "__main__":
    main()
