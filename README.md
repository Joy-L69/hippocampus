# Hippocampus 🧠

**一个拥有人脑式分级记忆的 AI 对话系统**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

## 💡 核心理念

当前的 AI 对话系统在回顾历史时，普遍采用"全量回顾"——每次推理都把整个历史记录塞进上下文。这导致：

- **Token 消耗巨大**：大量成本花在反复传输历史文本上
- **智能退化**：长上下文中的"迷失中间"现象，让模型逐渐丢失早期信息
- **隐私风险**：所有历史数据每次都要经过 API 传输

**Hippocampus 提出了一个不同的答案：像人脑一样记忆。**

人脑不会在每次思考时回顾一生，而是：

1. **优雅地遗忘**：不重要的细节自动沉底（遗忘曲线）
2. **按需检索**：只提取与当前问题最相关的记忆
3. **分级分类**：记忆按主题归类，检索范围精准缩小

## ✨ 核心功能

| 功能 | 说明 |
|---|---|
| **自动记忆检索** | 每次提问自动从历史记忆中检索最相关的 3 条 |
| **分层记忆树** | 记忆自动分类（个人偏好/项目工作/技术知识/日常闲聊/其他），检索时按类过滤 |
| **遗忘曲线** | 基于时间衰减的权重算法，久未访问的记忆自动降权，超过 9 天不注入上下文 |
| **记忆归档** | 30 天未访问的记忆可一键归档，保持主库精炼 |
| **零 Token 检索** | 语义搜索完全在本地 ChromaDB 完成，不消耗 API |
| **完全本地化** | 你的记忆数据只在你自己的硬盘上 |
| **.env 支持** | 可在 `.env` 文件中预先配置 API Key，免去每次手动输入 |

## 🏗️ 架构设计

```
┌─────────────────────────────────────┐
│           Streamlit UI              │
│  聊天界面  |  侧边栏  |  记忆展示    │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│         hippocampus/ 核心包          │
│                                     │
│  ┌─────────┐  ┌──────────────────┐  │
│  │ config  │  │      llm         │  │
│  │ 常量配置 │  │ DeepSeek API 调用 │  │
│  └─────────┘  └──────────────────┘  │
│                                     │
│  ┌────────────────────────────────┐  │
│  │         db (ChromaDB)          │  │
│  │ 存储 · 分层检索 · 遗忘曲线 · 归档 │  │
│  └────────────────────────────────┘  │
│                                     │
│  ┌────────────────────────────────┐  │
│  │         ui (Streamlit)         │  │
│  │  侧边栏 · 聊天渲染 · 输入处理   │  │
│  └────────────────────────────────┘  │
└─────────────────────────────────────┘
```

### 记忆处理流程

```
用户输入 → ① LLM 分类问题 → ② 按类别检索 ChromaDB
         → ③ 遗忘曲线重排序 → ④ 注入上下文
         → ⑤ LLM 回答 → ⑥ [可选] 摘要+分类 → 存入
```

## 🧰 技术栈

- **大模型 API**：DeepSeek Chat（OpenAI 兼容接口）
- **向量数据库**：ChromaDB（本地 ONNX 嵌入模型，无需外部 API）
- **用户界面**：Streamlit
- **语言**：Python 3.9+

## 🚀 快速开始

### 前提条件

- Python 3.9+
- 一个 [DeepSeek](https://platform.deepseek.com/) API Key

### 安装与运行

```bash
# 1. 克隆仓库
git clone https://github.com/你的用户名/hippocampus.git
cd hippocampus

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置 API Key（二选一）
# 方式 A：创建 .env 文件
echo "DEEPSEEK_API_KEY=sk-你的key" > .env

# 方式 B：启动后在侧边栏手动输入
# streamlit run app.py

# 4. 启动应用
streamlit run app.py
```

浏览器打开 `http://localhost:8501` 即可使用。

### 归档旧记忆（可选）

```bash
python archive_memory.py
```

## 📁 项目结构

```
hippocampus/
├── app.py                    # 入口文件（~80 行轻量组装）
├── archive_memory.py         # 记忆归档独立脚本
├── hippocampus/              # 核心包
│   ├── __init__.py
│   ├── config.py             # 常量与分类列表
│   ├── db.py                 # ChromaDB 操作（无 Streamlit 依赖）
│   ├── llm.py                # LLM 客户端与 API 调用
│   └── ui.py                 # Streamlit UI 组件
├── .env.example              # 环境变量模板
├── .gitignore
├── pyproject.toml            # 现代 Python 项目配置
├── requirements.txt          # 依赖清单
├── LICENSE                   # MIT 协议
└── README.md                 # 本文件
```

## 🧠 记忆系统详解

### 分层记忆树

记忆存入时自动被打上分类标签：

```
个人偏好 — "用户喜欢喝冰美式"
项目工作 — "项目 H 的数据库从 MySQL 迁移到了 PostgreSQL"
技术知识 — "Python 3.11 引入了 ExceptionGroup"
日常闲聊 — "用户上周去了趟杭州"
其他     — 无法归类的内容
```

检索时，先对当前问题分类，仅在同类别记忆中检索；无结果时自动回退全库。

### 遗忘曲线

每条记忆的权重由两个因子共同决定：

```
最终得分 = 语义相似度 × 遗忘系数

遗忘系数 = 1 / (1 + 距上次访问天数)

条件：遗忘系数 < 0.1（约 9 天未访问）→ 丢弃，不注入上下文
```

每次记忆被注入回答后，其 `last_accessed` 时间自动刷新。

### 记忆分类（LLM 辅助）

- **问题分类**：每次提问前，用 LLM 判断问题类别（用于检索过滤）
- **记忆分类**：存入记忆时，LLM 同时输出摘要和分类标签（一次 API 调用）

## 🤔 常见问题

**Q：第一次运行为什么慢？**
A：ChromaDB 首次使用会自动下载本地嵌入模型（约 80MB 的 ONNX 模型）。只需下载一次，后续秒级响应。

**Q：记忆存在哪里？**
A：存在项目根目录下的 `chromadb_data/` 文件夹中。已被 `.gitignore` 排除，不会提交到 Git。

**Q：可以换其他大模型吗？**
A：可以。修改 `hippocampus/llm.py` 中的 `base_url` 和 `model` 参数即可切换为 OpenAI 或其他兼容 API。

**Q：归档的记忆还能找回吗？**
A：可以。归档记忆存在同一个 ChromaDB 的 `archive_collection` 集合中，你可以编写查询脚本读取，或直接修改 `archive_memory.py` 的归档阈值。

## 📄 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 👤 作者

**Frankle**

> *"真正的智能，不该如此挥霍能量。"*
