# 🧠 Agent Memory

通用Agent记忆系统，受mem0 (57k stars)启发，支持短期/长期/语义记忆。

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python" />
  <img src="https://img.shields.io/badge/ChromaDB-Vector-green" />
  <img src="https://img.shields.io/badge/License-MIT-yellow" />
</p>

## ✨ 特性

- 💾 短期记忆（对话窗口）
- 🗄️ 长期记忆（持久化存储）
- 🔍 语义搜索（向量检索）
- 📝 自动信息提取
- 📊 记忆统计
- 📤 导入/导出

## 🚀 快速开始

```bash
pip install chromadb  # 可选，支持向量搜索

python memory.py
```

## 📖 使用

```python
from memory import create_memory

# 创建记忆
memory = create_memory("my_memory.json")

# 添加记忆
memory.add("用户喜欢Python", memory_type="preference")
memory.add("用户在北京工作", memory_type="fact")

# 搜索记忆
results = memory.search("Python")

# 获取上下文（用于LLM提示）
context = memory.get_context("编程语言")

# 从对话中提取信息
memory.extract_and_store("我喜欢用Flask开发Web应用", "好的，已记住")

# 统计
print(memory.get_stats())
```

## 📁 存储后端

| 后端 | 说明 | 安装 |
|------|------|------|
| JsonMemoryStore | JSON文件存储 | 内置 |
| ChromaMemoryStore | ChromaDB向量存储 | `pip install chromadb` |

## 📄 许可证

MIT License
