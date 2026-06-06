"""
Agent Memory - 通用Agent记忆系统
受 mem0 (57k stars) 启发，支持短期/长期/语义记忆
"""

import json
import os
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

try:
    import chromadb
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False


class MemoryStore:
    """记忆存储基类"""
    def add(self, content: str, metadata: Dict = None) -> str:
        raise NotImplementedError

    def search(self, query: str, limit: int = 5) -> List[Dict]:
        raise NotImplementedError

    def get_all(self) -> List[Dict]:
        raise NotImplementedError

    def delete(self, memory_id: str) -> bool:
        raise NotImplementedError

    def clear(self):
        raise NotImplementedError


class JsonMemoryStore(MemoryStore):
    """JSON文件存储"""
    def __init__(self, path: str = "memory.json"):
        self.path = Path(path)
        self.memories: List[Dict] = []
        self._load()

    def _load(self):
        if self.path.exists():
            with open(self.path, 'r', encoding='utf-8') as f:
                self.memories = json.load(f)

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, 'w', encoding='utf-8') as f:
            json.dump(self.memories, f, ensure_ascii=False, indent=2)

    def add(self, content: str, metadata: Dict = None) -> str:
        memory_id = f"mem_{int(time.time() * 1000)}"
        memory = {
            "id": memory_id,
            "content": content,
            "metadata": metadata or {},
            "created_at": datetime.now().isoformat(),
            "access_count": 0
        }
        self.memories.append(memory)
        self._save()
        return memory_id

    def search(self, query: str, limit: int = 5) -> List[Dict]:
        query_lower = query.lower()
        results = []
        for mem in self.memories:
            if query_lower in mem["content"].lower():
                mem["access_count"] = mem.get("access_count", 0) + 1
                results.append(mem)
        results.sort(key=lambda x: x.get("access_count", 0), reverse=True)
        return results[:limit]

    def get_all(self) -> List[Dict]:
        return self.memories

    def delete(self, memory_id: str) -> bool:
        before = len(self.memories)
        self.memories = [m for m in self.memories if m["id"] != memory_id]
        if len(self.memories) < before:
            self._save()
            return True
        return False

    def clear(self):
        self.memories.clear()
        self._save()


class ChromaMemoryStore(MemoryStore):
    """ChromaDB向量存储"""
    def __init__(self, path: str = "chroma_db", collection: str = "memory"):
        if not CHROMA_AVAILABLE:
            raise RuntimeError("chromadb未安装")
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(
            name=collection,
            metadata={"hnsw:space": "cosine"}
        )

    def add(self, content: str, metadata: Dict = None) -> str:
        memory_id = f"mem_{int(time.time() * 1000)}"
        self.collection.add(
            documents=[content],
            ids=[memory_id],
            metadatas=[metadata or {}]
        )
        return memory_id

    def search(self, query: str, limit: int = 5) -> List[Dict]:
        results = self.collection.query(
            query_texts=[query],
            n_results=limit
        )
        return [
            {"id": id, "content": doc, "score": score}
            for id, doc, score in zip(
                results["ids"][0],
                results["documents"][0],
                results["distances"][0]
            )
        ]

    def get_all(self) -> List[Dict]:
        results = self.collection.get()
        return [
            {"id": id, "content": doc}
            for id, doc in zip(results["ids"], results["documents"])
        ]

    def delete(self, memory_id: str) -> bool:
        try:
            self.collection.delete(ids=[memory_id])
            return True
        except:
            return False

    def clear(self):
        # 删除集合并重建
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            metadata={"hnsw:space": "cosine"}
        )


class AgentMemory:
    """
    Agent记忆系统
    支持：短期记忆、长期记忆、语义搜索、自动摘要
    """

    def __init__(self, store: MemoryStore = None, persist_path: str = None):
        self.store = store or JsonMemoryStore(persist_path or "agent_memory.json")
        self.short_term: List[Dict] = []  # 当前对话
        self.context_window: int = 20  # 短期记忆窗口

    def add(self, content: str, memory_type: str = "fact", **metadata) -> str:
        """添加记忆"""
        meta = {"type": memory_type, **metadata}
        return self.store.add(content, meta)

    def remember_conversation(self, role: str, content: str):
        """记录对话"""
        self.short_term.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        # 保持窗口大小
        if len(self.short_term) > self.context_window:
            self.short_term = self.short_term[-self.context_window:]

    def search(self, query: str, limit: int = 5) -> List[Dict]:
        """语义搜索"""
        return self.store.search(query, limit)

    def get_context(self, query: str = "", include_short_term: bool = True) -> str:
        """获取上下文（用于LLM提示）"""
        parts = []

        # 短期记忆
        if include_short_term and self.short_term:
            recent = self.short_term[-6:]  # 最近3轮对话
            conv_text = "\n".join(f"{m['role']}: {m['content']}" for m in recent)
            parts.append(f"最近对话:\n{conv_text}")

        # 长期记忆（语义搜索）
        if query:
            long_term = self.search(query, limit=3)
            if long_term:
                mem_text = "\n".join(f"- {m['content']}" for m in long_term)
                parts.append(f"相关记忆:\n{mem_text}")

        return "\n\n".join(parts)

    def extract_and_store(self, user_text: str, assistant_text: str):
        """从对话中提取重要信息并存储"""
        # 检测用户偏好
        preference_patterns = [
            ("我喜欢", "preference"),
            ("我讨厌", "preference"),
            ("我想要", "goal"),
            ("我的目标是", "goal"),
            ("记住", "explicit"),
            ("不要忘记", "explicit"),
        ]

        for pattern, mem_type in preference_patterns:
            if pattern in user_text:
                idx = user_text.find(pattern) + len(pattern)
                content = user_text[idx:].strip()
                if content and len(content) < 200:
                    self.add(content, memory_type=mem_type, source="conversation")

        # 检测重要事实
        fact_patterns = ["我是", "我叫", "我在", "我的工作是"]
        for pattern in fact_patterns:
            if pattern in user_text:
                idx = user_text.find(pattern) + len(pattern)
                content = user_text[idx:].strip()
                if content and 2 < len(content) < 50:
                    self.add(f"{pattern}{content}", memory_type="fact", source="conversation")

    def get_stats(self) -> Dict:
        """获取统计信息"""
        all_memories = self.store.get_all()
        type_counts = {}
        for m in all_memories:
            mem_type = m.get("metadata", {}).get("type", "unknown")
            type_counts[mem_type] = type_counts.get(mem_type, 0) + 1

        return {
            "total_memories": len(all_memories),
            "short_term_count": len(self.short_term),
            "type_distribution": type_counts
        }

    def export(self, path: str):
        """导出记忆"""
        data = {
            "short_term": self.short_term,
            "long_term": self.store.get_all(),
            "exported_at": datetime.now().isoformat()
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def import_memories(self, path: str):
        """导入记忆"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for mem in data.get("long_term", []):
            content = mem.get("content", "")
            metadata = mem.get("metadata", {})
            self.store.add(content, metadata)


def create_memory(persist_path: str = None, use_chroma: bool = False) -> AgentMemory:
    """创建记忆实例"""
    if use_chroma and CHROMA_AVAILABLE:
        store = ChromaMemoryStore(persist_path or "chroma_db")
    else:
        store = JsonMemoryStore(persist_path or "agent_memory.json")
    return AgentMemory(store)


if __name__ == "__main__":
    memory = create_memory()

    print("Agent Memory System")
    print(f"Stats: {memory.get_stats()}")
    print()

    # 测试
    memory.add("用户喜欢Python", memory_type="preference")
    memory.add("用户在北京", memory_type="fact")

    results = memory.search("Python")
    print(f"Search 'Python': {len(results)} results")
    for r in results:
        print(f"  - {r['content']}")

    print()
    print(f"Context: {memory.get_context('Python')}")
