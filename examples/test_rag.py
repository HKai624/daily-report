"""RAG 建库与检索验证"""
import sys
sys.path.insert(0, ".")

from src.rag.index_builder import build_index, get_collection
from src.rag.retriever_engine import retrieve, knowledge_is_ready

print("构建知识库...")
n = build_index()
print(f"已建立 {n} 个文本块")

print("\n验证检索...")
docs = retrieve("考勤制度")
print(f"检索到 {len(docs)} 篇文档:")
for d in docs:
    print(f"  [{d['source']}] {d['content'][:80]}...")

print("\n验证路由检索...")
docs2 = retrieve("会议室怎么预定")
print(f"检索到 {len(docs2)} 篇文档:")
for d in docs2:
    print(f"  [{d['source']}] {d['content'][:80]}...")
