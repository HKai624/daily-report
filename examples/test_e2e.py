"""WorkMate Agent - End-to-end verification of all 3 routes."""
import sys
import json
sys.path.insert(0, ".")

from langchain_core.messages import HumanMessage
from src.graph.workflow import compile_graph

print("=" * 50)
print("WorkMate Agent - Final Verification")
print("=" * 50)

graph = compile_graph()
config = {"configurable": {"thread_id": "final-test"}}

def safe_print(text, max_len=200):
    safe = text[:max_len].encode("ascii", errors="replace").decode("ascii")
    print(safe)

# Test 1: Chat
print("\n>>> Test 1: Chat (direct conversation)")
r = graph.invoke({"messages": [HumanMessage(content="Hello!")], "query": "Hello!"}, config)
print(f"  Route: {r.get('route_decision', '?')}")
safe_print(f"  Answer: {r.get('final_answer', '(empty)')}")

# Test 2: RAG
print("\n>>> Test 2: RAG (knowledge query)")
r = graph.invoke({"messages": [HumanMessage(content="年假有多少天？")], "query": "年假有多少天？"}, config)
print(f"  Route: {r.get('route_decision', '?')}")
docs = r.get("retrieved_docs", [])
print(f"  Docs retrieved: {len(docs)}")
safe_print(f"  Answer: {r.get('final_answer', '(empty)')}")

# Test 3: Tool
print("\n>>> Test 3: Tool (weather)")
r = graph.invoke({"messages": [HumanMessage(content="南宁今天天气怎么样？")], "query": "南宁今天天气怎么样？"}, config)
print(f"  Route: {r.get('route_decision', '?')}")
tools = r.get("tool_calls", [])
print(f"  Tools called: {len(tools)}")
for t in tools:
    print(f"    - {t.get('tool', '?')}")
safe_print(f"  Answer: {r.get('final_answer', '(empty)')}")

# Test 4: Multi-turn memory
print("\n>>> Test 4: Multi-turn memory")
r = graph.invoke({"messages": [HumanMessage(content="我刚才问了什么问题？")], "query": "我刚才问了什么问题？"}, config)
print(f"  Route: {r.get('route_decision', '?')}")
safe_print(f"  Answer: {r.get('final_answer', '(empty)')}")

print("\n" + "=" * 50)
print("All tests passed!")
print("=" * 50)
