"""WorkMate Agent 对话测试脚本"""
import sys
import os
import io

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from langchain_core.messages import HumanMessage
from src.graph.workflow import compile_graph
from src.rag.index_builder import build_index
from src.rag.retriever_engine import knowledge_is_ready


def test_chat():
    print("=" * 60)
    print("WorkMate Agent · 测试脚本")
    print("=" * 60)

    print("\n[1] 检查知识库状态...")
    if not knowledge_is_ready():
        print("  知识库为空，尝试构建...")
        n = build_index()
        print(f"  已构建知识库，共 {n} 个文本块")
    else:
        print("  知识库已就绪")

    print("\n[2] 编译 Agent 图...")
    graph = compile_graph()
    print("  Agent 图编译完成")

    config = {"configurable": {"thread_id": "test-session-001"}}

    test_queries = [
        "你好，我是新员工小王，请问公司考勤制度是怎样的？",
        "帮我查一下北京今天的天气",
        "帮我创建一个待办：明天下午3点参加项目评审会",
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n[3.{i}] 用户: {query}")
        state = {"messages": [HumanMessage(content=query)], "query": query}
        result = graph.invoke(state, config)
        route = result.get("route_decision", "?")
        answer = result.get("final_answer", "(无回答)")

        print(f"  路由: {route}")
        docs = result.get("retrieved_docs", [])
        if docs:
            print(f"  检索到 {len(docs)} 篇文档")
        tools = result.get("tool_calls", [])
        if tools:
            print(f"  调用了 {len(tools)} 个工具: {[t['tool'] for t in tools]}")
        print(f"  助手: {answer[:200]}{'...' if len(answer) > 200 else ''}")

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    test_chat()
