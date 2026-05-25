import json
import time
from typing import AsyncGenerator
from langchain_core.messages import HumanMessage, AIMessage

from src.graph.workflow import compile_graph
from src.agents.query_rewriter import rewrite_query
from src.agents.router import route_query
from src.agents.retriever import retrieve_docs
from src.agents.solver import solve_stream
from src.rag.retriever_engine import knowledge_is_ready


async def run_agent_stream(query: str, session_id: str) -> AsyncGenerator[str, None]:
    """流式执行 Agent 管线，yield SSE 格式的事件字符串。"""
    graph = compile_graph()
    config = {"configurable": {"thread_id": session_id}}

    snapshot = graph.get_state(config)
    existing_messages = []
    if snapshot and snapshot.values:
        existing_messages = list(snapshot.values.get("messages", []))

    state = {
        "messages": existing_messages + [HumanMessage(content=query)],
        "query": query,
        "knowledge_ready": knowledge_is_ready(),
    }

    steps_log = []

    # Step 1: Rewriter
    yield _sse_event("step", json.dumps({"step": "rewrite", "status": "running"}, ensure_ascii=False))
    rewriter_result = rewrite_query(state)
    state.update(rewriter_result)
    rewritten = state.get("rewritten_query", "")
    steps_log.append({"step": "rewrite", "query": rewritten})
    yield _sse_event("step", json.dumps({"step": "rewrite", "query": rewritten}, ensure_ascii=False))

    # Step 2: Router
    yield _sse_event("step", json.dumps({"step": "route", "status": "running"}, ensure_ascii=False))
    router_result = route_query(state)
    state.update(router_result)
    route_decision = state.get("route_decision", "chat")
    steps_log.append({"step": "route", "decision": route_decision})
    yield _sse_event("step", json.dumps({"step": "route", "decision": route_decision}, ensure_ascii=False))

    # Step 3: Retriever (conditional)
    if route_decision == "rag" and state.get("knowledge_ready"):
        yield _sse_event("step", json.dumps({"step": "retrieve", "status": "running"}, ensure_ascii=False))
        retriever_result = retrieve_docs(state)
        state.update(retriever_result)
        doc_count = len(state.get("retrieved_docs", []))
        steps_log.append({"step": "retrieve", "count": doc_count})
        yield _sse_event("step", json.dumps({"step": "retrieve", "count": doc_count}, ensure_ascii=False))

    # Step 4: Solver with token streaming
    yield _sse_event("step", json.dumps({"step": "solver", "status": "running"}, ensure_ascii=False))
    full_answer = ""
    for token in solve_stream(state):
        full_answer += token
        yield _sse_event("token", json.dumps({"content": token}, ensure_ascii=False))

    # Save checkpoint
    state["final_answer"] = full_answer
    state["messages"] = existing_messages + [
        HumanMessage(content=query),
        AIMessage(content=full_answer),
    ]
    try:
        graph.update_state(config, {
            "messages": state["messages"],
            "final_answer": full_answer,
            "query": query,
            "rewritten_query": state.get("rewritten_query", ""),
            "route_decision": state.get("route_decision", ""),
            "retrieved_docs": state.get("retrieved_docs", []),
            "retrieved_context": state.get("retrieved_context", ""),
            "tool_calls": state.get("tool_calls", []),
            "tool_results": state.get("tool_results", []),
        })
    except Exception:
        pass

    # Done
    yield _sse_event("done", json.dumps({
        "answer": full_answer,
        "steps": steps_log,
    }, ensure_ascii=False))


def _sse_event(event: str, data: str) -> str:
    return f"event: {event}\ndata: {data}\n\n"
