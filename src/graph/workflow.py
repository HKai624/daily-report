from typing import TypedDict, Annotated, Sequence, Optional
import operator
import logging
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage

from langchain_core.messages import AIMessage

from src.agents.query_rewriter import rewrite_query
from src.agents.router import route_query
from src.agents.retriever import retrieve_docs
from src.agents.solver import solve
from src.rag.retriever_engine import knowledge_is_ready
from src.memory.checkpoint_store import get_checkpointer
from src.vision_adapter import handle_message_with_image

thought_logger = logging.getLogger("agent_thought")


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    query: str
    image_url: Optional[str]
    image_description: Optional[str]
    rewritten_query: str
    route_decision: str
    retrieved_docs: list
    retrieved_context: str
    final_answer: str
    tool_calls: list
    tool_results: list
    knowledge_ready: bool


def vision_node(state: AgentState) -> dict:
    image_url = state.get("image_url", "") or ""
    query = state.get("query", "")
    messages = state.get("messages", [])

    if not image_url:
        thought_logger.info("vision_node: 无图片，跳过视觉分析")
        return {"image_description": ""}

    thought_logger.info(f"vision_node: 检测到图片 {image_url[:80]}...，调用智谱视觉分析")
    try:
        enhanced = handle_message_with_image(query, image_url)
        return {
            "query": enhanced,
            "image_description": enhanced,
        }
    except Exception as e:
        thought_logger.error(f"vision_node: 视觉分析失败: {e}")
        fallback = (
            f"用户说：'{query}'。"
            f"（图片识别暂时不可用，请直接输入文字描述）"
        )
        return {
            "query": fallback,
            "image_description": "",
        }


def rewriter_node(state: AgentState) -> dict:
    state["knowledge_ready"] = knowledge_is_ready()
    # 如果 vision_node 增强了 query（拼入了图片描述），确保 rewriter 使用增强后的版本
    enhanced_query = state.get("query", "")
    result = rewrite_query(dict(state), current_query=enhanced_query)
    return {"rewritten_query": result.get("rewritten_query", "")}


def router_node(state: AgentState) -> dict:
    result = route_query(dict(state))
    return {"route_decision": result.get("route_decision", "chat")}


def retriever_node(state: AgentState) -> dict:
    result = retrieve_docs(dict(state))
    return {
        "retrieved_docs": result.get("retrieved_docs", []),
        "retrieved_context": result.get("retrieved_context", ""),
    }


def solver_node(state: AgentState) -> dict:
    result = solve(dict(state))
    answer = result.get("final_answer", "")
    return {
        "final_answer": answer,
        "tool_calls": result.get("tool_calls", []),
        "tool_results": result.get("tool_results", []),
        "messages": [AIMessage(content=answer)],
    }


def route_after_router(state: AgentState) -> str:
    decision = state.get("route_decision", "chat")
    if decision == "rag" and state.get("knowledge_ready", False):
        return "retriever"
    return "solver"


def build_agent_graph():
    builder = StateGraph(AgentState)

    builder.add_node("vision", vision_node)
    builder.add_node("rewriter", rewriter_node)
    builder.add_node("router", router_node)
    builder.add_node("retriever", retriever_node)
    builder.add_node("solver", solver_node)

    builder.set_entry_point("vision")
    builder.add_edge("vision", "rewriter")
    builder.add_edge("rewriter", "router")
    builder.add_conditional_edges("router", route_after_router, {
        "retriever": "retriever",
        "solver": "solver",
    })
    builder.add_edge("retriever", "solver")
    builder.add_edge("solver", END)

    return builder


def compile_graph():
    builder = build_agent_graph()
    checkpointer = get_checkpointer()
    return builder.compile(checkpointer=checkpointer)
