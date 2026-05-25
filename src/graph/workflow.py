from typing import TypedDict, Annotated, Sequence
import operator
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage

from langchain_core.messages import AIMessage

from src.agents.query_rewriter import rewrite_query
from src.agents.router import route_query
from src.agents.retriever import retrieve_docs
from src.agents.solver import solve
from src.rag.retriever_engine import knowledge_is_ready
from src.memory.checkpoint_store import get_checkpointer


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    query: str
    rewritten_query: str
    route_decision: str
    retrieved_docs: list
    retrieved_context: str
    final_answer: str
    tool_calls: list
    tool_results: list
    knowledge_ready: bool


def rewriter_node(state: AgentState) -> dict:
    state["knowledge_ready"] = knowledge_is_ready()
    result = rewrite_query(dict(state))
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

    builder.add_node("rewriter", rewriter_node)
    builder.add_node("router", router_node)
    builder.add_node("retriever", retriever_node)
    builder.add_node("solver", solver_node)

    builder.set_entry_point("rewriter")
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
