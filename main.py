import os
import sys
import uuid
import json
import asyncio
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langchain_core.messages import HumanMessage

from src.graph.workflow import compile_graph
from src.graph.streaming import run_agent_stream
from src.rag.index_builder import build_index
from src.rag.retriever_engine import knowledge_is_ready
from src.memory.checkpoint_store import session_store
from src.config import DEEPSEEK_API_KEY

load_dotenv()

app = FastAPI(title="WorkMate Agent", version="2.0.0")

graph = compile_graph()


class AskRequest(BaseModel):
    question: str
    session_id: str = "default"


class SessionCreate(BaseModel):
    title: str = ""


# ── Static files ──────────────────────────────────────────────
@app.get("/")
async def index():
    return FileResponse("index.html")


# ── System status ─────────────────────────────────────────────
@app.get("/api/status")
async def status():
    return {
        "llm_connected": bool(DEEPSEEK_API_KEY),
        "model": "deepseek-chat",
        "knowledge_ready": knowledge_is_ready(),
        "sessions_count": len(session_store.list_all()),
        "version": "2.0.0",
    }


# ── Knowledge base upload ─────────────────────────────────────
@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename or not file.filename.endswith(".txt"):
        raise HTTPException(400, "只支持 .txt 文件")

    content = await file.read()
    file_path = os.path.join(
        os.path.dirname(__file__), "data", "knowledge", file.filename
    )
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(content)

    try:
        count = build_index()
        return {"msg": "知识库已就绪", "chunks": count}
    except Exception as e:
        raise HTTPException(500, f"索引构建失败: {e}")


# ── Non-streaming chat ────────────────────────────────────────
@app.post("/api/ask")
async def ask(req: AskRequest):
    if not knowledge_is_ready():
        raise HTTPException(400, "知识库未就绪，请先上传文档")

    config = {"configurable": {"thread_id": req.session_id}}

    snapshot = graph.get_state(config)
    existing_messages = []
    if snapshot and snapshot.values:
        existing_messages = list(snapshot.values.get("messages", []))

    is_first = len(existing_messages) == 0
    user_msg = HumanMessage(content=req.question.strip())

    try:
        # operator.add 自动追加到已有消息，只需传新消息
        result = graph.invoke(
            {"messages": [user_msg], "query": req.question.strip()},
            config,
        )

        answer = result.get("final_answer", "抱歉，处理出现错误。")

        session_store.touch(
            req.session_id,
            message_count=len(existing_messages) + 2,
        )
        if is_first:
            session_store.set_title_from_first_message(req.session_id, req.question)

        return {
            "answer": answer,
            "route_decision": result.get("route_decision", ""),
            "rewritten_query": result.get("rewritten_query", ""),
            "retrieved_count": len(result.get("retrieved_docs", [])),
            "tool_calls": [t.get("tool", "") for t in result.get("tool_calls", [])],
        }

    except Exception as e:
        raise HTTPException(500, f"处理请求失败: {e}")


# ── Streaming chat (SSE) ──────────────────────────────────────
@app.get("/api/ask/stream")
async def ask_stream(question: str, session_id: str = "default"):
    if not knowledge_is_ready():
        raise HTTPException(400, "知识库未就绪，请先上传文档")

    if not question.strip():
        raise HTTPException(400, "问题不能为空")

    # Ensure session exists
    if not session_store.get(session_id):
        session_store.create(session_id)

    # Check if this is the first message to set title
    snapshot = graph.get_state({"configurable": {"thread_id": session_id}})
    existing_messages = []
    if snapshot and snapshot.values:
        existing_messages = list(snapshot.values.get("messages", []))
    is_first = len(existing_messages) == 0
    if is_first:
        session_store.set_title_from_first_message(session_id, question)

    async def event_generator():
        try:
            async for sse_chunk in run_agent_stream(question.strip(), session_id):
                yield sse_chunk.encode("utf-8")
        except Exception as e:
            error_data = json.dumps({"error": str(e)}, ensure_ascii=False)
            yield f"event: error\ndata: {error_data}\n\n".encode("utf-8")

        session_store.touch(
            session_id,
            message_count=len(existing_messages) + 2,
        )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ── Session management ────────────────────────────────────────
@app.get("/api/sessions")
async def list_sessions():
    return {"sessions": session_store.list_all()}


@app.post("/api/sessions")
async def create_session(req: SessionCreate):
    sid = uuid.uuid4().hex
    session_store.create(sid, req.title or "新对话")
    return session_store.get(sid)


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    session_store.delete(session_id)
    return {"msg": "已删除"}


@app.get("/api/sessions/{session_id}/history")
async def get_history(session_id: str):
    config = {"configurable": {"thread_id": session_id}}
    snapshot = graph.get_state(config)
    messages = []
    if snapshot and snapshot.values:
        for msg in snapshot.values.get("messages", []):
            role = "user" if getattr(msg, "type", "human") == "human" else "assistant"
            content = msg.content if hasattr(msg, "content") else str(msg)
            messages.append({"role": role, "content": content})
    return {"session_id": session_id, "messages": messages}


# ── Startup ───────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    print(f"WorkMate Agent v2.0.0 启动中...")
    print(f"LLM: DeepSeek (deepseek-chat)")
    print(f"知识库: {'已就绪' if knowledge_is_ready() else '未建库'}")
    print(f"访问: http://127.0.0.1:8000")

    uvicorn.run(app, host="127.0.0.1", port=8000)
