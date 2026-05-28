import os
import sys
import uuid
import json
import asyncio
import time
import threading
from datetime import datetime
from collections import defaultdict

from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Query, Form
from fastapi.responses import FileResponse, StreamingResponse, PlainTextResponse
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
from src.config import (
    DEEPSEEK_API_KEY,
    FEISHU_APP_ID,
    FEISHU_APP_SECRET,
    FEISHU_VERIFICATION_TOKEN,
    FEISHU_ENCRYPT_KEY,
    SERVER_HOST,
    SERVER_PORT,
    DEBUG_MODE,
    RATE_LIMIT_PER_USER,
    PUBLIC_BASE_URL,
    UPLOAD_DIR,
)
from src.feishu.crypto import FeishuMsgCrypt
from src.logging_setup import setup_logging
from src.session_manager import get_user_session, USER_SESSIONS, cleanup_expired_sessions
from src.tool_governance import execute_tool_with_permission_check
from src.metrics import metrics
from src.thought_recorder import ThoughtRecorder
from src.retry_utils import call_llm_with_fallback
from src.pre_launch import run_pre_launch_checks
import httpx
import logging

load_dotenv()

# ── 日志初始化 ────────────────────────────────────────────────
setup_logging()

app = FastAPI(title="WorkMate Agent", version="2.0.0")

# ── 静态文件服务（图片上传目录）────────────────────────────────
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

graph = compile_graph()

# ── 会话锁：确保同一会话串行执行，避免上下文串台 ──────────────
session_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

# ── 简易频率限制 ──────────────────────────────────────────────
rate_limit_store: dict[str, list[float]] = defaultdict(list)


def check_rate_limit(user_id: str, limit: int = RATE_LIMIT_PER_USER) -> bool:
    now = time.time()
    window = 60.0
    rate_limit_store[user_id] = [
        t for t in rate_limit_store[user_id] if now - t < window
    ]
    if len(rate_limit_store[user_id]) >= limit:
        return False
    rate_limit_store[user_id].append(now)
    return True


# ── 简单认证 ──────────────────────────────────────────────────
AUTH_USERS = {
    "admin": {"password": "admin", "role": "admin"},
    "hk": {"password": "hk123456", "role": "feishu_user", "feishu_user_id": "ou_1061605d1e908bd38c523215faeb7428"},
}

auth_tokens: dict[str, dict] = {}


# ── 飞书 API 辅助 ──────────────────────────────────────────────
_tenant_token: dict = {"token": "", "expires_at": 0}
_token_lock = asyncio.Lock()


async def get_tenant_access_token() -> str:
    """获取/刷新飞书 tenant_access_token，带缓存"""
    async with _token_lock:
        now = time.time()
        if _tenant_token["token"] and now < _tenant_token["expires_at"] - 60:
            return _tenant_token["token"]

        url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                url,
                json={"app_id": FEISHU_APP_ID, "app_secret": FEISHU_APP_SECRET},
                timeout=10,
            )
            data = resp.json()
            if data.get("code") != 0:
                raise Exception(f"获取 token 失败: {data}")
            _tenant_token["token"] = data["tenant_access_token"]
            _tenant_token["expires_at"] = now + data.get("expire", 7200)
            logging.getLogger("agent_thought").info("飞书 tenant_access_token 已刷新")
            return _tenant_token["token"]


def _strip_markdown_for_feishu(text: str) -> str:
    """洗掉 Markdown 标记，输出适合飞书文本消息的纯文本"""
    import re

    lines = text.split("\n")
    result: list[str] = []
    in_table = False
    table_rows: list[list[str]] = []
    pending = False

    for line in lines:
        stripped = line.strip()

        # 代码块（跳过）
        if stripped.startswith("```"):
            continue

        # 表格分隔行
        if re.match(r"^\|[-:\s|]+\|$", stripped):
            continue

        # 表格行
        if stripped.startswith("|") and stripped.endswith("|"):
            cells = [c.strip() for c in stripped.split("|")[1:-1]]
            table_rows.append(cells)
            in_table = True
            continue

        # 表格结束，输出为键值对
        if in_table:
            in_table = False
            if table_rows:
                headers = table_rows[0] if len(table_rows) > 0 else []
                for row in table_rows[1:]:
                    for i, cell in enumerate(row):
                        if i < len(headers):
                            result.append(f"  • {headers[i]}：{cell}")
                table_rows = []
            pending = True

        # 标题 → 【标题】
        m = re.match(r"^#{1,4}\s+(.*)", stripped)
        if m:
            if pending:
                result.append("")
                pending = False
            result.append(f"【{m.group(1)}】")
            pending = True
            continue

        # 水平线
        if re.match(r"^---+$", stripped):
            if pending:
                result.append("")
                pending = False
            result.append("—" * 20)
            pending = True
            continue

        # 无序列表
        m = re.match(r"^[-*]\s+(.*)", stripped)
        if m:
            if pending:
                result.append("")
                pending = False
            result.append(f"  • {m.group(1)}")
            continue

        # 有序列表
        m = re.match(r"^\d+\.\s+(.*)", stripped)
        if m:
            if pending:
                result.append("")
                pending = False
            result.append(f"  {m.group(1)}")
            continue

        # 普通行：去掉行内 ** 和 ` 标记
        cleaned = re.sub(r"\*\*(.+?)\*\*", r"\1", stripped)
        cleaned = re.sub(r"\*(.+?)\*", r"\1", cleaned)
        cleaned = re.sub(r"`([^`]+)`", r"\1", cleaned)

        if pending:
            result.append("")
            pending = False
        result.append(cleaned)

    if in_table and table_rows:
        headers = table_rows[0] if len(table_rows) > 0 else []
        for row in table_rows[1:]:
            for i, cell in enumerate(row):
                if i < len(headers):
                    result.append(f"  • {headers[i]}：{cell}")

    text = "\n".join(result)
    # 全局兜底：去残留的 ** * 标记
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    # 去多余空行
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


async def send_feishu_message(chat_id: str, text: str):
    """通过飞书 API 发送消息，自动洗掉 Markdown 标记"""
    logger = logging.getLogger("agent_thought")
    token = await get_tenant_access_token()

    url = (
        "https://open.feishu.cn/open-apis/im/v1/messages"
        "?receive_id_type=chat_id"
    )

    cleaned = _strip_markdown_for_feishu(text)
    body = {
        "receive_id": chat_id,
        "msg_type": "text",
        "content": json.dumps({"text": cleaned}),
    }
    logger.info(f"发送飞书消息: chat_id={chat_id}, len={len(text)}")

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            url,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=15,
        )
        data = resp.json()
        logger.info(f"飞书发送响应: status={resp.status_code}, body={json.dumps(data, ensure_ascii=False)}")

        if resp.status_code >= 400 or data.get("code") != 0:
            logger.error(
                f"飞书消息发送失败! status={resp.status_code}, code={data.get('code')}, "
                f"msg={data.get('msg')}, chat_id={chat_id}"
            )
        return data


async def download_feishu_file(message_id: str, file_key: str, file_name: str) -> str:
    """下载飞书文件并保存到知识库目录，返回本地路径"""
    token = await get_tenant_access_token()
    url = f"https://open.feishu.cn/open-apis/im/v1/messages/{message_id}/resources/{file_key}?type=file"
    headers = {"Authorization": f"Bearer {token}"}

    knowledge_dir = os.path.join(os.path.dirname(__file__), "data", "knowledge")
    os.makedirs(knowledge_dir, exist_ok=True)
    safe_name = file_name or f"feishu_file_{int(time.time())}.txt"
    local_path = os.path.join(knowledge_dir, safe_name)

    logger = logging.getLogger("agent_thought")
    logger.info(f"下载飞书文件: {file_name} → {local_path}")

    async with httpx.AsyncClient() as client:
        resp = await client.get(url, headers=headers, timeout=30)
        if resp.status_code == 200:
            with open(local_path, "wb") as f:
                f.write(resp.content)
            logger.info(f"飞书文件已保存: {local_path} ({len(resp.content)} bytes)")
            return local_path
        else:
            logger.error(f"下载飞书文件失败: status={resp.status_code}, body={resp.text[:200]}")
            raise Exception(f"下载文件失败: {resp.status_code}")


async def download_feishu_image(message_id: str, image_key: str) -> str:
    """下载飞书图片并保存到上传目录，返回公网可访问 URL"""
    token = await get_tenant_access_token()
    url = f"https://open.feishu.cn/open-apis/im/v1/messages/{message_id}/resources/{image_key}?type=image"
    headers = {"Authorization": f"Bearer {token}"}

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    safe_name = f"feishu_img_{image_key}.jpg"
    local_path = os.path.join(UPLOAD_DIR, safe_name)

    logger = logging.getLogger("agent_thought")
    logger.info(f"下载飞书图片: image_key={image_key} → {local_path}")

    async with httpx.AsyncClient() as client:
        resp = await client.get(url, headers=headers, timeout=30)
        if resp.status_code == 200:
            with open(local_path, "wb") as f:
                f.write(resp.content)
            public_url = f"{PUBLIC_BASE_URL}/uploads/{safe_name}"
            logger.info(f"飞书图片已保存: {local_path} ({len(resp.content)} bytes)")
            return public_url
        else:
            logger.error(f"下载飞书图片失败: status={resp.status_code}, body={resp.text[:200]}")
            raise Exception(f"下载图片失败: {resp.status_code}")


# ── 统一消息处理（Webhook 和长连接复用） ──────────────────────
async def process_message(event_data):
    """统一的消息处理入口，集成会话治理、指标、思考记录
    支持两种输入：飞书 SDK 的 EventContext 对象、Webhook 的 dict
    """
    import logging
    thought_logger = logging.getLogger("agent_thought")

    # 兼容 EventContext 对象和 dict 两种格式
    msg_type = ""
    file_key = ""
    file_name = ""
    image_key = ""
    if hasattr(event_data, "event"):
        # 飞书 SDK EventContext 对象（长连接模式）
        event = event_data.event
        sender_id_obj = event.sender.sender_id
        user_id = sender_id_obj.user_id or sender_id_obj.open_id or "unknown"
        message = event.message
        chat_id = message.chat_id or user_id
        msg_content = message.content or "{}"
        msg_type = getattr(message, "msg_type", "") or getattr(message, "message_type", "")
        file_key = getattr(message, "file_key", "") or ""
        file_name = getattr(message, "file_name", "") or ""
        image_key = getattr(message, "image_key", "") or ""
    elif isinstance(event_data, dict):
        # Webhook JSON dict
        event = event_data.get("event", {})
        sender = event.get("sender", {}).get("sender_id", {})
        user_id = sender.get("user_id") or sender.get("open_id", "unknown")
        message = event.get("message", {})
        chat_id = message.get("chat_id", user_id)
        msg_content = message.get("content", "{}")
        msg_type = message.get("msg_type", "") or message.get("message_type", "")
        file_key = message.get("file_key", "")
        file_name = message.get("file_name", "")
        image_key = message.get("image_key", "")
    else:
        thought_logger.error(f"未知的消息格式: {type(event_data)}")
        return {"code": -1, "msg": "unknown message format"}

    # 文件消息 → 下载并入库 RAG 知识库
    if msg_type == "file" and file_key:
        message_id = message.message_id if hasattr(message, "message_id") else message.get("message_id", "")
        try:
            local_path = await download_feishu_file(message_id, file_key, file_name)
            build_index()
            thought_logger.info(f"飞书文件已入库: {local_path}")
            await send_feishu_message(chat_id, f"已收到文件「{file_name}」，已自动添加到知识库。")
        except Exception as e:
            thought_logger.error(f"飞书文件入库失败: {e}")
            await send_feishu_message(chat_id, f"文件接收失败：{e}")
        return {"code": 0, "msg": "file_processed"}

    # 图片消息 → 下载并通过智谱视觉分析
    image_url_for_graph = ""
    if msg_type == "image" and image_key:
        message_id = message.message_id if hasattr(message, "message_id") else message.get("message_id", "")
        try:
            image_url_for_graph = await download_feishu_image(message_id, image_key)
            thought_logger.info(f"飞书图片已转为URL: {image_url_for_graph}")
        except Exception as e:
            thought_logger.error(f"飞书图片下载失败: {e}")
            await send_feishu_message(chat_id, "图片接收失败，请重试。")
            return {"code": -1, "msg": "image_download_failed"}

    try:
        text = json.loads(msg_content).get("text", "")
    except (json.JSONDecodeError, TypeError):
        text = str(msg_content)

    # 纯图片消息（无文本）使用默认提示
    if image_url_for_graph and not text.strip():
        text = "请帮我分析这张图片的内容"

    if not text.strip() and not image_url_for_graph:
        return {"code": 0, "msg": "empty message"}

    # 飞书会话 ID（用于在 Web 端展示）
    feishu_session_id = f"feishu_{user_id}"

    # 需要注册到 session_store 才能在 Web 端看到
    if not session_store.get(feishu_session_id):
        session_store.create(feishu_session_id, f"飞书-{user_id[:10]}")
    session_store.touch(feishu_session_id)

    # 频率限制
    if not check_rate_limit(user_id):
        return {"code": -1, "msg": "请求过于频繁，请稍后再试"}

    # 获取会话锁，同会话串行执行
    async with session_locks[chat_id]:
        task_id = f"task_{int(time.time())}_{user_id}"
        recorder = ThoughtRecorder(user_id, task_id)
        recorder.record_thought("input", text)

        # 获取用户会话（自动处理过期）
        session = get_user_session(user_id)

        # 添加用户消息 → 自动走场景 1/2/3
        await session.add_message("user", text)
        recorder.record_thought(
            "session_update",
            f"历史轮数: {len(session.history)}, 摘要长度: {len(session.long_summary)}",
        )

        # 使用 LangGraph Agent 管线处理（和 Web 端一致）
        config = {"configurable": {"thread_id": feishu_session_id}}
        user_msg = HumanMessage(content=text)

        try:
            result = await asyncio.to_thread(
                graph.invoke,
                {
                    "messages": [user_msg],
                    "query": text,
                    "image_url": image_url_for_graph or "",
                },
                config,
            )
            ai_reply = result.get("final_answer", "")
            recorder.record_thought("agent_route", result.get("route_decision", ""))
            recorder.record_thought("agent_rewrite", result.get("rewritten_query", ""))
            recorder.record_thought("agent_retrieve", f"检索到 {len(result.get('retrieved_docs', []))} 篇文档")
        except Exception as e:
            ai_reply = f"抱歉，处理请求时遇到错误：{str(e)}"
            recorder.record_thought("error", str(e))
            metrics.record_task(False, {"total": 1, "success": 0})
            await send_feishu_message(chat_id, ai_reply)
            return {"code": -1, "msg": ai_reply}

        recorder.record_thought("ai_response", ai_reply)

        # 添加 AI 回复
        await session.add_message("assistant", ai_reply)

        # 记录指标
        context_tokens = (len(text) + len(ai_reply)) // 4
        metrics.record_task(True, {"total": 1, "success": 1}, context_tokens)
        metrics.update_session_stats(USER_SESSIONS)

        # 保存复盘
        recorder.save_session_review()

        # 发送飞书回复
        try:
            await send_feishu_message(chat_id, ai_reply)
        except Exception as e:
            logging.getLogger("agent_thought").error(f"飞书消息发送失败: {e}")

        return {"code": 0, "reply": ai_reply}


# ── 请求模型 ──────────────────────────────────────────────────
class AskRequest(BaseModel):
    question: str
    session_id: str = "default"
    image_url: str = ""


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


# ── Image upload (for vision pipeline) ──────────────────────────
@app.post("/api/upload-image")
async def upload_image(file: UploadFile = File(...)):
    allowed_ext = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}
    ext = os.path.splitext(file.filename or ".jpg")[1].lower()
    if ext not in allowed_ext:
        raise HTTPException(400, f"不支持的图片格式: {ext}，支持 {', '.join(allowed_ext)}")

    safe_name = f"{uuid.uuid4().hex}{ext}"
    file_path = os.path.join(UPLOAD_DIR, safe_name)
    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    public_url = f"{PUBLIC_BASE_URL}/uploads/{safe_name}"
    logging.getLogger("agent_thought").info(f"图片已上传: {safe_name} → {public_url}")
    return {"url": public_url, "filename": safe_name}


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

    # 接入 SessionManager
    web_session = get_user_session(req.session_id)

    try:
        result = graph.invoke(
            {
                "messages": [user_msg],
                "query": req.question.strip(),
                "image_url": req.image_url or "",
            },
            config,
        )

        answer = result.get("final_answer", "抱歉，处理出现错误。")

        session_store.touch(
            req.session_id,
            message_count=len(existing_messages) + 2,
        )
        if is_first:
            session_store.set_title_from_first_message(req.session_id, req.question)

        web_session.add_message_sync("user", req.question.strip())

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
async def ask_stream(
    question: str,
    session_id: str = "default",
    image_url: str = Query(""),
):
    if not knowledge_is_ready():
        raise HTTPException(400, "知识库未就绪，请先上传文档")

    if not question.strip():
        raise HTTPException(400, "问题不能为空")

    if not session_store.get(session_id):
        session_store.create(session_id)

    # 接入 SessionManager：治理 Web 会话（过期检测 + 指标统计）
    web_session = get_user_session(session_id)

    snapshot = graph.get_state({"configurable": {"thread_id": session_id}})
    existing_messages = []
    if snapshot and snapshot.values:
        existing_messages = list(snapshot.values.get("messages", []))
    is_first = len(existing_messages) == 0
    if is_first:
        session_store.set_title_from_first_message(session_id, question)

    async def event_generator():
        try:
            async for sse_chunk in run_agent_stream(
                question.strip(), session_id, image_url=image_url or None
            ):
                yield sse_chunk.encode("utf-8")
        except Exception as e:
            error_data = json.dumps({"error": str(e)}, ensure_ascii=False)
            yield f"event: error\ndata: {error_data}\n\n".encode("utf-8")

        msg_count = len(existing_messages) + 2
        session_store.touch(session_id, message_count=msg_count)
        web_session.add_message_sync("user", question.strip())

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
    cleanup_expired_sessions()
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


# ── 认证 ──────────────────────────────────────────────────────
class AuthRequest(BaseModel):
    username: str
    password: str


@app.post("/api/auth/login")
async def auth_login(req: AuthRequest):
    user = AUTH_USERS.get(req.username)
    if not user or user["password"] != req.password:
        raise HTTPException(401, "用户名或密码错误")

    token = uuid.uuid4().hex
    auth_tokens[token] = {
        "username": req.username,
        "role": user["role"],
        "feishu_user_id": user.get("feishu_user_id", ""),
        "created_at": time.time(),
    }
    return {
        "token": token,
        "username": req.username,
        "role": user["role"],
    }


@app.get("/api/auth/me")
async def auth_me(token: str = Query(...)):
    info = auth_tokens.get(token)
    if not info:
        raise HTTPException(401, "未登录或 token 已过期")
    return info


# ── 飞书回调 ──────────────────────────────────────────────────
@app.get("/feishu/callback")
async def feishu_verify(
    challenge: str = Query(None),
    token: str = Query(None),
):
    # 本地测试：无参数时返回提示
    if not challenge or not token:
        return PlainTextResponse("本地测试：飞书回调接口正常！")

    print(f"[Feishu GET] challenge={challenge[:30]}..., token={token}")
    if FEISHU_VERIFICATION_TOKEN and token != FEISHU_VERIFICATION_TOKEN:
        print("[Feishu GET] Token 验证失败")
        raise HTTPException(403, "Token 验证失败")

    return {"challenge": challenge}


@app.post("/feishu/callback")
async def feishu_callback(request: Request):
    body = await request.body()
    body_str = body.decode()
    print(f"[Feishu POST] body={body_str[:200]}")

    try:
        data = json.loads(body_str)
    except json.JSONDecodeError:
        raise HTTPException(400, "请求体不是有效的 JSON")

    # URL 验证（飞书新版验证用 POST）
    event_type = data.get("type", "")
    if event_type == "url_verification":
        token = data.get("token", "")
        if FEISHU_VERIFICATION_TOKEN and token != FEISHU_VERIFICATION_TOKEN:
            raise HTTPException(403, "Token 验证失败")

        if "encrypt" in data and FEISHU_ENCRYPT_KEY:
            crypt = FeishuMsgCrypt(FEISHU_ENCRYPT_KEY, FEISHU_APP_ID)
            decrypted = crypt.decrypt(data["encrypt"])
            challenge = decrypted.get("challenge", "")
        else:
            challenge = data.get("challenge", "")

        print(f"[Feishu] URL 验证成功")
        return {"challenge": challenge}

    # 事件解密
    event = None
    if "encrypt" in data and FEISHU_ENCRYPT_KEY:
        crypt = FeishuMsgCrypt(FEISHU_ENCRYPT_KEY, FEISHU_APP_ID)
        event = crypt.decrypt(data["encrypt"])
    else:
        event = data

    # 异步处理消息，不阻塞飞书 3 秒超时
    asyncio.create_task(process_message(event))

    return {"code": 0}


# ── 指标查询接口 ──────────────────────────────────────────────
@app.get("/api/metrics")
async def get_metrics():
    metrics.update_session_stats(USER_SESSIONS)
    metrics.save_metrics_to_file(force=True)
    return {
        "uptime_seconds": round(time.time() - metrics.start_time, 2),
        "total_tasks": metrics.total_tasks,
        "success_tasks": metrics.success_tasks,
        "tool_calls": metrics.tool_calls,
        "tool_success": metrics.tool_success,
        "session_stats": metrics.session_stats,
    }


# ── 健康检查 ──────────────────────────────────────────────────
@app.get("/api/health")
async def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


# ── 长连接模式（开发环境自动启动） ────────────────────────────
def start_long_connection():
    """飞书 WebSocket 长连接，无需内网穿透"""
    try:
        import lark_oapi as lark

        def on_message(event_context):
            """同步回调，在 WebSocket 事件循环中调度异步处理"""
            try:
                loop = asyncio.get_event_loop()
                loop.create_task(process_message(event_context))
            except Exception as e:
                logging.getLogger("agent_thought").error(f"长连接调度失败: {e}")

        event_handler = (
            lark.EventDispatcherHandler.builder(
                FEISHU_ENCRYPT_KEY, FEISHU_VERIFICATION_TOKEN
            )
            .register_p2_im_message_receive_v1(on_message)
            .build()
        )

        cli = lark.ws.Client(
            FEISHU_APP_ID,
            FEISHU_APP_SECRET,
            event_handler=event_handler,
            log_level=lark.LogLevel.DEBUG if DEBUG_MODE else lark.LogLevel.INFO,
        )
        print("[长连接] 飞书 WebSocket 已启动")
        cli.start()
    except ImportError:
        print("[长连接] lark-oapi 未安装，跳过 WebSocket 模式")
    except Exception as e:
        print(f"[长连接] 启动失败: {e}")


# ── Startup ───────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    print(f"WorkMate Agent v2.0.0 启动中...")
    print(f"LLM: DeepSeek (deepseek-chat)")
    print(f"知识库: {'已就绪' if knowledge_is_ready() else '未建库'}")
    print(f"访问: http://{SERVER_HOST}:{SERVER_PORT}")

    # 上线前检查
    run_pre_launch_checks()

    # 开发环境启动长连接
    if DEBUG_MODE and FEISHU_APP_ID and FEISHU_APP_SECRET:
        threading.Thread(target=start_long_connection, daemon=True).start()
        print("[启动] 开发模式，已启动飞书长连接线程")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT)
