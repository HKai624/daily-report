import json
import logging
from typing import Generator
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from src.config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, MODEL_NAME
from src.tools.weather import get_weather
from src.tools.todo import create_todo, list_todos
from src.tools.email_sender import send_email
from src.tools.notify import send_notification
from src.retry_utils import execute_tool_with_retry_sync
from src.tool_governance import execute_tool_with_audit

thought_logger = logging.getLogger("agent_thought")

llm = ChatOpenAI(
    model=MODEL_NAME,
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
    temperature=0.3,
)

stream_llm = ChatOpenAI(
    model=MODEL_NAME,
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
    temperature=0.3,
    streaming=True,
)

TOOLS = [
    {
        "name": "get_weather",
        "description": "查询指定城市的实时天气信息",
        "parameters": {"city": "城市名称，如 北京、南宁"},
    },
    {
        "name": "create_todo",
        "description": "创建一个新的待办事项",
        "parameters": {"title": "待办标题", "due_date": "截止日期(可选)，格式 YYYY-MM-DD"},
    },
    {
        "name": "list_todos",
        "description": "列出所有待办事项",
        "parameters": {},
    },
    {
        "name": "send_email",
        "description": "发送邮件",
        "parameters": {"to": "收件人邮箱", "subject": "邮件主题", "body": "邮件正文"},
    },
    {
        "name": "send_notification",
        "description": "发送企业IM通知",
        "parameters": {"recipient": "接收人", "message": "通知内容"},
    },
]

TOOL_FUNCTIONS = {
    "get_weather": lambda args: get_weather(args.get("city", "")),
    "create_todo": lambda args: create_todo(args.get("title", ""), args.get("due_date", "")),
    "list_todos": lambda args: list_todos(),
    "send_email": lambda args: send_email(args.get("to", ""), args.get("subject", ""), args.get("body", "")),
    "send_notification": lambda args: send_notification(args.get("recipient", ""), args.get("message", "")),
}


def _has_image_context(query: str) -> bool:
    """检测 query 是否包含来自 vision_node 的图片分析结果"""
    return query.startswith("[系统提示] 用户发送了一张图片")


def solve(state: dict) -> dict:
    route = state.get("route_decision", "chat")
    query = state.get("rewritten_query", "") or state.get("query", "")
    context = state.get("retrieved_context", "")
    messages = state.get("messages", [])

    tool_results = []
    tool_calls_made = []

    has_image = _has_image_context(query)
    system_prompt = _build_system_prompt(route, context, has_image=has_image)

    if route == "tool":
        result = _handle_tool_route(query, system_prompt)
        tool_calls_made = result.get("tool_calls", [])
        tool_results = result.get("tool_results", [])

        if tool_results:
            tool_context = "\n".join(
                [f"[工具: {t['tool']}] {t['result']}" for t in tool_results]
            )
            system_prompt += f"\n\n工具调用结果：\n{tool_context}"

    history = []
    for msg in messages[-10:]:
        role = "用户" if getattr(msg, "type", "human") == "human" else "助手"
        content = msg.content if hasattr(msg, "content") else str(msg)
        # 历史消息中的图片分析结果太长时截断
        if _has_image_context(content) and len(content) > 300:
            content = content[:300] + "...[图片描述已截断]"
        history.append(f"{role}: {content}")

    history_text = "\n".join(history) if history else "(无历史)"

    final_prompt = f"""{system_prompt}

对话历史：
{history_text}

用户当前问题：{query}

请用中文生成专业、有帮助的回答。如需结构化信息，使用 Markdown 格式。"""

    response = llm.invoke([HumanMessage(content=final_prompt)])
    answer = response.content.strip() if hasattr(response, "content") else str(response).strip()

    return {
        "final_answer": answer,
        "tool_calls": tool_calls_made,
        "tool_results": tool_results,
    }


def _build_system_prompt(route: str, context: str, has_image: bool = False) -> str:
    base = "你是 WorkMate Agent，一个专业的智能职场行政助理。"
    base += "你由 DeepSeek 4.0 Pro 驱动，擅长处理企业行政事务、信息查询、任务管理等。"

    if has_image:
        base += "\n\n用户通过图片与你沟通。请仔细阅读图片分析结果，理解图片中传达的信息，并结合用户的文字说明给出回复。如果图片中包含文档、表格、截图等结构化信息，请用 Markdown 格式整理输出。如果图片中包含待办事项、会议安排、任务指令等，请主动帮用户记录或执行。"

    if route == "rag" and context:
        base += f"\n\n以下是从企业内部知识库检索到的相关文档：\n{context}\n\n请基于以上文档回答问题。如果文档不足以回答，请如实说明。"

    if route == "tool":
        base += "\n\n你可以使用工具来完成任务。请根据工具返回的结果给出自然的回答。"

    if route == "chat":
        base += "\n\n这是通用对话，请友好、专业地回复用户。"

    return base


def _execute_single_tool(name: str, args: dict) -> str:
    """执行单个工具调用，带重试、超时、审计日志"""
    tool_func = TOOL_FUNCTIONS.get(name)
    if not tool_func:
        return f"未知工具: {name}"

    wrapped = lambda: tool_func(args)
    result = execute_tool_with_retry_sync(wrapped, name, args)
    execute_tool_with_audit(name, args, user_id="agent", tool_executor=wrapped)
    return str(result)


def _handle_tool_route(query: str, system_prompt: str) -> dict:
    tools_json = json.dumps(TOOLS, ensure_ascii=False, indent=2)

    tool_selection_prompt = f"""你有一个工具库。分析用户需求，决定需要调用哪些工具。

可用的工具：
{tools_json}

用户查询：{query}

请以JSON格式回复：
{{"tools": [{{"name": "工具名", "arguments": {{参数}}}}] }}

如果用户查询不需要工具，返回：{{"tools": []}}
只输出JSON，不要其他内容。"""

    def _parse_tool_plan(text: str) -> list:
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        plan = json.loads(text.strip())
        return plan.get("tools", [])

    def _execute_tool_batch(selected: list) -> tuple:
        tool_calls = []
        tool_results = []
        for item in selected[:3]:
            name = item.get("name", "")
            args = item.get("arguments", {})
            if name in TOOL_FUNCTIONS:
                try:
                    result = _execute_single_tool(name, args)
                    tool_calls.append({"tool": name, "arguments": args})
                    tool_results.append({"tool": name, "result": result})
                except Exception as e:
                    tool_calls.append({"tool": name, "arguments": args})
                    tool_results.append({"tool": name, "result": f"错误: {str(e)}"})
                    thought_logger.error(f"工具 {name} 最终失败: {e}")
        return tool_calls, tool_results

    try:
        response = llm.invoke([HumanMessage(content=tool_selection_prompt)])
        text = response.content.strip() if hasattr(response, "content") else str(response).strip()

        try:
            selected = _parse_tool_plan(text)
        except (json.JSONDecodeError, IndexError):
            thought_logger.warning("工具计划 JSON 解析失败，跳过工具调用")
            return {"tool_calls": [], "tool_results": []}

        tool_calls, tool_results = _execute_tool_batch(selected)

        # Reflection: 如果工具全部失败，让 LLM 修正参数后重试一次
        if tool_results and all("错误" in str(t.get("result", "")) for t in tool_results):
            reflection_prompt = f"""上次工具调用全部失败，请修正参数后重试。

失败信息：
{json.dumps([{"tool": t["tool"], "result": t["result"]} for t in tool_results], ensure_ascii=False)}

原始查询：{query}

请重新输出JSON工具计划（仅输出JSON）："""

            try:
                retry_response = llm.invoke([HumanMessage(content=reflection_prompt)])
                retry_text = retry_response.content.strip() if hasattr(retry_response, "content") else str(retry_response).strip()
                retry_selected = _parse_tool_plan(retry_text)
                retry_calls, retry_results = _execute_tool_batch(retry_selected)
                if retry_results:
                    thought_logger.info("Reflection 重试完成")
                    return {"tool_calls": tool_calls + retry_calls, "tool_results": retry_results}
            except Exception as e:
                thought_logger.warning(f"Reflection 重试失败: {e}")

        return {"tool_calls": tool_calls, "tool_results": tool_results}

    except Exception as e:
        thought_logger.error(f"工具路由异常: {e}")
        return {"tool_calls": [], "tool_results": []}


def solve_stream(state: dict) -> Generator[str, None, None]:
    route = state.get("route_decision", "chat")
    query = state.get("rewritten_query", "") or state.get("query", "")
    context = state.get("retrieved_context", "")
    messages = state.get("messages", [])

    tool_results = []
    tool_calls_made = []

    has_image = _has_image_context(query)
    system_prompt = _build_system_prompt(route, context, has_image=has_image)

    if route == "tool":
        result = _handle_tool_route(query, system_prompt)
        tool_calls_made = result.get("tool_calls", [])
        tool_results = result.get("tool_results", [])

        if tool_results:
            tool_context = "\n".join(
                [f"[工具: {t['tool']}] {t['result']}" for t in tool_results]
            )
            system_prompt += f"\n\n工具调用结果：\n{tool_context}"

    history = []
    for msg in messages[-10:]:
        role = "用户" if getattr(msg, "type", "human") == "human" else "助手"
        content = msg.content if hasattr(msg, "content") else str(msg)
        if _has_image_context(content) and len(content) > 300:
            content = content[:300] + "...[图片描述已截断]"
        history.append(f"{role}: {content}")

    history_text = "\n".join(history) if history else "(无历史)"

    final_prompt = f"""{system_prompt}

对话历史：
{history_text}

用户当前问题：{query}

请用中文生成专业、有帮助的回答。如需结构化信息，使用 Markdown 格式。"""

    for chunk in stream_llm.stream([HumanMessage(content=final_prompt)]):
        if hasattr(chunk, "content") and chunk.content:
            yield chunk.content
