import json
from typing import Generator
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from src.config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, MODEL_NAME
from src.tools.weather import get_weather
from src.tools.todo import create_todo, list_todos
from src.tools.email_sender import send_email
from src.tools.notify import send_notification

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


def solve(state: dict) -> dict:
    route = state.get("route_decision", "chat")
    query = state.get("rewritten_query", "") or state.get("query", "")
    context = state.get("retrieved_context", "")
    messages = state.get("messages", [])

    tool_results = []
    tool_calls_made = []

    system_prompt = _build_system_prompt(route, context)

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


def _build_system_prompt(route: str, context: str) -> str:
    base = "你是 WorkMate Agent，一个专业的智能职场行政助理。"
    base += "你由 DeepSeek 4.0 Pro 驱动，擅长处理企业行政事务、信息查询、任务管理等。"

    if route == "rag" and context:
        base += f"\n\n以下是从企业内部知识库检索到的相关文档：\n{context}\n\n请基于以上文档回答问题。如果文档不足以回答，请如实说明。"

    if route == "tool":
        base += "\n\n你可以使用工具来完成任务。请根据工具返回的结果给出自然的回答。"

    if route == "chat":
        base += "\n\n这是通用对话，请友好、专业地回复用户。"

    return base


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

    try:
        response = llm.invoke([HumanMessage(content=tool_selection_prompt)])
        text = response.content.strip() if hasattr(response, "content") else str(response).strip()

        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        text = text.strip()

        plan = json.loads(text)
        selected = plan.get("tools", [])

        tool_calls = []
        tool_results = []

        for item in selected[:3]:
            name = item.get("name", "")
            args = item.get("arguments", {})
            if name in TOOL_FUNCTIONS:
                try:
                    result = TOOL_FUNCTIONS[name](args)
                    tool_calls.append({"tool": name, "arguments": args})
                    tool_results.append({"tool": name, "result": str(result)})
                except Exception as e:
                    tool_results.append({"tool": name, "result": f"错误: {str(e)}"})

        return {"tool_calls": tool_calls, "tool_results": tool_results}

    except json.JSONDecodeError:
        return {"tool_calls": [], "tool_results": []}


def solve_stream(state: dict) -> Generator[str, None, None]:
    route = state.get("route_decision", "chat")
    query = state.get("rewritten_query", "") or state.get("query", "")
    context = state.get("retrieved_context", "")
    messages = state.get("messages", [])

    tool_results = []
    tool_calls_made = []

    system_prompt = _build_system_prompt(route, context)

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
