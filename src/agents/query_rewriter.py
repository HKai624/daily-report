from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from src.config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, MODEL_NAME

llm = ChatOpenAI(
    model=MODEL_NAME,
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
    temperature=0.1,
)


def rewrite_query(state: dict, current_query: str = "") -> dict:
    messages = state.get("messages", [])
    if not messages:
        return {"rewritten_query": current_query or state.get("query", "")}

    last_msg = messages[-1]
    user_query = current_query or (last_msg.content if hasattr(last_msg, "content") else str(last_msg))

    # 如果 query 包含图片分析结果（以 [系统提示] 开头），不对图片描述部分做重写
    has_image_context = user_query.startswith("[系统提示] 用户发送了一张图片")

    if len(messages) <= 1:
        return {"rewritten_query": user_query}

    history = []
    for msg in messages[:-1]:
        role = "用户" if getattr(msg, "type", "human") == "human" else "助手"
        content = msg.content if hasattr(msg, "content") else str(msg)
        history.append(f"{role}: {content}")

    context = "\n".join(history[-6:])

    if has_image_context:
        # 图片场景：从图片描述后的用户说明中提取核心意图，但保留完整的图片上下文
        marker = "用户对图片的说明：'"
        after_marker = ""
        if marker in user_query:
            parts = user_query.split(marker, 1)
            before = parts[0]  # 图片分析结果
            after = parts[1].rstrip("'。") if len(parts) > 1 else ""
            after_marker = after
        core_query = after_marker or user_query

        prompt = f"""你是一个查询重写助手。用户通过图片传达了信息，请根据对话历史和图片分析结果，重写一个完整的、清晰的、独立的查询。

对话历史：
{context}

用户通过图片传达了以下信息（不需要重写这部分）：
{user_query[:500]}...

请提取用户的核心意图，重写为一句清晰的查询。只输出重写后的查询，不要添加任何解释。"""

        response = llm.invoke([HumanMessage(content=prompt)])
        rewritten = response.content.strip() if hasattr(response, "content") else str(response).strip()
        # 将重写后的意图拼接到图片上下文之后，确保 solver 同时拥有两者
        combined = f"{user_query}\n\n[重写后的意图] {rewritten}"
        return {"rewritten_query": combined}
    else:
        prompt = f"""你是一个查询重写助手。根据对话历史和用户的最新问题，重写一个完整的、清晰的、独立的查询。
只输出重写后的查询，不要添加任何解释。

对话历史：
{context}

用户最新问题：{user_query}

重写后的查询："""

        response = llm.invoke([HumanMessage(content=prompt)])
        rewritten = response.content.strip() if hasattr(response, "content") else str(response).strip()
        return {"rewritten_query": rewritten}
