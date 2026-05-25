from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from src.config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, MODEL_NAME

llm = ChatOpenAI(
    model=MODEL_NAME,
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
    temperature=0.1,
)


def rewrite_query(state: dict) -> dict:
    messages = state.get("messages", [])
    if not messages:
        return {"rewritten_query": state.get("query", "")}

    last_msg = messages[-1]
    user_query = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

    if len(messages) <= 1:
        return {"rewritten_query": user_query}

    history = []
    for msg in messages[:-1]:
        role = "用户" if getattr(msg, "type", "human") == "human" else "助手"
        content = msg.content if hasattr(msg, "content") else str(msg)
        history.append(f"{role}: {content}")

    context = "\n".join(history[-6:])
    prompt = f"""你是一个查询重写助手。根据对话历史和用户的最新问题，重写一个完整的、清晰的、独立的查询。
只输出重写后的查询，不要添加任何解释。

对话历史：
{context}

用户最新问题：{user_query}

重写后的查询："""

    response = llm.invoke([HumanMessage(content=prompt)])
    rewritten = response.content.strip() if hasattr(response, "content") else str(response).strip()
    return {"rewritten_query": rewritten}
