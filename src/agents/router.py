from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from src.config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, MODEL_NAME

llm = ChatOpenAI(
    model=MODEL_NAME,
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
    temperature=0.0,
)


def route_query(state: dict) -> dict:
    query = state.get("rewritten_query", "") or state.get("query", "")
    knowledge_ready = state.get("knowledge_ready", False)

    available_routes = ["chat"]
    route_descriptions = ["直接回答：通用对话、问候、闲聊、逻辑推理等"]

    if knowledge_ready:
        available_routes.append("rag")
        route_descriptions.append("rag：查询公司内部制度、流程、FAQ、文档等")

    available_routes.append("tool")
    route_descriptions.append("tool：查天气、创建待办、发邮件、发通知等需要调用外部工具")

    routes_text = "\n".join([f"- {d}" for d in route_descriptions])
    routes_list = ", ".join(available_routes)

    prompt = f"""你是一个智能路由器。分析用户查询，将其分类到以下类别之一。

可用类别：
{routes_text}

分类规则：
- 如果用户询问公司内部制度、流程、政策、FAQ、考勤、请假、年假、报销、福利、会议室、IT设备、打印机、VPN、Wi-Fi、培训、晋升等企业内部信息，优先选择 rag
- 如果用户需要查天气、创建待办、发送邮件、发送通知等具体操作，选择 tool
- 如果用户是打招呼、闲聊、一般性问题，选择 chat

用户查询：{query}

请只输出一个类别名称（{routes_list}），不要输出其他内容。"""

    response = llm.invoke([HumanMessage(content=prompt)])
    decision = response.content.strip().lower() if hasattr(response, "content") else str(response).strip().lower()

    if decision not in ["rag", "tool", "chat"]:
        if "rag" in decision or "文档" in decision or "知识" in decision or "制度" in decision:
            decision = "rag"
        elif "tool" in decision or "工具" in decision or "天气" in decision or "待办" in decision or "邮件" in decision:
            decision = "tool"
        else:
            decision = "chat"

    return {"route_decision": decision}
