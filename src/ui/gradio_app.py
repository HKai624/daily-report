import json
import os
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import gradio as gr
from langchain_core.messages import HumanMessage

from src.graph.workflow import compile_graph
from src.rag.index_builder import build_index
from src.rag.retriever_engine import knowledge_is_ready

graph = compile_graph()

# 极简 CSS — 仅对 Gradio 暴露的 CSS 变量做定制
CUSTOM_CSS = """
.gradio-container {
    max-width: 1000px !important;
    margin: 0 auto !important;
}

/* 隐藏 footer */
footer { display: none !important; }

/* 输入框聚焦 */
textarea:focus {
    border-color: #4E6EF2 !important;
    box-shadow: 0 0 0 3px rgba(78,110,242,0.1) !important;
}

/* 状态栏 */
.status-bar {
    display: flex;
    align-items: center;
    gap: 16px;
    font-size: 13px;
    color: #666;
    padding: 8px 0;
}
.status-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 4px;
}
.status-dot.green { background: #00B42A; }
.status-dot.orange { background: #FF7D00; }

/* 思考过程面板 */
.step-panel {
    font-family: "Consolas", "Courier New", monospace;
    font-size: 12px;
    line-height: 1.7;
    color: #555;
    background: #FAFBFC;
    border-radius: 6px;
    padding: 10px 14px;
    border: 1px solid #EEE;
    max-height: 320px;
    overflow-y: auto;
}
.step-panel .label { font-weight: 600; color: #333; }
.step-panel .rewrite { color: #4E5969; }
.step-panel .retrieve { color: #4E6EF2; }
.step-panel .tool { color: #FF7D00; }
.step-panel .muted { color: #C9CDD4; }
"""


def _build_steps(result: dict) -> str:
    parts = []

    rewritten = result.get("rewritten_query", "")
    if rewritten and rewritten != result.get("query", ""):
        parts.append(
            f'<div class="rewrite">查询重写: {rewritten}</div>'
        )

    docs = result.get("retrieved_docs", [])
    if docs:
        parts.append(
            f'<div class="retrieve">检索到 {len(docs)} 篇相关文档:</div>'
        )
        for i, doc in enumerate(docs):
            src = doc.get("source", "?")
            preview = doc.get("content", "")[:100]
            parts.append(
                f'<div style="padding-left:10px;border-left:2px solid #E5E6EB;margin:3px 0;color:#86909C;font-size:11px;">'
                f'[{i+1}] {src}<br>{preview}...'
                f'</div>'
            )

    tool_calls = result.get("tool_calls", [])
    if tool_calls:
        parts.append(
            f'<div class="tool">调用 {len(tool_calls)} 个工具:</div>'
        )
        for tc in tool_calls:
            args = json.dumps(tc.get("arguments", {}), ensure_ascii=False)
            parts.append(
                f'<div style="color:#86909C;font-size:11px;">'
                f'&rarr; {tc.get("tool", "?")}({args})'
                f'</div>'
            )

    if not parts:
        parts.append('<div class="muted">等待输入...</div>')

    return "\n".join(parts)


def chat_fn(message, history, session_id):
    """处理用户消息，返回 (空字符串, 新history, 思考步骤, 路由信息, 工具结果)"""
    if not message or not message.strip():
        return "", history, _build_steps({}), "{}", "{}"

    if not session_id:
        session_id = uuid.uuid4().hex

    config = {"configurable": {"thread_id": session_id}}

    # 从 graph 状态恢复历史消息
    snapshot = graph.get_state(config)
    existing_messages = []
    if snapshot and snapshot.values:
        existing_messages = list(snapshot.values.get("messages", []))

    user_msg = HumanMessage(content=message.strip())
    all_messages = existing_messages + [user_msg]

    try:
        result = graph.invoke(
            {"messages": all_messages, "query": message.strip()},
            config,
        )
        answer = result.get("final_answer", "抱歉，处理过程出现错误。")

        # Gradio 6.0 Chatbot 使用 {"role": "...", "content": "..."} 格式
        history = history or []
        history.append({"role": "user", "content": message.strip()})
        history.append({"role": "assistant", "content": answer})

        steps = _build_steps(result)

        route_display = {
            "rag": "知识检索", "tool": "工具调用", "chat": "直接对话"
        }.get(result.get("route_decision", ""), result.get("route_decision", "N/A"))

        route_info = json.dumps({
            "路由决策": route_display,
            "重写查询": result.get("rewritten_query", ""),
            "检索文档": len(result.get("retrieved_docs", [])),
            "调用工具": [t.get("tool", "") for t in result.get("tool_calls", [])],
        }, ensure_ascii=False, indent=2)

        tools_info = json.dumps(
            result.get("tool_results", []), ensure_ascii=False, indent=2
        )

        return "", history, steps, route_info, tools_info

    except Exception as e:
        error_msg = f"系统错误: {str(e)}"
        history = history or []
        history.append({"role": "user", "content": message.strip()})
        history.append({"role": "assistant", "content": error_msg})
        return "", history, _build_steps({}), "{}", "{}"


def clear_fn(session_id):
    """清空对话"""
    return uuid.uuid4().hex, [], _build_steps({}), "{}", "{}"


def build_ui():
    theme = gr.themes.Soft(
        primary_hue="blue",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
    )
    # 覆盖为更亮的白色
    theme.set(
        body_background_fill="*neutral_50",
        block_background_fill="white",
        block_border_width="1px",
        block_border_color="*neutral_200",
        block_radius="md",
        input_background_fill="*neutral_50",
    )

    with gr.Blocks(title="WorkMate Agent 智能行政助理") as app:
        # === Header ===
        kn_ready = knowledge_is_ready()
        gr.HTML(f"""
        <div style="display:flex;align-items:center;justify-content:space-between;padding:18px 24px 14px;">
            <div style="display:flex;align-items:center;gap:10px;">
                <div style="width:32px;height:32px;border-radius:6px;background:linear-gradient(135deg,#4E6EF2,#7B61FF);display:flex;align-items:center;justify-content:center;font-size:16px;color:#fff;font-weight:700;">W</div>
                <div>
                    <div style="font-size:16px;font-weight:600;color:#1D2129;">WorkMate Agent 智能行政助理</div>
                    <div style="font-size:11px;color:#86909C;">多智能体协作 &middot; DeepSeek 4.0 Pro &middot; LangGraph</div>
                </div>
            </div>
            <div class="status-bar">
                <span><span class="status-dot green"></span>DeepSeek 4.0 Pro 已连接</span>
                <span style="color:#E5E6EB;">|</span>
                <span><span class="status-dot {'green' if kn_ready else 'orange'}"></span>{'知识库已就绪' if kn_ready else '知识库未建库'}</span>
            </div>
        </div>
        """)

        session_state = gr.State(uuid.uuid4().hex)

        with gr.Row():
            # === Left Sidebar ===
            with gr.Column(scale=1, min_width=240):
                gr.Markdown("""
                **智能体流水线**
                ```
                01 查询重写器  →  优化用户问题
                02 路由器      →  意图分类
                03 检索器      →  ChromaDB 向量检索
                04 解答器      →  生成最终回答
                ```
                """)

                gr.Markdown("""
                **可用工具**
                - 天气查询 (wttr.in)
                - 待办管理
                - 邮件发送
                - 消息通知
                """)

                with gr.Accordion("智能体追踪", open=True):
                    steps_display = gr.HTML(
                        value='<div class="step-panel"><div class="muted">等待输入...</div></div>',
                        show_label=False,
                    )

                with gr.Accordion("路由信息", open=False):
                    route_info = gr.Code(
                        language="json", value="{}",
                        label=None, show_label=False,
                    )

                with gr.Accordion("工具调用结果", open=False):
                    tools_info = gr.Code(
                        language="json", value="{}",
                        label=None, show_label=False,
                    )

            # === Right Chat Area ===
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label=None,
                    show_label=False,
                    placeholder="等待您的第一个问题...",
                    autoscroll=True,
                )

                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="输入您的问题，Enter 发送...",
                        show_label=False,
                        scale=8,
                        container=True,
                    )
                    send_btn = gr.Button("发送", variant="primary", scale=1, min_width=60)
                    clear_btn = gr.Button("清空", variant="secondary", scale=0, min_width=60)

        # === Event Handlers ===
        msg_input.submit(
            fn=chat_fn,
            inputs=[msg_input, chatbot, session_state],
            outputs=[msg_input, chatbot, steps_display, route_info, tools_info],
        )

        send_btn.click(
            fn=chat_fn,
            inputs=[msg_input, chatbot, session_state],
            outputs=[msg_input, chatbot, steps_display, route_info, tools_info],
        )

        clear_btn.click(
            fn=clear_fn,
            inputs=[session_state],
            outputs=[session_state, chatbot, steps_display, route_info, tools_info],
        )

    return app, theme


if __name__ == "__main__":
    app, theme = build_ui()
    app.launch(
        server_name="127.0.0.1",
        server_port=8000,
        share=False,
        theme=theme,
        css=CUSTOM_CSS,
    )
