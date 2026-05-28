import time
import logging
from typing import Callable, Awaitable

from src.config import SESSION_MAX_LENGTH, SESSION_EXPIRE_SECONDS

session_logger = logging.getLogger("session_state")


class SessionManager:
    """多轮对话记忆管理：滑动窗口 + 摘要 + 过期清空"""

    def __init__(self, user_id: str):
        self.user_id = user_id
        self.history: list[dict] = []
        self.long_summary: str = ""
        self.last_active_time: float = time.time()

    def refresh_active_time(self):
        self.last_active_time = time.time()

    def is_expired(self) -> bool:
        expired = time.time() - self.last_active_time > SESSION_EXPIRE_SECONDS
        if expired:
            session_logger.info(f"会话 {self.user_id} 已过期，即将清空")
        return expired

    def clear_all(self):
        self.history = []
        self.long_summary = ""
        session_logger.info(f"会话 {self.user_id} 已清空所有数据")

    async def summarize_history(self, llm_call: Callable[..., Awaitable[str]]):
        keep_count = SESSION_MAX_LENGTH // 2
        keep_recent = self.history[-keep_count:]
        old_messages = self.history[:-keep_count]

        prompt = (
            "请总结以下办公对话，只保留核心信息（人物、决策、待办事项）：\n"
            f"{old_messages}\n总结："
        )
        try:
            summary = await llm_call(prompt)
        except Exception as e:
            session_logger.error(f"摘要生成失败: {e}")
            summary = "（摘要生成失败）"

        if self.long_summary:
            merge_prompt = (
                f"合并两个摘要，去重并保留关键信息：\n"
                f"摘要1: {self.long_summary}\n摘要2: {summary}"
            )
            try:
                self.long_summary = await llm_call(merge_prompt)
            except Exception:
                self.long_summary = f"{self.long_summary}\n{summary}"
        else:
            self.long_summary = summary

        self.history = keep_recent
        session_logger.info(
            f"会话 {self.user_id} 已压缩历史对话，"
            f"保留 {len(self.history)} 轮，摘要长度: {len(self.long_summary)}"
        )

    async def add_message(
        self,
        role: str,
        content: str,
        llm_call: Callable[..., Awaitable[str]] = None,
    ):
        # 场景 3：过期清空
        if self.is_expired():
            self.clear_all()

        # 刷新活跃时间（重置 7 天倒计时）
        self.refresh_active_time()

        # 添加新消息
        self.history.append({"role": role, "content": content})

        # 场景 2：超过 SESSION_MAX_LENGTH 轮 → 压缩摘要
        if len(self.history) > SESSION_MAX_LENGTH and llm_call:
            await self.summarize_history(llm_call)

        session_logger.info(
            f"会话 {self.user_id} 新增消息: role={role}, "
            f"当前轮数={len(self.history)}, 摘要长度={len(self.long_summary)}"
        )

    def add_message_sync(self, role: str, content: str):
        """同步版消息追踪：只做过期检查和活跃时间刷新，不做摘要压缩"""
        if self.is_expired():
            self.clear_all()
        self.refresh_active_time()
        self.history.append({"role": role, "content": content})
        session_logger.info(
            f"会话 {self.user_id} 新增消息: role={role}, "
            f"当前轮数={len(self.history)}"
        )

    def get_context(self) -> list[dict]:
        context = []
        if self.long_summary:
            context.append({
                "role": "system",
                "content": f"历史摘要：{self.long_summary}",
            })
        context.extend(self.history)
        return context


# 全局用户会话管理
USER_SESSIONS: dict[str, SessionManager] = {}


def get_user_session(user_id: str) -> SessionManager:
    if user_id not in USER_SESSIONS:
        USER_SESSIONS[user_id] = SessionManager(user_id)
        session_logger.info(f"创建新会话: {user_id}")
    return USER_SESSIONS[user_id]


def cleanup_expired_sessions() -> int:
    """清理所有过期会话，返回清理数量"""
    expired_ids = [
        uid for uid, sess in USER_SESSIONS.items() if sess.is_expired()
    ]
    for uid in expired_ids:
        del USER_SESSIONS[uid]
        session_logger.info(f"已清理过期会话: {uid}")
    return len(expired_ids)
