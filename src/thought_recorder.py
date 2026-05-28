import json
import os
import logging
from datetime import datetime

from src.config import SESSION_MAX_LENGTH, SESSION_EXPIRE_SECONDS, DATA_DIR

thought_logger = logging.getLogger("agent_thought")


class ThoughtRecorder:
    """记录 Agent 思考链路：思考→拆步骤→调用工具→结果验证"""

    def __init__(self, user_id: str, task_id: str):
        self.user_id = user_id
        self.task_id = task_id
        self.start_time = datetime.now()
        self.thoughts: list[dict] = []

    def record_thought(self, step: str, content: str):
        thought = {
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "content": content,
        }
        self.thoughts.append(thought)
        thought_logger.info(
            f"[{self.user_id}][{self.task_id}][{step}] {content[:200]}"
        )

    def save_session_review(self) -> dict:
        reviews_dir = os.path.join(DATA_DIR, "reviews")
        os.makedirs(reviews_dir, exist_ok=True)

        tool_calls_count = sum(
            1 for t in self.thoughts if t["step"] in ("tool_call", "tool_execute")
        )

        issues = []
        learnings = []

        for thought in self.thoughts:
            content = thought.get("content", "")
            if thought["step"] in ("tool_result", "tool_execute") and "error" in content.lower():
                issues.append(content)
            if thought["step"] == "reflection" and content:
                learnings.append(content)

        review = {
            "user_id": self.user_id,
            "task_id": self.task_id,
            "start_time": self.start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "duration_seconds": (datetime.now() - self.start_time).total_seconds(),
            "thoughts_count": len(self.thoughts),
            "tool_calls_count": tool_calls_count,
            "issues": issues,
            "learnings": learnings,
            "config_snapshot": {
                "SESSION_MAX_LENGTH": SESSION_MAX_LENGTH,
                "SESSION_EXPIRE_SECONDS": SESSION_EXPIRE_SECONDS,
            },
        }

        filepath = os.path.join(reviews_dir, f"{self.task_id}.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(review, f, ensure_ascii=False, indent=2)

        thought_logger.info(
            f"[{self.user_id}][{self.task_id}] 复盘已保存: "
            f"issues={len(issues)}, learnings={len(learnings)}"
        )
        return review
