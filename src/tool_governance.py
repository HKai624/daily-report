import time
import logging
from datetime import datetime
from enum import Enum
from typing import Any

audit_logger = logging.getLogger("tool_audit")


class ToolPermission(Enum):
    LOW = "low"        # 查询、读取
    MEDIUM = "medium"  # 创建、更新
    HIGH = "high"      # 删除、外发、系统修改


# 工具权限配置（默认 deny：未知工具归为 HIGH）
TOOL_PERMISSIONS: dict[str, ToolPermission] = {
    "search_documents": ToolPermission.LOW,
    "read_calendar": ToolPermission.LOW,
    "get_weather": ToolPermission.LOW,
    "get_news": ToolPermission.LOW,
    "create_task": ToolPermission.MEDIUM,
    "create_todo": ToolPermission.MEDIUM,
    "send_email": ToolPermission.HIGH,
    "delete_record": ToolPermission.HIGH,
}

# 风险动作审批队列
approval_queue: dict[str, dict] = {}


async def execute_tool_with_permission_check(
    tool_name: str,
    tool_args: dict,
    user_id: str,
    tool_executor,
    auto_approve_low_risk: bool = True,
) -> Any:
    """带权限检查和审计日志的工具执行包装（异步版）"""
    permission = TOOL_PERMISSIONS.get(tool_name, ToolPermission.HIGH)

    # 审计日志 — 尝试
    audit_info = {
        "timestamp": datetime.now().isoformat(),
        "user_id": user_id,
        "tool_name": tool_name,
        "tool_args": str(tool_args),
        "permission_level": permission.value,
        "action": "attempt",
    }
    audit_logger.info(f"工具调用审计: {audit_info}")

    # 高风险动作需要审批
    if permission == ToolPermission.HIGH and not auto_approve_low_risk:
        approval_id = f"approval_{int(time.time())}_{user_id}"
        approval_queue[approval_id] = {
            "tool_name": tool_name,
            "tool_args": tool_args,
            "user_id": user_id,
            "status": "pending",
            "created_at": datetime.now().isoformat(),
        }

        audit_info["action"] = "waiting_approval"
        audit_info["approval_id"] = approval_id
        audit_logger.info(f"工具调用审计: {audit_info}")
        return f"该操作需要审批，审批ID: {approval_id}"

    # 执行工具
    try:
        result = await tool_executor(tool_name, tool_args)
        audit_info["action"] = "success"
        audit_info["result_preview"] = str(result)[:200]
        audit_logger.info(f"工具调用审计: {audit_info}")
        return result
    except Exception as e:
        audit_info["action"] = "failed"
        audit_info["error"] = str(e)
        audit_logger.error(f"工具调用审计: {audit_info}")
        raise


def execute_tool_with_audit(
    tool_name: str,
    tool_args: dict,
    user_id: str = "system",
    tool_executor: Any = None,
) -> Any:
    """同步版工具权限审计包装器：记录审计日志 + 权限校验"""
    permission = TOOL_PERMISSIONS.get(tool_name, ToolPermission.HIGH)

    audit_info = {
        "timestamp": datetime.now().isoformat(),
        "user_id": user_id,
        "tool_name": tool_name,
        "tool_args": str(tool_args),
        "permission_level": permission.value,
        "action": "attempt",
    }
    audit_logger.info(f"工具调用审计: {audit_info}")

    try:
        result = tool_executor(tool_name, tool_args) if callable(tool_executor) else None
        audit_info["action"] = "success"
        audit_info["result_preview"] = str(result)[:200]
        audit_logger.info(f"工具调用审计: {audit_info}")
        return result
    except Exception as e:
        audit_info["action"] = "failed"
        audit_info["error"] = str(e)
        audit_logger.error(f"工具调用审计: {audit_info}")
        raise
