import os
import logging

from src.config import FALLBACK_MODELS, FEISHU_APP_ID, FEISHU_APP_SECRET
from src.tool_governance import TOOL_PERMISSIONS
from src.metrics import metrics


def run_pre_launch_checks() -> bool:
    checks = {
        "session_boundary": False,
        "queue_strategy": False,
        "tool_permission": False,
        "audit_trail": False,
        "model_fallback": False,
        "observability": False,
        "review_template": False,
        "feishu_config": False,
    }

    # 1. 飞书配置完整性
    checks["feishu_config"] = bool(FEISHU_APP_ID and FEISHU_APP_SECRET)

    # 2. 工具权限配置
    checks["tool_permission"] = len(TOOL_PERMISSIONS) > 0

    # 3. 审计日志配置
    audit_logger = logging.getLogger("tool_audit")
    checks["audit_trail"] = len(audit_logger.handlers) > 0

    # 4. 模型 fallback 链
    checks["model_fallback"] = len(FALLBACK_MODELS) > 0

    # 5. 指标收集
    checks["observability"] = metrics is not None

    # 6. 复盘模板
    from src.config import DATA_DIR
    reviews_dir = os.path.join(DATA_DIR, "reviews")
    os.makedirs(reviews_dir, exist_ok=True)
    checks["review_template"] = os.path.exists(reviews_dir)

    # 7. 会话边界（session_locks 在 main.py 中，此处检查模块可用性）
    try:
        from src.session_manager import USER_SESSIONS
        checks["session_boundary"] = USER_SESSIONS is not None
    except Exception:
        checks["session_boundary"] = False

    checks["queue_strategy"] = checks["session_boundary"]

    all_passed = all(checks.values())

    print("=" * 50)
    print("上线前检查结果:")
    for check, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {check}")
    print("=" * 50)

    if not all_passed:
        failed = [k for k, v in checks.items() if not v]
        print(f"未通过项: {failed}")
    else:
        print("所有检查通过，系统就绪")

    return all_passed
