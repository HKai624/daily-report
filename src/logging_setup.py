import os
import logging
from logging.handlers import RotatingFileHandler

from src.config import DATA_DIR


def setup_logging():
    log_dir = os.path.join(DATA_DIR, "logs")
    os.makedirs(log_dir, exist_ok=True)

    # 基础日志配置
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(module)s | %(message)s",
        handlers=[
            RotatingFileHandler(
                os.path.join(log_dir, "agent_run.log"),
                maxBytes=10 * 1024 * 1024,
                backupCount=5,
                encoding="utf-8",
            ),
            logging.StreamHandler(),
        ],
    )

    # Agent 思考过程日志
    thought_logger = logging.getLogger("agent_thought")
    thought_logger.setLevel(logging.DEBUG)
    thought_logger.propagate = False
    thought_logger.addHandler(
        logging.FileHandler(os.path.join(log_dir, "agent_thought.log"), encoding="utf-8")
    )

    # 会话状态变更日志
    session_logger = logging.getLogger("session_state")
    session_logger.setLevel(logging.INFO)
    session_logger.propagate = False
    session_logger.addHandler(
        logging.FileHandler(os.path.join(log_dir, "session_state.log"), encoding="utf-8")
    )

    # 工具调用审计日志
    audit_logger = logging.getLogger("tool_audit")
    audit_logger.setLevel(logging.INFO)
    audit_logger.propagate = False
    audit_logger.addHandler(
        logging.FileHandler(os.path.join(log_dir, "tool_audit.log"), encoding="utf-8")
    )

    logging.getLogger("agent_thought").info("日志系统初始化完成")
