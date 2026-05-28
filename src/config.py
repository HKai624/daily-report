import os
from dotenv import load_dotenv

load_dotenv()

# ── LLM 配置 ──────────────────────────────────────────────────
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
MODEL_NAME = "deepseek-chat"
MODEL_NAME_REASONER = "deepseek-reasoner"

EMBEDDING_MODEL = "deepseek-chat"
EMBEDDING_DIM = 1024

# ── RAG 配置 ──────────────────────────────────────────────────
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "chroma_db")
KNOWLEDGE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "knowledge")

CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
TOP_K_RETRIEVAL = 4

# ── 数据存储 ──────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
DB_PATH = os.path.join(DATA_DIR, "checkpoints.db")

# ── 飞书基础配置 ──────────────────────────────────────────────
FEISHU_APP_ID = os.getenv("FEISHU_APP_ID", "")
FEISHU_APP_SECRET = os.getenv("FEISHU_APP_SECRET", "")
FEISHU_VERIFICATION_TOKEN = os.getenv("FEISHU_VERIFICATION_TOKEN", "")
FEISHU_ENCRYPT_KEY = os.getenv("FEISHU_ENCRYPT_KEY", "")

# ── 服务运行配置 ──────────────────────────────────────────────
SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("SERVER_PORT", "8000"))
DEBUG_MODE = os.getenv("DEBUG_MODE", "true").lower() == "true"

# ── Agent 核心参数 ────────────────────────────────────────────
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "10"))
API_RETRY_TIMES = int(os.getenv("API_RETRY_TIMES", "2"))
SESSION_MAX_LENGTH = int(os.getenv("SESSION_MAX_LENGTH", "8"))
SESSION_EXPIRE_DAYS = int(os.getenv("SESSION_EXPIRE_DAYS", "7"))
SESSION_EXPIRE_SECONDS = SESSION_EXPIRE_DAYS * 24 * 3600

# ── 稳定性治理参数 ────────────────────────────────────────────
FALLBACK_MODELS = ["deepseek-chat", "deepseek-reasoner"]
RATE_LIMIT_PER_USER = int(os.getenv("RATE_LIMIT_PER_USER", "10"))
TOOL_CALL_TIMEOUT = int(os.getenv("TOOL_CALL_TIMEOUT", "30"))

STREAM_ENABLED = True
MAX_HISTORY_MESSAGES = 20

# ── 智谱 GLM-4.6V 视觉配置 ─────────────────────────────────────
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY", "64396755eafc4dd49f09040ab6d2c6c5.SVfVGZIv0tjMm1Li")
VISION_MODEL = os.getenv("VISION_MODEL", "glm-4.6v")
VISION_ENDPOINT = "https://open.bigmodel.cn/api/paas/v4/chat/completions"

# ── 图片上传配置 ────────────────────────────────────────────────
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", f"http://localhost:{SERVER_PORT}")
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "uploads")

# ── 每日早报配置 ──────────────────────────────────────────────
ENABLE_LLM_SUMMARY = os.getenv("ENABLE_LLM_SUMMARY", "true").lower() == "true"
NEWS_MAX_ITEMS = int(os.getenv("NEWS_MAX_ITEMS", "5"))
