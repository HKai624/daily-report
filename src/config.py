import os
from dotenv import load_dotenv

load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
MODEL_NAME = "deepseek-chat"
MODEL_NAME_REASONER = "deepseek-reasoner"

EMBEDDING_MODEL = "deepseek-chat"
EMBEDDING_DIM = 1024

CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "chroma_db")
KNOWLEDGE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "knowledge")

CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
TOP_K_RETRIEVAL = 4

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
DB_PATH = os.path.join(DATA_DIR, "checkpoints.db")

STREAM_ENABLED = True
MAX_HISTORY_MESSAGES = 20
