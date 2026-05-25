import os
import hashlib
import struct
import chromadb
from chromadb.api.types import EmbeddingFunction, Documents
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import CHROMA_PERSIST_DIR, KNOWLEDGE_DIR, CHUNK_SIZE, CHUNK_OVERLAP

COLLECTION_NAME = "workmate_knowledge"
EMBEDDING_DIM = 384


class SimpleEmbeddingFunction(EmbeddingFunction):
    """轻量级嵌入函数，基于字符 n-gram 哈希，无需下载任何模型。"""
    def __call__(self, texts: Documents) -> list[list[float]]:
        embeddings = []
        for text in texts:
            vec = self._text_to_vector(text)
            embeddings.append(vec)
        return embeddings

    def _text_to_vector(self, text: str) -> list[float]:
        vec = [0.0] * EMBEDDING_DIM
        text = text.lower()

        for i in range(len(text)):
            bigram = text[i:i+2]
            if len(bigram) == 2:
                h = hashlib.md5(bigram.encode()).digest()
                idx = struct.unpack("<I", h[:4])[0] % EMBEDDING_DIM
                val = (struct.unpack("<H", h[4:6])[0] / 65535.0) * 2 - 1
                vec[idx] += val * 0.3

        for i in range(len(text)):
            trigram = text[i:i+3]
            if len(trigram) == 3:
                h = hashlib.md5(trigram.encode()).digest()
                idx = struct.unpack("<I", h[:4])[0] % EMBEDDING_DIM
                val = (struct.unpack("<H", h[4:6])[0] / 65535.0) * 2 - 1
                vec[idx] += val * 0.2

        norm = max(sum(v * v for v in vec), 1e-10) ** 0.5
        return [v / norm for v in vec]


def get_chroma_client():
    os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
    return chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)


def build_index():
    client = get_chroma_client()

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    ef = SimpleEmbeddingFunction()
    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", ".", " ", ""],
    )

    docs_loaded = 0
    for root, _, files in os.walk(KNOWLEDGE_DIR):
        for fname in files:
            if fname.startswith("."):
                continue
            fpath = os.path.join(root, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    text = f.read()
            except UnicodeDecodeError:
                continue
            if not text.strip():
                continue
            chunks = splitter.split_text(text)
            for i, chunk in enumerate(chunks):
                collection.add(
                    documents=[chunk],
                    ids=[f"{fname}_{i}"],
                    metadatas=[{"source": fname, "chunk": i}],
                )
                docs_loaded += 1

    return docs_loaded


def get_collection():
    client = get_chroma_client()
    ef = SimpleEmbeddingFunction()
    try:
        return client.get_collection(COLLECTION_NAME, embedding_function=ef)
    except Exception:
        return None
