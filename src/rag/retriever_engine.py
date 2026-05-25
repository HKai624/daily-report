from src.rag.index_builder import get_collection
from src.config import TOP_K_RETRIEVAL


def retrieve(query: str, k: int = TOP_K_RETRIEVAL) -> list[dict]:
    collection = get_collection()
    if collection is None or collection.count() == 0:
        return []

    results = collection.query(query_texts=[query], n_results=min(k, collection.count()))

    docs = []
    if results.get("documents") and results["documents"][0]:
        for i, doc in enumerate(results["documents"][0]):
            meta = {}
            if results.get("metadatas") and results["metadatas"][0]:
                meta = results["metadatas"][0][i] or {}
            dist = None
            if results.get("distances") and results["distances"][0]:
                dist = results["distances"][0][i]
            docs.append({
                "content": doc,
                "source": meta.get("source", "unknown"),
                "distance": dist,
            })
    return docs


def knowledge_is_ready() -> bool:
    collection = get_collection()
    return collection is not None and collection.count() > 0
