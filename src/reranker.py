from sentence_transformers import CrossEncoder
from typing import List, Dict, Any

class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)
        print(f"[INFO] Loaded reranker model: {model_name}")

    def rerank(self, query: str, chunks: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
        if not chunks:
            return []

        pairs = [[query, chunk["text"]] for chunk in chunks]
        scores = self.model.predict(pairs)

        for chunk, score in zip(chunks, scores):
            chunk["rerank_score"] = float(score)

        reranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
        # return reranked[:top_k]
        return [
        {
            "text": chunk["text"],
            "source": chunk["source"],
            "page": chunk["page"],
            "rerank_score": chunk["rerank_score"],
        }
        for chunk in reranked[:top_k]
]
