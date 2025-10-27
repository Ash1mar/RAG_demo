from typing import List, Dict, Any
import numpy as np
import faiss
from app.vector_store.base import VectorStore

class FaissVectorStore(VectorStore):
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # 内积 + 预归一化 = 余弦
        self.meta: List[Dict[str, Any]] = []

    def add_texts(self, doc_id: str, chunks: List[str], embeddings: np.ndarray) -> None:
        assert embeddings.shape[1] == self.dim
        # 确保单位向量
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        for c in chunks:
            self.meta.append({"doc_id": doc_id, "text": c})

    def search(self, query_emb: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        q = query_emb.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(q)
        D, I = self.index.search(q, min(top_k, len(self.meta) or 1))
        hits: List[Dict[str, Any]] = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(self.meta):
                continue
            item = self.meta[idx]
            hits.append({"score": float(score), **item})
        return hits

    def reset(self) -> None:
        self.index.reset()
        self.meta.clear()
