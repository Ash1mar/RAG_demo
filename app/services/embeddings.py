import os
from typing import List
import math
import re
import json
from urllib.parse import urlparse
import numpy as np
import requests

class Embedder:
    """
    use_mock=True 时使用“哈希投影词袋”生成确定性向量（零依赖、启动快）；
    否则加载 SentenceTransformer。
    """
    def __init__(self, model_name: str, use_mock: bool = False, dim: int = 384):
        self.use_mock = use_mock
        self.dim = dim
        self._model = None
        # Optional remote embedding service endpoint
        self._emb_url = os.getenv("EMB_URL")
        self._emb_timeout = float(os.getenv("EMB_TIMEOUT", "12"))
        if not self.use_mock and not self._emb_url:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(model_name)

    def _mock_encode(self, texts: List[str]) -> np.ndarray:
        def tok(s: str):
            return re.findall(r"\w+", s.lower(), flags=re.UNICODE)
        vecs = []
        for t in texts:
            v = np.zeros(self.dim, dtype=np.float32)
            for w in tok(t):
                h = hash(w) % self.dim
                v[h] += 1.0
            # l2 normalize
            n = np.linalg.norm(v)
            if n > 0:
                v /= n
            vecs.append(v)
        return np.vstack(vecs)

    def encode(self, texts: List[str]) -> np.ndarray:
        if self.use_mock:
            return self._mock_encode(texts)
        # Prefer remote embedder when EMB_URL is configured
        if self._emb_url:
            payload = {"inputs": list(texts), "normalize": True}
            try:
                resp = requests.post(self._emb_url, json=payload, timeout=self._emb_timeout)
                resp.raise_for_status()
                data = resp.json()
                arr = np.asarray(data.get("embeddings") or data, dtype=np.float32)
                return arr
            except Exception as exc:
                # Fallback to local model if available
                if self._model is None:
                    raise
        embs = self._model.encode(texts, normalize_embeddings=True)
        return np.asarray(embs, dtype=np.float32)
