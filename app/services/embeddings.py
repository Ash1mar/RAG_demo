import os
from typing import List
import math
import re
import numpy as np

class Embedder:
    """
    use_mock=True 时使用“哈希投影词袋”生成确定性向量（零依赖、启动快）；
    否则加载 SentenceTransformer。
    """
    def __init__(self, model_name: str, use_mock: bool = False, dim: int = 384):
        self.use_mock = use_mock
        self.dim = dim
        self._model = None
        if not self.use_mock:
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
        embs = self._model.encode(texts, normalize_embeddings=True)
        return np.asarray(embs, dtype=np.float32)
