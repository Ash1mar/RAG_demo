from __future__ import annotations

import os
import re
from typing import List

import numpy as np
import requests


class Embedder:
    """Embedding helper.

    - `use_mock=True`: deterministic hash-based bag-of-words vectors (no model downloads).
    - `use_mock=False`: requires a remote embedding service (`EMB_URL`) that returns JSON
      `{ "embeddings": [[...], ...] }`.
    """

    def __init__(self, model_name: str, use_mock: bool = False, dim: int = 384):
        # `model_name` is kept for backward compatibility; remote services may ignore it.
        self.use_mock = use_mock
        self.dim = dim
        self._emb_url = os.getenv("EMB_URL")
        self._emb_timeout = float(os.getenv("EMB_TIMEOUT", "12"))
        if not self.use_mock and not self._emb_url:
            raise RuntimeError(
                "Remote embedding URL is not configured. Set EMB_URL or enable MOCK_EMB=1."
            )

    def _mock_encode(self, texts: List[str]) -> np.ndarray:
        def tok(s: str) -> List[str]:
            return re.findall(r"\w+", s.lower(), flags=re.UNICODE)

        vecs = []
        for t in texts:
            v = np.zeros(self.dim, dtype=np.float32)
            for w in tok(t):
                h = hash(w) % self.dim
                v[h] += 1.0
            n = np.linalg.norm(v)
            if n > 0:
                v /= n
            vecs.append(v)
        return np.vstack(vecs)

    def encode(self, texts: List[str]) -> np.ndarray:
        if self.use_mock:
            return self._mock_encode(texts)
        if not self._emb_url:
            raise RuntimeError("EMB_URL must be set when MOCK_EMB=0.")

        payload = {"inputs": list(texts), "normalize": True}
        try:
            resp = requests.post(self._emb_url, json=payload, timeout=self._emb_timeout)
            resp.raise_for_status()
            data = resp.json()
            arr = np.asarray(data.get("embeddings") or data, dtype=np.float32)
            return arr
        except Exception as exc:  # pragma: no cover - defensive
            raise RuntimeError(f"Remote embedding request failed: {exc}") from exc

