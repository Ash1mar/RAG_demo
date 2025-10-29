# app/services/keyword.py
from __future__ import annotations
from typing import List, Dict, Any
import re

try:
    from rank_bm25 import BM25Okapi
except ImportError as e:
    BM25Okapi = None


def tokenize(text: str) -> List[str]:
    """
    简易分词：英文/数字按单词；中文按单字切分（兼容中英混排）。
    例： "RAG 是增强生成" -> ["rag", "是", "增", "强", "生", "成"]
    """
    text = (text or "").lower()
    # 提取英文/数字 token 或单个 CJK 字符
    tokens = re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", text)
    return tokens


class KeywordIndex:
    """
    关键词倒排与 BM25 检索（Demo 版）
    - 以 chunk 级为文档单位
    - 简化做法：每次增量后重建 BM25（小数据量足够）
    """
    def __init__(self):
        self._docs_tokens: List[List[str]] = []
        self._docs_meta: List[Dict[str, Any]] = []
        self._bm25: BM25Okapi | None = None

    def add(self, doc_id: str, chunks: List[str]) -> None:
        for c in chunks:
            toks = tokenize(c)
            self._docs_tokens.append(toks)
            self._docs_meta.append({"doc_id": doc_id, "text": c})

        if BM25Okapi is None:
            # 没装 rank-bm25 时，仅保留语料，检索返回空
            self._bm25 = None
        else:
            # 直接重建 BM25（小数据量足够）
            self._bm25 = BM25Okapi(self._docs_tokens)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not self._docs_tokens or self._bm25 is None:
            return []
        q = tokenize(query)
        scores = self._bm25.get_scores(q)
        # 取 top_k
        import numpy as np
        k = min(top_k, len(scores))
        idx = np.argpartition(scores, -k)[-k:]
        idx = idx[np.argsort(-scores[idx])]
        results: List[Dict[str, Any]] = []
        for i in idx:
            meta = self._docs_meta[int(i)]
            results.append({"score": float(scores[int(i)]), **meta})
        return results

    def reset(self) -> None:
        self._docs_tokens.clear()
        self._docs_meta.clear()
        self._bm25 = None

    @property
    def size(self) -> int:
        return len(self._docs_meta)
