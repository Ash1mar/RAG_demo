# app/services/answer.py
from __future__ import annotations
from typing import List, Dict, Any


def build_answer(
    hits: List[Dict[str, Any]],
    max_chars: int = 600,
    include_scores: bool = True,
) -> Dict[str, Any]:
    """
    将检索到的片段拼接成一个可读“回答”，附带 citations。
    - 简单去重（同一文本只取一次）
    - 按分数高低拼接，控制总字数上限
    """
    used = set()
    parts: List[str] = []
    citations: List[Dict[str, Any]] = []

    total = 0
    for h in hits:
        text = (h.get("text") or "").strip()
        if not text or text in used:
            continue
        # 预估加入后长度，超限则停止
        if total + len(text) + (2 if parts else 0) > max_chars:
            break

        parts.append(text)
        used.add(text)
        total += len(text) + (2 if parts else 0)

        c = {"doc_id": h.get("doc_id"), "text": text}
        if include_scores and "score" in h:
            c["score"] = float(h["score"])
        citations.append(c)

    answer = "  ".join(parts) if parts else ""
    return {"answer": answer, "citations": citations}
