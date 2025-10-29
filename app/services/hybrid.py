# app/services/hybrid.py
from __future__ import annotations
from typing import List, Dict, Any, Tuple

def _minmax(scores: List[Tuple[str, float]]) -> Dict[str, float]:
    """
    对 (key, score) 列表做 min-max 归一化到 [0,1]
    key 采用 唯一键（如 doc_id + '\t' + text）
    """
    if not scores:
        return {}
    vals = [s for _, s in scores]
    lo, hi = min(vals), max(vals)
    if hi <= lo:
        return {k: 0.0 for k, _ in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores}

def make_key(hit: Dict[str, Any]) -> str:
    # 以 (doc_id, text) 作为唯一键（同一 doc 的相同 chunk 视为同一命中）
    return f"{hit.get('doc_id','')}\t{hit.get('text','')}"

def merge_scores(
    vector_hits: List[Dict[str, Any]],
    keyword_hits: List[Dict[str, Any]],
    k: int = 5,
    alpha: float = 0.6,
) -> List[Dict[str, Any]]:
    """
    将两路命中结果融合：
    - 先分别 min-max 归一化到 [0,1]
    - 最终得分 = alpha * vec_score + (1-alpha) * kw_score
    """
    # 1) 准备 (key, score) 列表
    vec_pairs = [(make_key(h), float(h.get("score", 0.0))) for h in vector_hits]
    kw_pairs  = [(make_key(h), float(h.get("score", 0.0))) for h in keyword_hits]

    # 2) 归一化
    vec_norm = _minmax(vec_pairs)
    kw_norm  = _minmax(kw_pairs)

    # 3) 收集所有键并融合
    keys = set(vec_norm.keys()) | set(kw_norm.keys())
    fused: List[Tuple[str, float]] = []
    for key in keys:
        vs = vec_norm.get(key, 0.0)
        ks = kw_norm.get(key, 0.0)
        fused.append((key, alpha * vs + (1 - alpha) * ks))

    # 4) 取 Top-K
    fused.sort(key=lambda x: x[1], reverse=True)
    top = fused[:k]

    # 5) 还原出字典（以 vector_hits 优先回填 meta，不在 vec 中则从 kw 中拿）
    #   这样可得到 {score, doc_id, text}
    cache = {}
    for h in vector_hits + keyword_hits:
        cache.setdefault(make_key(h), h)

    results: List[Dict[str, Any]] = []
    for key, s in top:
        base = dict(cache.get(key, {}))
        base["score"] = float(s)
        results.append(base)
    return results
