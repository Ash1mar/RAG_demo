from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import faiss

from app.services.embeddings import Embedder
from app.tasks_store.base import TasksStore


def _norm_text(s: str) -> str:
    s = s.strip()
    s = re.sub(r"[\s\t\n\r]+", "", s)
    s = re.sub(r"[，。！？,.!?()（）:：；\-_/]", "", s)
    return s


INTENT_STATUS_KWS = [
    "完成",
    "未完成",
    "是否完成",
    "状态",
    "进度",
    "搞定",
    "结束",
    "done",
    "todo",
]


@dataclass
class ResolverConfig:
    topk: int = 3
    alpha_vec: float = 0.65  # weight for vector score in fusion
    thresh: float = 0.58     # acceptance threshold after fusion
    mode: str = "hybrid"      # one of: "rules" | "embeddings" | "hybrid"


@dataclass
class EntityResolver:
    embedder: Embedder
    persons: List[str] = field(default_factory=list)
    tasks: List[str] = field(default_factory=list)
    cfg: ResolverConfig = field(default_factory=ResolverConfig)

    # internals
    _idx_person: Optional[faiss.Index] = field(init=False, default=None)
    _idx_task: Optional[faiss.Index] = field(init=False, default=None)
    _pers_vecs: Optional[np.ndarray] = field(init=False, default=None)
    _task_vecs: Optional[np.ndarray] = field(init=False, default=None)

    # simple alias dictionary for people (extendable)
    alias_map: Dict[str, str] = field(default_factory=lambda: {
        "老张": "张三",
    })

    def build(self) -> None:
        # Use original strings (keep case/punct for embeddings), normalization is only for rules
        p_vecs = self.embedder.encode(self.persons).astype(np.float32) if self.persons else np.zeros((0, self.embedder.dim), dtype=np.float32)
        t_vecs = self.embedder.encode(self.tasks).astype(np.float32) if self.tasks else np.zeros((0, self.embedder.dim), dtype=np.float32)
        if p_vecs.size:
            faiss.normalize_L2(p_vecs)
        if t_vecs.size:
            faiss.normalize_L2(t_vecs)
        self._pers_vecs = p_vecs
        self._task_vecs = t_vecs
        self._idx_person = faiss.IndexFlatIP(p_vecs.shape[1]) if p_vecs.size else None
        self._idx_task = faiss.IndexFlatIP(t_vecs.shape[1]) if t_vecs.size else None
        if self._idx_person is not None and p_vecs.shape[0] > 0:
            self._idx_person.add(p_vecs)
        if self._idx_task is not None and t_vecs.shape[0] > 0:
            self._idx_task.add(t_vecs)

    def _kw_score(self, query: str, cand: str) -> float:
        qn = _norm_text(query)
        cn = _norm_text(cand)
        if not qn or not cn:
            return 0.0
        if qn == cn:
            return 1.0
        if qn in cn or cn in qn:
            return 0.8
        qset = set(qn)
        cset = set(cn)
        inter = len(qset & cset)
        union = len(qset | cset) or 1
        return inter / union

    def _search(self, idx: Optional[faiss.Index], vecs: Optional[np.ndarray], cands: List[str], query: str, *, k: Optional[int] = None) -> List[Tuple[str, float]]:
        if idx is None or vecs is None or not cands:
            return []
        qv = self.embedder.encode([query]).astype(np.float32)
        faiss.normalize_L2(qv)
        topk = min(self.cfg.topk if k is None else k, len(cands))
        D, I = idx.search(qv, topk)
        scores = D[0]
        ids = I[0]
        out: List[Tuple[str, float]] = []
        for s, i in zip(scores, ids):
            if i < 0 or i >= len(cands):
                continue
            out.append((cands[i], float(s)))
        return out

    def _rule_rank(self, cands: List[str], query: str, *, k: Optional[int] = None) -> List[Tuple[str, float]]:
        items: List[Tuple[str, float]] = []
        for cand in cands:
            items.append((cand, float(self._kw_score(query, cand))))
        items.sort(key=lambda x: x[1], reverse=True)
        kk = self.cfg.topk if k is None else k
        return items[:kk]

    def _vector_rank_with_focus(
        self,
        vecs: Optional[np.ndarray],
        cands: List[str],
        query: str,
        focus: List[str],
        *,
        k: Optional[int] = None,
    ) -> List[Tuple[str, float]]:
        """
        embeddings-only 的“聚焦查询”实现：
        1) 对整句 query 编码；
        2) 若存在规则法高置信候选（>=0.8），则将这些候选文本也编码；
        3) 用 [query] + focus 的多路向量与候选向量矩阵计算相似度，逐候选取最大值；
        4) 返回 Top‑k。
        说明：build() 已对库向量做 L2 归一化，此处对查询向量再次归一化。
        """
        if vecs is None or vecs.size == 0 or not cands:
            return []
        queries: List[str] = [query]
        if focus:
            queries.extend(focus)
        embs = self.embedder.encode(queries).astype(np.float32)
        faiss.normalize_L2(embs)
        # 余弦相似（已归一化）→ 点积
        sims = embs @ vecs.T  # (m, N)
        best = sims.max(axis=0)  # (N,)
        topk = min(self.cfg.topk if k is None else k, len(cands))
        order = np.argsort(-best)[:topk]
        return [(cands[int(i)], float(best[int(i)])) for i in order]

    def resolve_person(self, query: str) -> List[Tuple[str, float]]:
        # alias substitution first
        q = query
        for alias, real in self.alias_map.items():
            if alias in q:
                q = q.replace(alias, real)
        mode = (self.cfg.mode or "hybrid").lower()
        if mode == "rules":
            return self._rule_rank(self.persons, q)
        if mode == "embeddings":
            # 规则粗提（>=0.8）作为聚焦词，提升短实体名对齐的分数稳定性
            focus = [cand for cand in self.persons if self._kw_score(q, cand) >= 0.8]
            return self._vector_rank_with_focus(self._pers_vecs, self.persons, q, focus)
        # hybrid: collect wider candidates then fuse
        vec_hits = self._search(self._idx_person, self._pers_vecs, self.persons, q, k=max(self.cfg.topk * 2, 10))
        rule_hits = self._rule_rank(self.persons, q, k=max(self.cfg.topk * 2, 10))
        table: Dict[str, Tuple[float, float]] = {}
        for cand, vs in vec_hits:
            table[cand] = (float(vs), float(table.get(cand, (0.0, 0.0))[1]))
        for cand, ks in rule_hits:
            cur = table.get(cand, (0.0, 0.0))
            table[cand] = (float(cur[0]), float(ks))
        fused: List[Tuple[str, float]] = []
        a = float(self.cfg.alpha_vec)
        for cand, (vs, ks) in table.items():
            fused.append((cand, a * vs + (1 - a) * ks))
        fused.sort(key=lambda x: x[1], reverse=True)
        return fused[: self.cfg.topk]

    def resolve_task(self, query: str) -> List[Tuple[str, float]]:
        mode = (self.cfg.mode or "hybrid").lower()
        if mode == "rules":
            return self._rule_rank(self.tasks, query)
        if mode == "embeddings":
            focus = [cand for cand in self.tasks if self._kw_score(query, cand) >= 0.8]
            return self._vector_rank_with_focus(self._task_vecs, self.tasks, query, focus)
        vec_hits = self._search(self._idx_task, self._task_vecs, self.tasks, query, k=max(self.cfg.topk * 2, 10))
        rule_hits = self._rule_rank(self.tasks, query, k=max(self.cfg.topk * 2, 10))
        table: Dict[str, Tuple[float, float]] = {}
        for cand, vs in vec_hits:
            table[cand] = (float(vs), float(table.get(cand, (0.0, 0.0))[1]))
        for cand, ks in rule_hits:
            cur = table.get(cand, (0.0, 0.0))
            table[cand] = (float(cur[0]), float(ks))
        fused: List[Tuple[str, float]] = []
        a = float(self.cfg.alpha_vec)
        for cand, (vs, ks) in table.items():
            fused.append((cand, a * vs + (1 - a) * ks))
        fused.sort(key=lambda x: x[1], reverse=True)
        return fused[: self.cfg.topk]


def is_status_intent(q: str) -> bool:
    s = q.lower()
    return any(kw in s for kw in INTENT_STATUS_KWS)


def ts_to_str(ms: int) -> str:
    try:
        dt = datetime.fromtimestamp(ms / 1000, tz=timezone.utc).astimezone()
        return dt.strftime("%Y-%m-%d %H:%M:%S %Z")
    except Exception:
        return str(ms)


@dataclass
class TaskQueryEngine:
    tasks_store: TasksStore
    embedder: Embedder
    resolver: Optional[EntityResolver] = None
    resolver_mode: str = "hybrid"

    def ensure_built(self) -> None:
        if self.resolver is not None:
            return
        persons = self.tasks_store.list_persons()
        tasks = self.tasks_store.list_tasks()
        # 模式自适应阈值（可被运行时 thresh 覆盖）
        mode = self.resolver_mode
        def _default_thresh(m: str) -> float:
            m = (m or "hybrid").lower()
            if m == "rules":
                return 0.8
            if m == "embeddings":
                return 0.45
            return 0.58  # hybrid

        cfg = ResolverConfig(mode=mode, thresh=_default_thresh(mode))
        res = EntityResolver(
            embedder=self.embedder,
            persons=persons,
            tasks=tasks,
            cfg=cfg,
        )
        res.build()
        self.resolver = res

    def reload(self) -> Dict[str, Any]:
        self.resolver = None
        self.ensure_built()
        return {
            "persons": len(self.resolver.persons if self.resolver else []),
            "tasks": len(self.resolver.tasks if self.resolver else []),
        }

    def answer(self, q: str, topk: int = 3, thresh: Optional[float] = None) -> Dict[str, Any]:
        self.ensure_built()
        assert self.resolver is not None

        # runtime overrides
        if topk != self.resolver.cfg.topk:
            self.resolver.cfg.topk = int(topk)
        if thresh is not None and abs(float(thresh) - self.resolver.cfg.thresh) > 1e-9:
            self.resolver.cfg.thresh = float(thresh)

        intent = "status_query" if is_status_intent(q) else "unknown"
        person_hits = self.resolver.resolve_person(q)[:topk]
        task_hits = self.resolver.resolve_task(q)[:topk]

        payload: Dict[str, Any] = {
            "intent": intent,
            "resolver_mode": self.resolver.cfg.mode,
            "alpha_vec": round(float(self.resolver.cfg.alpha_vec), 4),
            "thresh": round(float(self.resolver.cfg.thresh), 4),
            "candidates": {
                "persons": [{"value": v, "score": round(float(s), 4)} for v, s in person_hits],
                "tasks": [{"value": v, "score": round(float(s), 4)} for v, s in task_hits],
            },
            "sql": "SELECT id, person, task, status, ts FROM tasks WHERE person = ? AND task = ? ORDER BY ts DESC, id DESC LIMIT 1",
        }

        best_p = person_hits[0] if person_hits else None
        best_t = task_hits[0] if task_hits else None
        if not best_p or not best_t:
            payload["answer"] = "未识别出人员或任务，请从候选中选择"
            return payload

        if best_p[1] < self.resolver.cfg.thresh or best_t[1] < self.resolver.cfg.thresh:
            payload["answer"] = "识别置信度不足，请确认候选是否正确"
            return payload

        person = best_p[0]
        task = best_t[0]
        rec = self.tasks_store.get_latest_status(person, task)
        if not rec:
            payload["answer"] = "未在任务库中找到对应记录"
            payload["person"] = person
            payload["task"] = task
            return payload

        status = str(rec.get("status", "")).upper()
        ts = int(rec.get("ts", -1))
        ts_str = ts_to_str(ts) if ts >= 0 else "未知时间"

        zh_status = "已完成" if status == "DONE" else "未完成/待办"
        answer = f"{person} 的「{task}」{zh_status}（最近更新时间：{ts_str}）"

        payload.update({
            "answer": answer,
            "person": person,
            "task": task,
            "status": status,
            "ts": ts,
        })
        return payload
