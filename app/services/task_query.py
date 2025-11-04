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
    # remove common punctuation/spaces
    s = re.sub(r"[\s\t\n\r]+", "", s)
    s = re.sub(r"[，。！？、,.!?()（）:：;；\-_/]", "", s)
    return s


INTENT_STATUS_KWS = [
    "完成", "未完成", "是否完成", "状态", "进度", "搞定", "结束", "done", "todo",
]


@dataclass
class ResolverConfig:
    topk: int = 3
    alpha_vec: float = 0.65  # weight for vector score in fusion
    thresh: float = 0.58     # acceptance threshold after fusion


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
        # build normalized text lists
        p_norm = [p for p in self.persons]
        t_norm = [t for t in self.tasks]

        p_vecs = self.embedder.encode(p_norm).astype(np.float32)
        t_vecs = self.embedder.encode(t_norm).astype(np.float32)
        # L2 normalization for cosine via inner product
        if p_vecs.size:
            faiss.normalize_L2(p_vecs)
        if t_vecs.size:
            faiss.normalize_L2(t_vecs)
        self._pers_vecs = p_vecs
        self._task_vecs = t_vecs
        self._idx_person = faiss.IndexFlatIP(p_vecs.shape[1]) if p_vecs.size else None
        self._idx_task = faiss.IndexFlatIP(t_vecs.shape[1]) if t_vecs.size else None
        if self._idx_person is not None:
            self._idx_person.add(p_vecs)
        if self._idx_task is not None:
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
        # rough character overlap score
        qset = set(qn)
        cset = set(cn)
        inter = len(qset & cset)
        union = len(qset | cset) or 1
        return inter / union

    def _search(self, idx: Optional[faiss.Index], vecs: Optional[np.ndarray], cands: List[str], query: str) -> List[Tuple[str, float]]:
        if idx is None or vecs is None or not cands:
            return []
        qv = self.embedder.encode([query]).astype(np.float32)
        faiss.normalize_L2(qv)
        topk = min(self.cfg.topk, len(cands))
        D, I = idx.search(qv, topk)
        scores = D[0]
        ids = I[0]
        out: List[Tuple[str, float]] = []
        for s, i in zip(scores, ids):
            if i < 0 or i >= len(cands):
                continue
            out.append((cands[i], float(s)))
        return out

    def resolve_person(self, query: str) -> List[Tuple[str, float]]:
        # alias substitution first
        q = query
        for alias, real in self.alias_map.items():
            if alias in q:
                q = q.replace(alias, real)
        vec_hits = self._search(self._idx_person, self._pers_vecs, self.persons, q)
        fused: List[Tuple[str, float]] = []
        for cand, vscore in vec_hits:
            kscore = self._kw_score(q, cand)
            score = self.cfg.alpha_vec * vscore + (1 - self.cfg.alpha_vec) * kscore
            fused.append((cand, float(score)))
        fused.sort(key=lambda x: x[1], reverse=True)
        return fused

    def resolve_task(self, query: str) -> List[Tuple[str, float]]:
        vec_hits = self._search(self._idx_task, self._task_vecs, self.tasks, query)
        fused: List[Tuple[str, float]] = []
        for cand, vscore in vec_hits:
            kscore = self._kw_score(query, cand)
            score = self.cfg.alpha_vec * vscore + (1 - self.cfg.alpha_vec) * kscore
            fused.append((cand, float(score)))
        fused.sort(key=lambda x: x[1], reverse=True)
        return fused


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

    def ensure_built(self) -> None:
        if self.resolver is not None:
            return
        persons = self.tasks_store.list_persons()
        tasks = self.tasks_store.list_tasks()
        res = EntityResolver(embedder=self.embedder, persons=persons, tasks=tasks)
        res.build()
        self.resolver = res

    def reload(self) -> Dict[str, Any]:
        self.resolver = None
        self.ensure_built()
        return {"persons": len(self.resolver.persons if self.resolver else []), "tasks": len(self.resolver.tasks if self.resolver else [])}

    def answer(self, q: str, topk: int = 3, thresh: float = 0.58) -> Dict[str, Any]:
        self.ensure_built()
        assert self.resolver is not None

        intent = "status_query" if is_status_intent(q) else "unknown"
        person_hits = self.resolver.resolve_person(q)[:topk]
        task_hits = self.resolver.resolve_task(q)[:topk]

        payload: Dict[str, Any] = {
            "intent": intent,
            "candidates": {
                "persons": [{"value": v, "score": round(float(s), 4)} for v, s in person_hits],
                "tasks": [{"value": v, "score": round(float(s), 4)} for v, s in task_hits],
            },
            "sql": "SELECT id, person, task, status, ts FROM tasks WHERE person = ? AND task = ? ORDER BY ts DESC, id DESC LIMIT 1",
        }

        best_p = person_hits[0] if person_hits else None
        best_t = task_hits[0] if task_hits else None
        if not best_p or not best_t:
            payload["answer"] = "未识别出人员或任务，请从候选中选择。"
            return payload

        if best_p[1] < thresh or best_t[1] < thresh:
            payload["answer"] = "识别置信度不足，请确认候选是否正确。"
            return payload

        person = best_p[0]
        task = best_t[0]
        rec = self.tasks_store.get_latest_status(person, task)
        if not rec:
            payload["answer"] = "未在任务库中找到对应记录。"
            payload["person"] = person
            payload["task"] = task
            return payload

        status = str(rec.get("status", "")).upper()
        ts = int(rec.get("ts", -1))
        ts_str = ts_to_str(ts) if ts >= 0 else "未知时间"

        zh_status = "已完成" if status == "DONE" else "未完成/待办"
        answer = f"{person} 的「{task}」{zh_status}（最近更新时间：{ts_str}）。"

        payload.update({
            "answer": answer,
            "person": person,
            "task": task,
            "status": status,
            "ts": ts,
        })
        return payload

