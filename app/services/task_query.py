from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import httpx
import faiss
import numpy as np
from pydantic import BaseModel, ValidationError

from app.config import llm_settings
from app.services.embeddings import Embedder
from app.services.nl2sql_engine import (
    QueryFilter,
    TaskAnswerMode,
    TaskQueryIntent,
    TaskStatus,
    TaskQuerySpec,
    build_task_query_plan,
    is_complex_by_text,
    is_simple_intent,
    parse_task_query_nl,
    too_many_entities,
)
from app.services.sql_compiler import TaskSqlCompileError, compile_tasks_sql
from app.tasks_store.base import TasksStore


def _norm_text(s: str) -> str:
    """Lightweight normalization for rule-based matching."""
    s = s.strip()
    s = re.sub(r"[\s\t\n\r]+", "", s)
    s = re.sub(r"[，。！？!?.()（）:：；、\-_/]", "", s)
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


logger = logging.getLogger(__name__)

TEXT2SQL_SCHEMA = """
table task_latest (
  id INTEGER PRIMARY KEY,
  person TEXT NOT NULL,
  task TEXT NOT NULL,
  status TEXT NOT NULL,        -- DONE | TODO | IN_PROGRESS | BLOCKED
  ts INTEGER NOT NULL,         -- epoch milliseconds
  project TEXT,
  tags TEXT,                   -- comma-separated strings
  priority INTEGER,            -- 1 = highest priority
  due_ts INTEGER,
  created_ts INTEGER,
  updated_ts INTEGER,
  status_note TEXT
);

table tasks (
  id INTEGER PRIMARY KEY,
  person TEXT NOT NULL,
  task TEXT NOT NULL,
  status TEXT NOT NULL,
  ts INTEGER NOT NULL,
  project TEXT,
  tags TEXT,
  priority INTEGER,
  due_ts INTEGER,
  created_ts INTEGER,
  updated_ts INTEGER,
  status_note TEXT
);
"""

TEXT2SQL_SYSTEM_PROMPT = (
    "You are a precise Text-to-SQL assistant for a SQLite database that tracks task status updates. "
    "You must only emit read-only SELECT statements that reference the task_latest or tasks tables described in the schema. "
    "Never produce DML/DDL (INSERT/UPDATE/DELETE/ALTER/etc.), and always include a LIMIT of at most 100 rows. "
    "Return only JSON with the structure {\"queries\":[{\"sql\":\"...\",\"description\":\"...\"}]}. "
    "If the request needs multiple SQL statements, include up to two queries in the JSON array."
)

TEXT2SQL_MAX_QUERIES = 2
TEXT2SQL_ROW_PREVIEW = 3
TEXT2SQL_FORBIDDEN_KEYWORDS = (
    'insert',
    'update',
    'delete',
    'drop',
    'alter',
    'truncate',
    'create',
    'attach',
    'detach',
    'pragma',
)
TEXT2SQL_ALLOWED_TABLE_SNIPPETS = (' from task_latest', ' from tasks')

class Text2SQLQueryModel(BaseModel):
    sql: str
    description: Optional[str] = None

class Text2SQLResponseModel(BaseModel):
    queries: List[Text2SQLQueryModel]

class Text2SQLGenerateError(Exception):
    """Raised when the LLM cannot produce a valid Text2SQL payload."""

    def __init__(self, message: str, *, raw_response: Optional[str] = None):
        super().__init__(message)
        self.raw_response: Optional[str] = raw_response

class Text2SQLValidationError(Exception):
    """Raised when the generated SQL does not pass safety checks."""


@dataclass
class ResolverConfig:
    topk: int = 3
    alpha_vec: float = 1.0  # default; may be tuned per mode
    thresh: float = 0.58  # default; may be overridden per mode
    mode: str = "hybrid"  # one of: "rules" | "embeddings" | "hybrid" | "hybrid_plus_rules"
    # Fine-grained controls (hybrid/hybrid_plus_rules)
    thresh_person: Optional[float] = None
    thresh_task: Optional[float] = None
    delta_min: Optional[float] = None  # Top1-Top2 margin for weak accept
    weak_task_min: Optional[float] = None  # low bar for weak accept
    rules_assist_min: Optional[float] = None  # relaxed low bar when rules strongly agree


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
    alias_map: Dict[str, str] = field(default_factory=lambda: {"老张": "张三"})

    def build(self) -> None:
        # Use original strings (keep case/punct for embeddings); normalization is only for rules
        p_vecs = (
            self.embedder.encode(self.persons).astype(np.float32)
            if self.persons
            else np.zeros((0, self.embedder.dim), dtype=np.float32)
        )
        t_vecs = (
            self.embedder.encode(self.tasks).astype(np.float32)
            if self.tasks
            else np.zeros((0, self.embedder.dim), dtype=np.float32)
        )
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

    def _search(
        self,
        idx: Optional[faiss.Index],
        vecs: Optional[np.ndarray],
        cands: List[str],
        query: str,
        *,
        k: Optional[int] = None,
    ) -> List[Tuple[str, float]]:
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
        Matrix-based "focus query" for embeddings-only mode:
        - Encode [query] plus any high-confidence rule matches (focus);
        - Compute cosine sim against all candidates and take per-candidate max;
        - Return top-k.
        """
        if vecs is None or vecs.size == 0 or not cands:
            return []
        queries: List[str] = [query]
        if focus:
            queries.extend(focus)
        embs = self.embedder.encode(queries).astype(np.float32)
        faiss.normalize_L2(embs)
        sims = embs @ vecs.T  # cosine sim because all vectors are L2-normalized
        best = sims.max(axis=0)
        topk = min(self.cfg.topk if k is None else k, len(cands))
        order = np.argsort(-best)[:topk]
        return [(cands[int(i)], float(best[int(i)])) for i in order]

    def _faiss_rank_with_focus(
        self,
        idx: Optional[faiss.Index],
        vecs: Optional[np.ndarray],
        cands: List[str],
        query: str,
        focus: List[str],
        *,
        k: Optional[int] = None,
    ) -> List[Tuple[str, float]]:
        """
        FAISS variant of focus query, keeping behavior aligned with the
        matrix-based version while using the FAISS index for retrieval.
        """
        if idx is None or vecs is None or not cands:
            return []
        queries: List[str] = [query]
        if focus:
            queries.extend(focus)
        embs = self.embedder.encode(queries).astype(np.float32)
        faiss.normalize_L2(embs)
        n = len(cands)
        best = np.full(n, -1.0, dtype=np.float32)
        K_all = n  # small candidate sets: search all to avoid truncation misses
        for i in range(embs.shape[0]):
            D, I = idx.search(embs[i : i + 1], K_all)
            scores = D[0]
            ids = I[0]
            for s, j in zip(scores, ids):
                if 0 <= j < n and s > best[j]:
                    best[j] = float(s)
        topk = min(self.cfg.topk if k is None else k, n)
        order = np.argsort(-best)[:topk]
        return [(cands[int(i)], float(best[int(i)])) for i in order]

    def resolve_person(self, query: str) -> List[Tuple[str, float]]:
        # Alias substitution first
        q = query
        for alias, real in self.alias_map.items():
            if alias in q:
                q = q.replace(alias, real)
        mode = (self.cfg.mode or "hybrid").lower()
        if mode == "rules":
            return self._rule_rank(self.persons, q)
        if mode == "embeddings":
            focus = [cand for cand in self.persons if self._kw_score(q, cand) >= 0.8]
            return self._vector_rank_with_focus(self._pers_vecs, self.persons, q, focus)
        # hybrid: vector-only with FAISS focus query
        focus = [cand for cand in self.persons if self._kw_score(q, cand) >= 0.8]
        return self._faiss_rank_with_focus(self._idx_person, self._pers_vecs, self.persons, q, focus, k=self.cfg.topk)

    def resolve_task(self, query: str) -> List[Tuple[str, float]]:
        mode = (self.cfg.mode or "hybrid").lower()
        if mode == "rules":
            return self._rule_rank(self.tasks, query)
        if mode == "embeddings":
            focus = [cand for cand in self.tasks if self._kw_score(query, cand) >= 0.8]
            return self._vector_rank_with_focus(self._task_vecs, self.tasks, query, focus)
        # hybrid: vector-only with FAISS focus query
        focus = [cand for cand in self.tasks if self._kw_score(query, cand) >= 0.8]
        return self._faiss_rank_with_focus(self._idx_task, self._task_vecs, self.tasks, query, focus, k=self.cfg.topk)


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

        mode = self.resolver_mode

        def _default_thresh(m: str) -> float:
            m = (m or "hybrid").lower()
            if m == "rules":
                return 0.8
            if m == "embeddings":
                return 0.45
            # hybrid / hybrid_llm
            return 0.45

        cfg = ResolverConfig(mode=mode, thresh=_default_thresh(mode))
        res = EntityResolver(embedder=self.embedder, persons=persons, tasks=tasks, cfg=cfg)
        res.build()

        m = (mode or "hybrid").lower()
        if m in ("hybrid", "hybrid_llm"):
            res.cfg.thresh_person = 0.45
            res.cfg.thresh_task = 0.40
            res.cfg.delta_min = 0.09
            res.cfg.weak_task_min = 0.40
            res.cfg.alpha_vec = 1.0
        elif m == "hybrid_plus_rules":
            res.cfg.thresh_person = 0.45
            res.cfg.thresh_task = 0.40
            res.cfg.delta_min = 0.09
            res.cfg.weak_task_min = 0.40
            res.cfg.rules_assist_min = 0.37
            res.cfg.alpha_vec = 0.9
        self.resolver = res

    def reload(self) -> Dict[str, Any]:
        self.resolver = None
        self.ensure_built()
        return {
            "persons": len(self.resolver.persons if self.resolver else []),
            "tasks": len(self.resolver.tasks if self.resolver else []),
        }

    @staticmethod
    def _intent_label(intent: Optional[TaskQueryIntent]) -> str:
        if isinstance(intent, TaskQueryIntent):
            if intent == TaskQueryIntent.task_list_by_person:
                return "task_list"
            if intent == TaskQueryIntent.task_history:
                return "task_history"
            if intent in (
                TaskQueryIntent.task_status_single,
                TaskQueryIntent.task_status_list,
            ):
                return "status_query"
            if intent == TaskQueryIntent.person_summary:
                return "person_summary"
        return "unknown"

    def _compute_routing_debug(self, spec: TaskQuerySpec) -> Tuple[bool, Dict[str, Any]]:
        complex_flag = is_complex_by_text(getattr(spec, "raw_query", ""))
        multi_flag = too_many_entities(spec)
        simple_flag = is_simple_intent(spec)
        intent = getattr(spec, "intent", None)
        debug = {
            "intent": intent.value if isinstance(intent, TaskQueryIntent) else str(intent),
            "is_supported": getattr(spec, "is_supported", None),
            "intent_confidence": getattr(spec, "intent_confidence", None),
            "raw_intent_nl": getattr(spec, "raw_intent_nl", None),
            "complex_by_text": complex_flag,
            "too_many_entities": multi_flag,
            "is_simple_intent": simple_flag,
        }
        return simple_flag, debug

    def _align_filters_with_resolver(self, spec: TaskQuerySpec) -> Dict[str, List[str]]:
        """
        Align QueryFilter values for person/task using the resolver so that
        downstream SQL compilation can leverage multi-value filters directly.
        """
        alignment: Dict[str, List[str]] = {}
        filters = getattr(spec, "filters", None) or []
        if not filters or self.resolver is None:
            return alignment

        for field_name in ("person", "task"):
            raw_values: List[str] = []
            target_filters: List[QueryFilter] = []
            for flt in filters:
                if not isinstance(flt, QueryFilter):
                    continue
                field = str(getattr(flt, "field", "") or "")
                if field.lower() != field_name:
                    continue
                target_filters.append(flt)
                op = str(getattr(flt, "op", "eq") or "").lower()
                if op == "in":
                    values = getattr(flt, "values", None) or []
                    raw_values.extend([str(v) for v in values if v])
                elif getattr(flt, "value", None):
                    raw_values.append(str(flt.value))
            if not raw_values:
                continue

            resolved_values: List[str] = []
            resolver_fn = (
                self.resolver.resolve_person
                if field_name == "person"
                else self.resolver.resolve_task
            )
            for raw in raw_values:
                hits = resolver_fn(raw)
                if hits:
                    resolved_values.append(hits[0][0])
                else:
                    resolved_values.append(raw)

            alignment[field_name] = resolved_values

            idx = 0
            for flt in target_filters:
                op = str(getattr(flt, "op", "eq") or "").lower()
                if op == "in":
                    flt.values = list(resolved_values)
                    idx = len(resolved_values)
                else:
                    if idx < len(resolved_values):
                        flt.value = resolved_values[idx]
                        idx += 1

            if field_name == "person":
                if len(resolved_values) > 1:
                    spec.person = None
                elif len(resolved_values) == 1 and not spec.person:
                    spec.person = resolved_values[0]
            else:
                if len(resolved_values) > 1:
                    spec.task = None
                elif len(resolved_values) == 1 and not spec.task:
                    spec.task = resolved_values[0]

        return alignment

    def answer(self, q: str, topk: int = 3, thresh: Optional[float] = None) -> Dict[str, Any]:
        """Main entry for non-LLM task status queries."""
        mode_raw = (self.resolver_mode or "hybrid").lower()
        nl2sql_attempted = False
        nl2sql_error: Optional[Dict[str, Any]] = None

        if mode_raw == "nl2sql":
            nl2sql_attempted = True
            try:
                return self._answer_via_nl2sql(q)
            except Exception as exc:  # defensive: fall back to legacy flow
                nl2sql_error = {
                    "resolver_mode": "nl2sql_failed_fallback_legacy",
                    "nl2sql_error": "unexpected_failure",
                    "nl2sql_reason": str(exc),
                }

        if mode_raw == "hybrid_llm":
            result = self._answer_via_hybrid_llm(q, topk=topk, thresh=thresh)
            if nl2sql_attempted and nl2sql_error is not None:
                result.update(nl2sql_error)
            return result

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

        mode_decide_rank = (self.resolver.cfg.mode or "hybrid").lower()
        if mode_decide_rank == "hybrid_plus_rules" and task_hits:
            # light rule-assisted re-scoring
            lambda_rule = 0.15
            kw_tokens = ["接口", "联调"]
            enriched: List[Tuple[str, float, float]] = []
            for idx, (val, score_vec) in enumerate(task_hits):
                rule_s = float(self.resolver._kw_score(q, val))
                kw_bonus = 0.05 if any(tok in q and tok in val for tok in kw_tokens) else 0.0
                score_final = float(score_vec) + lambda_rule * rule_s + kw_bonus
                if rule_s >= 0.8 and idx in (1, 2):
                    score_final += 0.2
                enriched.append((val, float(score_vec), score_final))
            enriched.sort(key=lambda x: x[2], reverse=True)
            task_hits = [(v, sv) for (v, sv, _) in enriched]

        payload: Dict[str, Any] = {
            "intent": intent,
            "resolver_mode": self.resolver.cfg.mode,
            "alpha_vec": round(float(self.resolver.cfg.alpha_vec), 4),
            "thresh": round(float(self.resolver.cfg.thresh), 4),
            "candidates": {
                "persons": [{"value": v, "score": round(float(s), 4)} for v, s in person_hits],
                "tasks": [{"value": v, "score": round(float(s), 4)} for v, s in task_hits],
            },
            "sql": "SELECT id, person, task, status, ts FROM task_latest WHERE person = ? AND task = ? ORDER BY ts DESC, id DESC LIMIT 1",
        }

        best_p = person_hits[0] if person_hits else None
        best_t = task_hits[0] if task_hits else None
        if not best_p and spec.person:
            best_p = (spec.person, 1.0)
        if not best_t and spec.task:
            best_t = (spec.task, 1.0)
        if not best_p or not best_t:
            payload["answer"] = "Could not resolve person or task; please pick from candidates."
            return payload

        low_conf = False
        mode_decide = (self.resolver.cfg.mode or "hybrid").lower()
        if mode_decide in ("hybrid", "hybrid_plus_rules", "hybrid_llm"):
            tp = 0.45 if getattr(self.resolver.cfg, "thresh_person", None) is None else float(self.resolver.cfg.thresh_person)
            tt = 0.40 if getattr(self.resolver.cfg, "thresh_task", None) is None else float(self.resolver.cfg.thresh_task)
            delta_min = 0.09 if getattr(self.resolver.cfg, "delta_min", None) is None else float(self.resolver.cfg.delta_min)
            weak_min = tt if getattr(self.resolver.cfg, "weak_task_min", None) is None else float(self.resolver.cfg.weak_task_min)

            p_ok = best_p[1] >= tp
            t_ok = best_t[1] >= tt
            if not t_ok:
                second_t = task_hits[1][1] if len(task_hits) > 1 else -1.0
                if best_t[1] >= weak_min and (second_t >= 0) and (best_t[1] - second_t) >= delta_min:
                    t_ok = True
                    low_conf = True
                elif mode_decide == "hybrid_plus_rules":
                    rule_s = float(self.resolver._kw_score(q, best_t[0]))
                    assist_min = weak_min if getattr(self.resolver.cfg, "rules_assist_min", None) is None else float(self.resolver.cfg.rules_assist_min)
                    if rule_s >= 0.8 and best_t[1] >= assist_min:
                        t_ok = True
                        low_conf = True

            if not p_ok or not t_ok:
                payload["answer"] = "Low confidence; please confirm the candidates."
                return payload
            # prevent legacy single-threshold check below from blocking
            self.resolver.cfg.thresh = 0.0

        if best_p[1] < self.resolver.cfg.thresh or best_t[1] < self.resolver.cfg.thresh:
            payload["answer"] = "Low confidence; please confirm the candidates."
            return payload

        person = best_p[0]
        task = best_t[0]
        rec = self.tasks_store.get_latest_status(person, task)
        if not rec:
            payload["answer"] = "No matching record found in the task store."
            payload["person"] = person
            payload["task"] = task
            return payload

        status = str(rec.get("status", "")).upper()
        ts = int(rec.get("ts", -1))
        ts_str = ts_to_str(ts) if ts >= 0 else "unknown time"

        answer = f'{person} / "{task}" is {"completed" if status == "DONE" else "not completed"} (latest update: {ts_str}).'

        payload.update(
            {
                "answer": answer,
                "person": person,
                "task": task,
                "status": status,
                "ts": ts,
            }
        )
        if low_conf:
            payload["answer"] = str(payload.get("answer", "")) + " (low confidence)"
        if nl2sql_attempted and nl2sql_error is not None:
            payload.update(nl2sql_error)
        return payload

    def _answer_via_hybrid_llm(self, q: str, topk: int = 3, thresh: Optional[float] = None) -> Dict[str, Any]:
        """NL -> IR (LLM/rules) + hybrid entity alignment + SQL compiler."""
        payload: Dict[str, Any] = {
            "intent": "unknown",
            "resolver_mode": "hybrid_llm",
        }

        # 1) NL -> IR (LLM/rules)
        try:
            spec = parse_task_query_nl(q)
        except Exception as exc:
            payload["error"] = "hybrid_llm_parse_failed"
            payload["reason"] = str(exc)
            return payload

        spec_intent = getattr(spec, "intent", None)
        payload["intent"] = self._intent_label(spec_intent)

        raw_answer_mode = getattr(spec, "answer_mode", TaskAnswerMode.default)
        if isinstance(raw_answer_mode, TaskAnswerMode):
            answer_mode_hint = raw_answer_mode
        elif isinstance(raw_answer_mode, str):
            try:
                answer_mode_hint = TaskAnswerMode(raw_answer_mode)
            except ValueError:
                answer_mode_hint = TaskAnswerMode.default
        else:
            answer_mode_hint = TaskAnswerMode.default

        is_simple, routing_debug = self._compute_routing_debug(spec)
        if not is_simple:
            routing_debug["routed_via"] = "text2sql"
            payload["routing_debug"] = routing_debug
            return self._answer_via_text2sql(q, spec, payload)
        routing_debug["routed_via"] = "hybrid_llm"
        payload["routing_debug"] = routing_debug

        # 2) Hybrid entity alignment on top of IR
        self.ensure_built()
        assert self.resolver is not None

        if topk != self.resolver.cfg.topk:
            self.resolver.cfg.topk = int(topk)
        if thresh is not None and abs(float(thresh) - self.resolver.cfg.thresh) > 1e-9:
            self.resolver.cfg.thresh = float(thresh)

        filter_alignment = self._align_filters_with_resolver(spec)
        person_filter_values = filter_alignment.get("person", [])
        task_filter_values = filter_alignment.get("task", [])

        q_person = spec.person or (person_filter_values[0] if person_filter_values else q)
        q_task = spec.task or (task_filter_values[0] if task_filter_values else q)

        person_filters_active = bool(person_filter_values)
        task_filters_active = bool(task_filter_values)

        person_hits = (
            [(val, 1.0) for val in person_filter_values]
            if person_filters_active
            else self.resolver.resolve_person(q_person)[:topk]
        )
        task_hits = (
            [(val, 1.0) for val in task_filter_values]
            if task_filters_active
            else self.resolver.resolve_task(q_task)[:topk]
        )

        payload.update(
            {
                "alpha_vec": round(float(self.resolver.cfg.alpha_vec), 4),
                "thresh": round(float(self.resolver.cfg.thresh), 4),
                "candidates": {
                    "persons": [{"value": v, "score": round(float(s), 4)} for v, s in person_hits],
                    "tasks": [{"value": v, "score": round(float(s), 4)} for v, s in task_hits],
                },
            }
        )
        if person_filters_active:
            payload["filter_persons"] = person_filter_values
        if task_filters_active:
            payload["filter_tasks"] = task_filter_values

        best_p = person_hits[0] if person_hits else None
        best_t = task_hits[0] if task_hits else None
        task_required = True
        if spec_intent in (TaskQueryIntent.task_list_by_person, TaskQueryIntent.person_summary):
            task_required = False
        elif spec_intent == TaskQueryIntent.task_status_list and not spec.task:
            # ?????????????????????????????????????
            task_required = False
        if answer_mode_hint in (
            TaskAnswerMode.task_count_by_status,
            TaskAnswerMode.person_summary_by_project,
            TaskAnswerMode.overdue_count_by_person,
        ):
            task_required = False

        if spec_intent in (TaskQueryIntent.task_status_list,):
            person_required = bool(person_filters_active or spec.person)
        elif spec_intent == TaskQueryIntent.task_list_by_person:
            person_required = bool(person_filters_active or spec.person)
        elif spec_intent in (TaskQueryIntent.task_status_single, TaskQueryIntent.task_history):
            person_required = True
        elif answer_mode_hint in (
            TaskAnswerMode.task_count_by_status,
            TaskAnswerMode.person_summary_by_project,
            TaskAnswerMode.overdue_count_by_person,
        ):
            person_required = False
        else:
            person_required = bool(person_filters_active or spec.person)


        if (person_required and not best_p) or (task_required and not best_t):
            payload["error"] = "hybrid_llm_no_candidates"
            payload["answer"] = "Could not resolve person or task; please pick from candidates."
            return payload

        low_conf = False
        mode_decide = (self.resolver.cfg.mode or "hybrid").lower()
        if mode_decide in ("hybrid", "hybrid_plus_rules", "hybrid_llm"):
            tp = 0.45 if getattr(self.resolver.cfg, "thresh_person", None) is None else float(self.resolver.cfg.thresh_person)
            tt = 0.40 if getattr(self.resolver.cfg, "thresh_task", None) is None else float(self.resolver.cfg.thresh_task)
            delta_min = 0.09 if getattr(self.resolver.cfg, "delta_min", None) is None else float(self.resolver.cfg.delta_min)
            weak_min = tt if getattr(self.resolver.cfg, "weak_task_min", None) is None else float(self.resolver.cfg.weak_task_min)

            if person_required:
                if person_filters_active:
                    p_ok = True
                else:
                    p_ok = bool(best_p) and (best_p[1] >= tp)
                    if not p_ok and spec.person:
                        p_ok = True
                        low_conf = True
            else:
                p_ok = True
            t_ok = True
            if task_required:
                t_ok = task_filters_active or (best_t[1] >= tt)
                if not t_ok and not task_filters_active:
                    second_t = task_hits[1][1] if len(task_hits) > 1 else -1.0
                    if best_t[1] >= weak_min and (second_t >= 0) and (best_t[1] - second_t) >= delta_min:
                        t_ok = True
                        low_conf = True
                    elif spec.task:
                        t_ok = True
                        low_conf = True

            if not p_ok or not t_ok:
                payload["answer"] = "Low confidence; please confirm the candidates."
                payload["error"] = "hybrid_llm_low_confidence"
                return payload

        if person_filters_active:
            person = None
        elif best_p:
            person = best_p[0]
        else:
            person = spec.person
        if task_filters_active:
            task_val = None
        elif best_t:
            task_val = best_t[0]
        else:
            task_val = spec.task

        # 清理 status: 去掉 ANY，列表类查询可选择性放宽
        if getattr(spec, "status", None):
            spec.status = [
                s
                for s in spec.status
                if not (isinstance(s, TaskStatus) and s == TaskStatus.ANY) and str(s).upper() != "ANY"
            ]
            if not spec.status and spec_intent in (TaskQueryIntent.task_list_by_person, TaskQueryIntent.task_status_list):
                spec.status = []

        if spec_intent == TaskQueryIntent.task_list_by_person:
            spec.person = None if person_filters_active else person
            spec.task = None
            if not getattr(spec, "status", None):
                # Only widen to “all statuses” when用户没有指定状态过滤
                spec.status = []
        else:
            if person_filters_active:
                spec.person = None
            else:
                spec.person = person
            if task_required:
                if task_filters_active:
                    spec.task = None
                else:
                    spec.task = task_val or spec.task
            else:
                spec.task = None

        try:
            payload["nl_ir"] = spec.dict()
            payload["ir"] = build_task_query_plan(spec)
        except Exception:
            payload["nl_ir"] = {"raw_query": q, "error": "failed_to_serialize_aligned_ir"}
            payload["ir"] = {"raw_query": q, "error": "failed_to_build_query_plan_from_aligned_spec"}

        try:
            compiled = compile_tasks_sql(spec)
        except TaskSqlCompileError as exc:
            payload["error"] = "hybrid_llm_compile_failed"
            payload["reason"] = str(exc)
            payload["person"] = person
            payload["task"] = task_val
            return payload

        payload["sql"] = compiled.sql
        payload["params"] = compiled.params

        query_fn = getattr(self.tasks_store, "query", None)
        if query_fn is None:
            payload["error"] = "hybrid_llm_query_not_supported_by_tasks_store"
            return payload

        def _run_query(sql: str, params: Tuple[Any, ...]) -> List[Dict[str, Any]]:
            return query_fn(sql, params)

        rows: List[Dict[str, Any]] = []
        try:
            rows = _run_query(compiled.sql, compiled.params)
        except Exception as exc:  # defensive
            payload["error"] = "hybrid_llm_db_query_failed"
            payload["reason"] = str(exc)
            return payload

        # Fallback: relax status filter when it blocks recall
        if not rows and getattr(spec, "status", None):
            try:
                relaxed_spec = spec.copy(deep=True)
                relaxed_spec.status = []
                relaxed_compiled = compile_tasks_sql(relaxed_spec)
                rows = _run_query(relaxed_compiled.sql, relaxed_compiled.params)
                payload["relaxed_status_filter"] = True
                payload["sql_relaxed"] = relaxed_compiled.sql
                payload["params_relaxed"] = relaxed_compiled.params
            except Exception:
                # Best-effort fallback; ignore errors here
                pass

        payload["rows"] = rows

        if not rows:
            payload["answer"] = "No matching records found (hybrid_llm)."
            payload["person"] = person
            payload["task"] = task_val
            return payload

        answer_mode = answer_mode_hint
        payload["answer_mode"] = answer_mode.value

        if answer_mode == TaskAnswerMode.completion_time_latest:
            done_row = next(
                (
                    rec
                    for rec in rows
                    if str(rec.get("status", "")).upper() == TaskStatus.DONE.value
                ),
                rows[0],
            )
            ts = int(done_row.get("ts", -1))
            ts_str = ts_to_str(ts) if ts >= 0 else "unknown time"
            payload.update(
                {
                    "answer": f'{person} / "{task_val or spec.task}" was completed at {ts_str}.',
                    "person": person,
                    "task": task_val or spec.task,
                    "status": str(done_row.get("status", "")).upper(),
                    "ts": ts,
                }
            )
            if low_conf:
                payload["answer"] = str(payload.get("answer", "")) + " (low confidence)"
            return payload

        if answer_mode == TaskAnswerMode.task_count_by_status:
            counts_map: Dict[str, int] = {}
            for rec in rows:
                status = str(rec.get("status", "")).upper() or "UNKNOWN"
                raw_count = rec.get("task_count")
                try:
                    cnt = int(raw_count)
                except (TypeError, ValueError):
                    cnt = 1
                if cnt < 0:
                    cnt = 0
                counts_map[status] = counts_map.get(status, 0) + cnt
            counts = [
                {"status": status, "count": counts_map[status]}
                for status in sorted(counts_map.keys(), key=lambda s: (-counts_map[s], s))
            ]
            total = sum(item["count"] for item in counts)
            stats_str = ", ".join(f"{item['status']}={item['count']}" for item in counts) or "none"
            if person_filters_active and person_filter_values:
                subject_label = ", ".join(person_filter_values)
            elif person:
                subject_label = str(person)
            else:
                subject_label = "Tasks"
            scope_bits: List[str] = []
            time_range = getattr(spec, "time_range", None)
            if time_range:
                scope_bits.append(
                    f"time_range={getattr(time_range, 'start', None) or '*'}~{getattr(time_range, 'end', None) or '*'}"
                )
            due_range = getattr(spec, "due_range", None)
            if due_range:
                scope_bits.append(
                    f"due_range={getattr(due_range, 'start', None) or '*'}~{getattr(due_range, 'end', None) or '*'}"
                )
            scope_suffix = f" within {', '.join(scope_bits)}" if scope_bits else ""
            if subject_label == "Tasks":
                answer_prefix = "Tasks by status"
            else:
                answer_prefix = f"{subject_label} tasks by status"
            payload.update(
                {
                    "answer": f"{answer_prefix}{scope_suffix}: {stats_str} (total {total}).",
                    "person": None if person_filters_active else person,
                    "persons": person_filter_values if person_filters_active else ([person] if person else []),
                    "task": None,
                    "status_counts": counts,
                    "total_tasks": total,
                }
            )
            if low_conf:
                payload["answer"] = str(payload.get("answer", "")) + " (low confidence)"
            return payload

        if answer_mode == TaskAnswerMode.person_summary_by_project:
            summary: Dict[str, Dict[str, Dict[str, int]]] = {}
            for rec in rows:
                project = str(rec.get("project", "") or "Unspecified")
                person_name = str(rec.get("person", "") or "Unknown")
                status_val = str(rec.get("status", "") or "UNKNOWN").upper()
                count_val = rec.get("task_count")
                try:
                    cnt = int(count_val)
                except (TypeError, ValueError):
                    cnt = 0
                summary.setdefault(project, {}).setdefault(person_name, {})[status_val] = cnt

            parts: List[str] = []
            for project, people in summary.items():
                person_bits: List[str] = []
                for person_name, status_map in people.items():
                    status_bits = [f"{status}={count}" for status, count in status_map.items()]
                    person_bits.append(f"{person_name}({', '.join(status_bits)})")
                project_summary = "; ".join(person_bits) if person_bits else "no data"
                parts.append(f"{project}: {project_summary}")
            answer = " | ".join(parts) if parts else "No summary data."
            payload.update(
                {
                    "answer": f"Project/person status summary: {answer}",
                    "project_summary": summary,
                    "person": None,
                    "persons": [],
                    "task": None,
                }
            )
            if low_conf:
                payload["answer"] = str(payload.get("answer", "")) + " (low confidence)"
            return payload

        if answer_mode == TaskAnswerMode.overdue_count_by_person:
            rows_summary: List[Dict[str, Any]] = []
            for rec in rows:
                person_name = str(rec.get("person", "") or "Unknown")
                raw_count = rec.get("overdue_count")
                try:
                    cnt = int(raw_count)
                except (TypeError, ValueError):
                    cnt = 0
                rows_summary.append({"person": person_name, "count": cnt})
            rows_summary.sort(key=lambda item: (-item["count"], item["person"]))
            scope_bits: List[str] = []
            time_range = getattr(spec, "time_range", None)
            if time_range:
                scope_bits.append(
                    f"time_range={getattr(time_range, 'start', None) or '*'}~{getattr(time_range, 'end', None) or '*'}"
                )
            due_range = getattr(spec, "due_range", None)
            if due_range:
                scope_bits.append(
                    f"due_range={getattr(due_range, 'start', None) or '*'}~{getattr(due_range, 'end', None) or '*'}"
                )
            scope_suffix = f" within {', '.join(scope_bits)}" if scope_bits else ""
            summary_str = ", ".join(f"{item['person']}={item['count']}" for item in rows_summary) or "none"
            payload.update(
                {
                    "answer": f"Overdue tasks per person{scope_suffix}: {summary_str}.",
                    "overdue_counts": rows_summary,
                    "person": None,
                    "persons": [item["person"] for item in rows_summary],
                    "task": None,
                }
            )
            if low_conf:
                payload["answer"] = str(payload.get("answer", "")) + " (low confidence)"
            return payload

        spec_intent = getattr(spec, "intent", None)

        if spec_intent == TaskQueryIntent.task_list_by_person:
            count = len(rows)
            preview_tasks = []
            for rec in rows[:5]:
                t_name = str(rec.get("task", ""))
                t_status = str(rec.get("status", "")).upper()
                rec_person = str(rec.get("person", ""))
                if person_filters_active and rec_person:
                    preview_tasks.append(f"{rec_person}:{t_name}({t_status})")
                else:
                    preview_tasks.append(f"{t_name}({t_status})")
            preview = ", ".join(preview_tasks) if preview_tasks else "none"
            if person_filters_active:
                names = ", ".join(person_filter_values)
                payload.update(
                    {
                        "answer": f"Tasks for {names}: {preview}",
                        "person": None,
                        "persons": person_filter_values,
                        "task": None,
                    }
                )
            else:
                payload.update(
                    {
                        "answer": f"{person} has {count} tasks: {preview}",
                        "person": person,
                        "task": None,
                    }
                )
            if low_conf:
                payload["answer"] = str(payload.get("answer", "")) + " (low confidence)"
            return payload

        if spec_intent == TaskQueryIntent.task_status_list:
            count = len(rows)
            preview = []
            for rec in rows[:5]:
                t_name = str(rec.get("task", ""))
                t_status = str(rec.get("status", "")).upper()
                rec_person = str(rec.get("person", ""))
                if person_filters_active and rec_person:
                    preview.append(f"{rec_person}:{t_name}({t_status})")
                else:
                    preview.append(f"{t_name}({t_status})")
            preview_str = ", ".join(preview) if preview else "none"
            if person_filters_active:
                names = ", ".join(person_filter_values)
                payload.update(
                    {
                        "answer": f"{names} have {count} task status records: {preview_str}",
                        "person": None,
                        "persons": person_filter_values,
                        "task": None,
                    }
                )
            else:
                payload.update(
                    {
                        "answer": f"{person} has {count} task status records: {preview_str}",
                        "person": person,
                        "task": None,
                    }
                )
            if low_conf:
                payload["answer"] = str(payload.get("answer", "")) + " (low confidence)"
            return payload

        if spec_intent == TaskQueryIntent.task_history:
            count = len(rows)
            rec = rows[0]
            status = str(rec.get("status", "")).upper()
            ts = int(rec.get("ts", -1))
            ts_str = ts_to_str(ts) if ts >= 0 else "unknown time"
            payload.update(
                {
                    "answer": f'{person} / "{task_val}" has {count} status records; latest is {status} at {ts_str}.',
                    "person": person,
                    "task": task_val,
                    "status": status,
                    "ts": ts,
                }
            )
            if low_conf:
                payload["answer"] = str(payload.get("answer", "")) + " (low confidence)"
            return payload

        if spec_intent == TaskQueryIntent.person_summary:
            summary: Dict[str, List[str]] = {}
            for rec in rows:
                p_name = str(rec.get("person", ""))
                status = str(rec.get("status", "")).upper()
                count_val = rec.get("task_count")
                try:
                    cnt = int(count_val)
                except (TypeError, ValueError):
                    cnt = count_val
                summary.setdefault(p_name, []).append(f"{status}={cnt}")
            parts = []
            for p_name, stats in summary.items():
                stats_str = ", ".join(stats)
                parts.append(f"{p_name}: {stats_str}")
            answer = "; ".join(parts) if parts else "No summary data."
            payload.update(
                {
                    "answer": answer,
                    "person": None if person_filters_active else person,
                    "persons": person_filter_values if person_filters_active else ([person] if person else []),
                    "task": None,
                }
            )
            if low_conf:
                payload["answer"] = str(payload.get("answer", "")) + " (low confidence)"
            return payload

        rec = rows[0]
        status = str(rec.get("status", "")).upper()
        ts = int(rec.get("ts", -1))
        ts_str = ts_to_str(ts) if ts >= 0 else "unknown time"
        payload.update(
            {
                "answer": f'{person} / "{task_val}" is {"completed" if status == "DONE" else status.lower()} (latest update: {ts_str}).',
                "person": person,
                "task": task_val,
                "status": status,
                "ts": ts,
            }
        )
        if low_conf:
            payload["answer"] = str(payload.get("answer", "")) + " (low confidence)"
        return payload

    def _resolve_via_ir_fast_path(self, spec: TaskQuerySpec, routing_debug: Dict[str, Any]) -> Dict[str, Any]:
        debug = dict(routing_debug or {})
        debug["routed_via"] = "ir_fast_path"
        payload: Dict[str, Any] = {
            "intent": self._intent_label(getattr(spec, "intent", None)),
            "resolver_mode": "hybrid_llm_ir_fast_path",
            "routing_debug": debug,
        }
        return self._execute_ir_plan(
            spec,
            payload,
            compile_error_code="ir_fast_path_compile_failed",
            query_not_supported_code="ir_fast_path_query_not_supported_by_tasks_store",
            query_error_code="ir_fast_path_db_query_failed",
            no_rows_message="No matching records found (IR fast path).",
        )

    def _answer_via_nl2sql(self, q: str) -> Dict[str, Any]:
        """Experimental NL→JSON→SQL resolver path."""
        payload: Dict[str, Any] = {
            "intent": "unknown",
            "resolver_mode": "nl2sql",
        }

        try:
            spec = parse_task_query_nl(q)
        except Exception as exc:
            payload["error"] = "nl2sql_parse_failed"
            payload["reason"] = str(exc)
            return payload

        payload["intent"] = self._intent_label(getattr(spec, "intent", None))
        return self._execute_ir_plan(
            spec,
            payload,
            compile_error_code="nl2sql_compile_failed",
            query_not_supported_code="nl2sql_query_not_supported_by_tasks_store",
            query_error_code="nl2sql_db_query_failed",
            no_rows_message="No matching records found (NL2SQL).",
        )

    def _execute_ir_plan(
        self,
        spec: TaskQuerySpec,
        payload: Dict[str, Any],
        *,
        compile_error_code: str,
        query_not_supported_code: str,
        query_error_code: str,
        no_rows_message: str,
    ) -> Dict[str, Any]:
        payload["nl_ir"] = spec.dict()
        try:
            compiled = compile_tasks_sql(spec)
        except TaskSqlCompileError as exc:
            payload["error"] = compile_error_code
            payload["reason"] = str(exc)
            return payload

        payload["sql"] = compiled.sql
        payload["params"] = compiled.params

        query_fn = getattr(self.tasks_store, "query", None)
        if query_fn is None:
            payload["error"] = query_not_supported_code
            return payload

        def _run_query(sql: str, params: Tuple[Any, ...]) -> List[Dict[str, Any]]:
            return query_fn(sql, params)

        try:
            rows = _run_query(compiled.sql, compiled.params)
        except Exception as exc:
            payload["error"] = query_error_code
            payload["reason"] = str(exc)
            return payload

        payload["rows"] = rows

        if rows:
            rec = rows[0]
            person = rec.get("person")
            task = rec.get("task")
            status = str(rec.get("status", "")).upper()
            ts = int(rec.get("ts", -1))
            ts_str = ts_to_str(ts) if ts >= 0 else "unknown time"
            payload.update(
                {
                    "answer": f'{person} / "{task}" is {"completed" if status == "DONE" else status.lower()} (latest update: {ts_str}).',
                    "person": person,
                    "task": task,
                    "status": status,
                    "ts": ts,
                }
            )
        else:
            payload["answer"] = no_rows_message
        return payload

    def _answer_via_text2sql(
        self,
        q: str,
        spec: Optional[TaskQuerySpec],
        payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Text2SQL pipeline entry point (LLM-generated SQL with safety checks)."""

        base: Dict[str, Any] = dict(payload or {})
        base["resolver_mode"] = "text2sql"

        if spec is not None:
            base.setdefault("intent", self._intent_label(getattr(spec, "intent", None)))
            base["nl_ir"] = spec.dict()
        else:
            base.setdefault("intent", "unknown")
            base["nl_ir"] = {"error": "missing_spec"}

        if not llm_settings.enabled or llm_settings.provider == "dummy":
            base["error"] = "text2sql_llm_disabled"
            base["answer"] = "Text2SQL pipeline requires a configured LLM provider."
            return base
        if llm_settings.provider != "ollama":
            base["error"] = "text2sql_llm_provider_unsupported"
            base["answer"] = (
                f"Text2SQL is not yet supported for provider {llm_settings.provider}."
            )
            return base

        prompt = _build_text2sql_prompt(q, spec)
        try:
            llm_result = _call_text2sql_llm(prompt)
        except Text2SQLGenerateError as exc:
            base["error"] = "text2sql_llm_failed"
            base["answer"] = "Failed to generate SQL from the LLM."
            base["reason"] = str(exc)
            if getattr(exc, "raw_response", None):
                base["text2sql_raw_response"] = exc.raw_response
            logger.warning("Text2SQL LLM failed: %s", exc)
            return base

        if not llm_result.queries:
            base["error"] = "text2sql_empty_response"
            base["answer"] = "LLM did not return any SQL queries."
            return base

        query_fn = getattr(self.tasks_store, "query", None)
        if query_fn is None:
            base["error"] = "text2sql_query_not_supported_by_tasks_store"
            base["answer"] = "Current TasksStore cannot execute SQL queries."
            return base

        executed: List[Dict[str, Any]] = []
        primary_rows: List[Dict[str, Any]] = []
        primary_sql: Optional[str] = None

        for item in llm_result.queries[:TEXT2SQL_MAX_QUERIES]:
            try:
                normalized_sql = _normalize_and_validate_text2sql_query(item.sql)
            except Text2SQLValidationError as exc:
                base["error"] = "text2sql_invalid_sql"
                base["answer"] = "Generated SQL failed validation."
                base["reason"] = str(exc)
                base["invalid_sql"] = item.sql
                return base

            try:
                rows = query_fn(normalized_sql, tuple())
            except Exception as exc:
                base["error"] = "text2sql_db_query_failed"
                base["answer"] = "Generated SQL failed when executed against the database."
                base["reason"] = str(exc)
                base["sql"] = normalized_sql
                base["params"] = []
                logger.warning("Text2SQL query failed: %s", exc)
                return base

            executed.append(
                {
                    "sql": normalized_sql,
                    "description": item.description,
                    "rows": rows,
                }
            )
            if not primary_rows:
                primary_rows = rows
                primary_sql = normalized_sql

        base["text2sql"] = executed
        if primary_sql is not None:
            base["sql"] = primary_sql
            base["params"] = []
            base["rows"] = primary_rows

        base.pop("error", None)
        base.pop("reason", None)

        if primary_rows:
            base["answer"] = _summarize_text2sql_rows(primary_rows)
        else:
            base["answer"] = "Text2SQL query returned no rows."

        return base


def _build_text2sql_prompt(question: str, spec: Optional[TaskQuerySpec]) -> str:
    hint = _make_text2sql_ir_hint(spec)
    hint_json = json.dumps(hint, ensure_ascii=False, indent=2)
    return (
        "Generate at most two SQL queries that answer the user's question using the "
        "SQLite schema below. SQL requirements:\n"
        "- Only SELECT statements are allowed.\n"
        "- Target the task_latest or tasks tables (task_latest contains the latest row per person+task).\n"
        "- Always include an ORDER BY when the user cares about recency.\n"
        "- ALWAYS include a LIMIT clause (<= 100 rows).\n"
        "- Do not invent tables or columns.\n"
        "- Do not use parameters; embed literal values directly in the SQL.\n"
        "\n"
        "Return your answer as pure JSON matching this shape (no extra commentary):\n"
        '{"queries":[{"sql":"SELECT ...","description":"short natural language summary"}]}\n'
        "\n"
        "### Database schema\n"
        f"{TEXT2SQL_SCHEMA.strip()}\n\n"
        "### Natural language question\n"
        f"{question}\n\n"
        "### IR hint (may contain mistakes, but usually helpful)\n"
        f"{hint_json}\n"
    )


def _make_text2sql_ir_hint(spec: Optional[TaskQuerySpec]) -> Dict[str, Any]:
    if spec is None:
        return {}

    def _enum_to_str(value: Any) -> Any:
        if isinstance(value, Enum):
            return value.value
        return value

    hint: Dict[str, Any] = {
        "intent": _enum_to_str(getattr(spec.intent, "value", spec.intent))
        if getattr(spec, "intent", None) is not None
        else None,
        "person": spec.person,
        "task": spec.task,
        "project": spec.project,
        "tags": spec.tags,
        "priority": spec.priority,
        "status": [_enum_to_str(s) for s in (spec.status or [])],
        "time_range": spec.time_range.dict() if spec.time_range else None,
        "due_range": spec.due_range.dict() if spec.due_range else None,
        "created_range": spec.created_range.dict() if spec.created_range else None,
        "limit": spec.limit,
        "order_by": [ob.dict() for ob in (spec.order_by or [])],
        "filters": [flt.dict() for flt in (spec.filters or [])],
    }
    return hint


def _call_text2sql_llm(prompt: str) -> Text2SQLResponseModel:
    if not llm_settings.enabled or llm_settings.provider == "dummy":
        raise Text2SQLGenerateError("LLM provider is not configured")
    if llm_settings.provider != "ollama":
        raise Text2SQLGenerateError(
            f"Provider {llm_settings.provider} is not supported for Text2SQL"
        )

    payload: Dict[str, Any] = {
        "model": llm_settings.model,
        "messages": [
            {"role": "system", "content": TEXT2SQL_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {
            "temperature": 0.0,
        },
    }

    url = f"{llm_settings.ollama_base_url.rstrip('/')}/api/chat"
    try:
        resp = httpx.post(url, json=payload, timeout=60.0)
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise Text2SQLGenerateError(f"Ollama request failed: {exc}") from exc

    response_text = resp.text
    try:
        data = resp.json()
        content = data["message"]["content"]
    except (ValueError, KeyError, TypeError) as exc:
        raise Text2SQLGenerateError(
            "Invalid response format from Ollama", raw_response=response_text
        ) from exc

    try:
        json_payload = _extract_json_payload(content)
    except ValueError as exc:
        raise Text2SQLGenerateError(
            "LLM output is not valid JSON", raw_response=content
        ) from exc

    try:
        parsed = json.loads(json_payload)
    except json.JSONDecodeError as exc:
        raise Text2SQLGenerateError(
            "LLM output is not valid JSON", raw_response=content
        ) from exc

    try:
        return Text2SQLResponseModel.parse_obj(parsed)
    except ValidationError as exc:
        raise Text2SQLGenerateError(
            f"LLM JSON does not match expected schema: {exc}", raw_response=content
        ) from exc


def _extract_json_payload(raw: str) -> str:
    if not raw:
        raise ValueError("empty response")
    s = raw.strip()
    if not s:
        raise ValueError("empty response")

    if s.startswith("```"):
        # Strip leading ```lang(optional)\n
        end_fence = s.find("```", 3)
        if end_fence == -1:
            raise ValueError("unterminated markdown fence")
        inner = s[3:end_fence]
        # remove optional language tag (e.g., 'json')
        inner = inner.lstrip()
        if inner.lower().startswith("json"):
            inner = inner[4:].lstrip()
        return inner.strip()

    start = s.find("{")
    end = s.find("}", start)
    if start != -1 and end != -1:
        candidate = s[start : s.rfind("}") + 1]
        candidate = candidate.strip()
        if candidate.endswith("```"):
            candidate = candidate[:-3].strip()
        tail = s[s.rfind("}") + 1 :].strip()
        for marker in ("```", "^^^", "---"):
            marker_pos = candidate.find(marker)
            if marker_pos != -1:
                candidate = candidate[:marker_pos].strip()
        return candidate
    raise ValueError("no JSON object found")


def _normalize_and_validate_text2sql_query(sql: str) -> str:
    if not sql or not sql.strip():
        raise Text2SQLValidationError("SQL query is empty.")
    normalized = sql.strip()
    if normalized.endswith(";"):
        normalized = normalized[:-1].strip()
    lowered = normalized.lower()
    if not lowered.startswith("select"):
        raise Text2SQLValidationError("Only SELECT statements are permitted.")
    if not any(snippet in lowered for snippet in TEXT2SQL_ALLOWED_TABLE_SNIPPETS):
        raise Text2SQLValidationError("Query must reference task_latest or tasks.")
    if not re.search(r"\blimit\b", lowered):
        raise Text2SQLValidationError("Query must include a LIMIT clause.")
    for keyword in TEXT2SQL_FORBIDDEN_KEYWORDS:
        if keyword in lowered:
            raise Text2SQLValidationError(f"Forbidden keyword detected: {keyword}")
    return normalized


def _summarize_text2sql_rows(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "Text2SQL query returned no rows."

    parts: List[str] = []
    preview = rows[:TEXT2SQL_ROW_PREVIEW]
    for row in preview:
        person = row.get("person") or "unknown person"
        task = row.get("task") or "unknown task"
        status = str(row.get("status", "") or "").upper() or "UNKNOWN"
        ts_val = row.get("ts")
        try:
            ts_int = int(ts_val)
        except (TypeError, ValueError):
            ts_int = None
        ts_str = ts_to_str(ts_int) if ts_int is not None else str(ts_val)
        parts.append(f'{person} / "{task}" -> {status} (ts={ts_str})')

    remainder = len(rows) - len(preview)
    summary = "; ".join(parts)
    if remainder > 0:
        summary += f" (+{remainder} more)"
    return summary
