from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import httpx
import faiss
import numpy as np
from pydantic import BaseModel, ValidationError
from sqlglot import parse_one, exp

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
from app.tasks_domain import get_tasks_domain
from app.tasks_intent import get_intent_handler, intent_label
from app.tasks_intent.base import AnswerContext
from app.tasks_store.base import TasksStore
from app.tasks_schema import TasksSchemaConfig, get_tasks_schema_config


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

_TEXT2SQL_COLUMN_SPECS: Tuple[Tuple[str, str, Optional[str]], ...] = (
    ("id", "INTEGER PRIMARY KEY", None),
    ("person", "TEXT NOT NULL", None),
    ("owner", "TEXT", None),
    ("task", "TEXT NOT NULL", None),
    ("status", "TEXT NOT NULL", "DONE | TODO | IN_PROGRESS | BLOCKED"),
    ("ts", "INTEGER NOT NULL", "epoch milliseconds"),
    ("project", "TEXT", None),
    ("tags", "TEXT", "comma-separated strings"),
    ("org_name", "TEXT", None),
    ("division_name", "TEXT", None),
    ("post_name", "TEXT", None),
    ("is_read", "INTEGER", "0/1"),
    ("is_delegated", "INTEGER", "0/1"),
    ("priority", "INTEGER", "1 = highest priority"),
    ("due_ts", "INTEGER", None),
    ("created_ts", "INTEGER", None),
    ("updated_ts", "INTEGER", None),
    ("status_note", "TEXT", None),
    ("description", "TEXT", None),
)


def _normalize_relation_name(name: str) -> str:
    text = (name or "").strip()
    if not text:
        return ""
    text = text.strip("[]")
    if "." in text:
        text = text.split(".")[-1]
    return text.strip("[]").lower()


def _allowed_tables() -> set[str]:
    schema = get_tasks_schema_config()
    allowed = {_normalize_relation_name(name) for name in schema.allowed_relations}
    return {name for name in allowed if name}


def _tsql_prefix_unicode_literals(sql: str) -> str:
    """Ensure Unicode string literals use N'...' in T-SQL.

    SQL Server treats non-N-prefixed string literals as VARCHAR, which can cause
    comparisons like person = '杨洁' to fail depending on collation/codepage.
    """
    if not sql:
        return sql

    def _needs_prefix(literal: str) -> bool:
        # Only prefix when literal contains any non-ascii character.
        return any(ord(ch) > 127 for ch in literal)

    # Match single-quoted literals, respecting escaped quotes ('').
    pattern = re.compile(r"(?<![Nn])'((?:[^']|'')*)'")

    def _repl(match: re.Match) -> str:
        inner = match.group(1) or ""
        if not _needs_prefix(inner):
            return match.group(0)
        return "N'" + inner + "'"

    return pattern.sub(_repl, sql)


def _text2sql_disallowed_comparisons(schema: TasksSchemaConfig) -> Tuple[str, ...]:
    ts_col = schema.translate_field("ts")
    created_col = schema.translate_field("created_ts")
    due_col = schema.translate_field("due_ts")

    pairs: List[str] = []

    def _emit(left: str, right: str) -> None:
        if not left or not right:
            return
        for op in (">", "<", ">=", "<="):
            pairs.append(f"{left} {op} {right}".lower())

    _emit(ts_col, created_col)
    _emit(created_col, ts_col)
    _emit(ts_col, due_col)
    _emit(due_col, ts_col)
    return tuple(sorted(set(pairs)))


def _text2sql_literal_columns(schema: TasksSchemaConfig) -> List[str]:
    cols: List[str] = []
    for logical in ("ts", "created_ts", "due_ts"):
        col = schema.translate_field(logical)
        if col and col not in cols:
            cols.append(col)
    return cols


def _render_text2sql_columns(schema: TasksSchemaConfig) -> str:
    rendered: List[Tuple[str, str, str]] = []
    for logical, col_type, comment in _TEXT2SQL_COLUMN_SPECS:
        physical = schema.translate_field(logical)
        if not physical:
            continue
        notes: List[str] = []
        if comment:
            notes.append(comment)
        if physical != logical:
            notes.append(f"logical: {logical}")
        comment_text = f" -- {', '.join(notes)}" if notes else ""
        rendered.append((physical, col_type, comment_text))

    lines: List[str] = []
    for idx, (physical, col_type, comment_text) in enumerate(rendered):
        comma = "," if idx + 1 < len(rendered) else ""
        lines.append(f"  {physical} {col_type}{comma}{comment_text}")
    return "\n".join(lines)


def _build_text2sql_schema(schema: TasksSchemaConfig) -> str:
    columns = _render_text2sql_columns(schema)
    return f"""
table {schema.latest_relation} (
{columns}
);

table {schema.history_relation} (
{columns}
);
"""


def _text2sql_schema() -> str:
    return _build_text2sql_schema(get_tasks_schema_config())

TEXT2SQL_SYSTEM_PROMPT_SQLITE = (
    "You are a precise Text-to-SQL assistant for a SQLite database that tracks task status updates. "
    "You must only emit read-only SELECT statements that reference the task_latest or tasks tables described in the schema. "
    "Never produce DML/DDL (INSERT/UPDATE/DELETE/ALTER/etc.), and always include a LIMIT of at most 100 rows. "
    "When the question explicitly says \"executor\"/\"执行\"/\"执行人\", filter by the `person` column. "
    "When it says \"owner\"/\"发起人\"/\"负责人\" (or similar), filter by the `owner` column. "
    "Return only JSON with the structure {\"queries\":[{\"sql\":\"...\",\"description\":\"...\"}]}. "
    "If the request needs multiple SQL statements, include up to two queries in the JSON array."
)

TEXT2SQL_SYSTEM_PROMPT_MSSQL = (
    "You are a precise Text-to-SQL assistant for a SQL Server (T-SQL) database that tracks task status updates. "
    "You must only emit read-only SELECT statements that reference the task_latest or tasks tables described in the schema. "
    "Never produce DML/DDL (INSERT/UPDATE/DELETE/ALTER/etc.), and always include TOP (N) with N <= 100. "
    "Do not use LIMIT; SQL Server requires TOP. "
    "When the question explicitly says \"executor\"/\"执行\"/\"执行人\", filter by the `person` column. "
    "When it says \"owner\"/\"发起人\"/\"负责人\" (or similar), filter by the `owner` column. "
    "Return only JSON with the structure {\"queries\":[{\"sql\":\"...\",\"description\":\"...\"}]}. "
    "If the request needs multiple SQL statements, include up to two queries in the JSON array."
)

_TEXT2SQL_JSON_SHAPE = (
    "Return only JSON with the structure {\"queries\":[{\"sql\":\"...\",\"description\":\"...\"}]}. "
)

TEXT2SQL_MAX_QUERIES = 2
TEXT2SQL_ROW_PREVIEW = 3
TEXT2SQL_ANSWER_MAX_ROWS = 5
TEXT2SQL_FORBIDDEN_KEYWORDS = (
    "insert",
    "update",
    "delete",
    "drop",
    "alter",
    "truncate",
    "create",
    "attach",
    "detach",
    "pragma",
)
TEXT2SQL_DISALLOWED_FUNCTIONS = ('date_sub', 'curdate')
TEXT2SQL_SUSPICIOUS_LITERAL_THRESHOLD = 10_000_000_000
_MILLIS_PER_SECOND = 1000
_MILLIS_PER_DAY = 24 * 60 * 60 * _MILLIS_PER_SECOND
TEXT2SQL_ANSWER_SYSTEM_PROMPT = (
    "You are a precise assistant for summarizing SQL query results about task statuses. "
    "Use the provided rows to answer the user's question in Chinese. "
    "Base the answer strictly on the rows; if no relevant data is provided, say that no matching records were found. "
    "Be concise and do not invent information beyond the rows or obvious aggregations."
)


def _resolve_text2sql_dialect() -> str:
    raw = os.getenv("TASKS_TEXT2SQL_DIALECT") or os.getenv("TASKS_DIALECT")
    if raw:
        value = raw.strip().lower()
        if value:
            return value
    backend = os.getenv("TASKS_BACKEND", "sqlite").strip().lower()
    return "mssql" if backend == "mssql" else "sqlite"


def _sqlglot_dialect(dialect: str) -> str:
    return "tsql" if (dialect or "").strip().lower() == "mssql" else "sqlite"


def _text2sql_system_prompt(dialect: str) -> str:
    schema = get_tasks_schema_config()
    person_col = schema.translate_field("person") or "person"
    owner_col = schema.translate_field("owner") or "owner"
    tables = f"{schema.latest_relation} or {schema.history_relation}"
    if (dialect or "").strip().lower() == "mssql":
        return (
            "You are a precise Text-to-SQL assistant for a SQL Server (T-SQL) database that tracks task status updates. "
            f"You must only emit read-only SELECT statements that reference the {tables} tables described in the schema. "
            "Never produce DML/DDL (INSERT/UPDATE/DELETE/ALTER/etc.), and always include TOP (N) with N <= 100. "
            "Do not use LIMIT; SQL Server requires TOP. "
            "When the question explicitly says \"executor\"/\"执行\"/\"执行人\", "
            f"filter by the `{person_col}` column. "
            "When it says \"owner\"/\"发起人\"/\"负责人\" (or similar), "
            f"filter by the `{owner_col}` column. "
            + _TEXT2SQL_JSON_SHAPE
            + "If the request needs multiple SQL statements, include up to two queries in the JSON array."
        )
    return (
        "You are a precise Text-to-SQL assistant for a SQLite database that tracks task status updates. "
        f"You must only emit read-only SELECT statements that reference the {tables} tables described in the schema. "
        "Never produce DML/DDL (INSERT/UPDATE/DELETE/ALTER/etc.), and always include a LIMIT of at most 100 rows. "
        "When the question explicitly says \"executor\"/\"执行\"/\"执行人\", "
        f"filter by the `{person_col}` column. "
        "When it says \"owner\"/\"发起人\"/\"负责人\" (or similar), "
        f"filter by the `{owner_col}` column. "
        + _TEXT2SQL_JSON_SHAPE
        + "If the request needs multiple SQL statements, include up to two queries in the JSON array."
    )


def _resolve_symbolic_time(text: str) -> Optional[int]:
    """Resolve tokens like now-7d, now, start_of_week into epoch millis."""
    t = text.strip().lower()
    if not t:
        return None
    now = datetime.now(timezone.utc)
    if t == "now":
        return int(now.timestamp() * 1000)
    if t == "start_of_week":
        start = now - timedelta(days=now.weekday())
        start = start.replace(hour=0, minute=0, second=0, microsecond=0)
        return int(start.timestamp() * 1000)
    if t == "end_of_week":
        start = now - timedelta(days=now.weekday())
        end = start + timedelta(days=7, seconds=-1)
        return int(end.timestamp() * 1000)
    if t == "start_of_month":
        start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        return int(start.timestamp() * 1000)
    if t == "end_of_month":
        if now.month == 12:
            next_month = now.replace(year=now.year + 1, month=1, day=1)
        else:
            next_month = now.replace(month=now.month + 1, day=1)
        end = next_month - timedelta(seconds=1)
        return int(end.timestamp() * 1000)
    if t == "next_week":
        start = now - timedelta(days=now.weekday())
        next_start = start + timedelta(days=7)
        return int(next_start.timestamp() * 1000)
    if t == "next_month":
        if now.month == 12:
            next_month = now.replace(year=now.year + 1, month=1, day=1)
        else:
            next_month = now.replace(month=now.month + 1, day=1)
        return int(next_month.timestamp() * 1000)
    m = re.fullmatch(r"now-(\d+)([dwm])", t)
    if m:
        amount = int(m.group(1))
        unit = m.group(2)
        days = amount
        if unit == "w":
            days = amount * 7
        elif unit == "m":
            days = amount * 30
        dt = now - timedelta(days=days)
        return int(dt.timestamp() * 1000)
    m = re.fullmatch(r"now\+(\d+)([dwm])", t)
    if m:
        amount = int(m.group(1))
        unit = m.group(2)
        days = amount
        if unit == "w":
            days = amount * 7
        elif unit == "m":
            days = amount * 30
        dt = now + timedelta(days=days)
        return int(dt.timestamp() * 1000)
    return None

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
    def _intent_label(spec: Optional[TaskQuerySpec]) -> str:
        return intent_label(spec)

    @staticmethod
    def _build_intent_answer(
        *,
        spec: TaskQuerySpec,
        rows: List[Dict[str, Any]],
        person: Optional[str],
        task_val: Optional[str],
        person_filters_active: bool,
        person_filter_values: List[str],
        low_conf: bool,
        answer_mode: TaskAnswerMode,
    ) -> Dict[str, Any]:
        handler = get_intent_handler(spec, answer_mode)
        ctx = AnswerContext(
            spec=spec,
            rows=rows,
            person=person,
            task=task_val,
            person_filters_active=person_filters_active,
            person_filter_values=person_filter_values,
            low_conf=low_conf,
            answer_mode=answer_mode,
            format_ts=ts_to_str,
        )
        return handler.build_answer(ctx)

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

        if mode_raw == "text2sql":
            try:
                spec = parse_task_query_nl(q)
            except Exception:
                spec = None
            result = self._answer_via_text2sql(q, spec)
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
        payload["intent"] = self._intent_label(spec)

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
        payload.update(
            self._build_intent_answer(
                spec=spec,
                rows=rows,
                person=person,
                task_val=task_val,
                person_filters_active=person_filters_active,
                person_filter_values=person_filter_values,
                low_conf=low_conf,
                answer_mode=answer_mode,
            )
        )
        return payload

    def _resolve_via_ir_fast_path(self, spec: TaskQuerySpec, routing_debug: Dict[str, Any]) -> Dict[str, Any]:
        debug = dict(routing_debug or {})
        debug["routed_via"] = "ir_fast_path"
        payload: Dict[str, Any] = {
            "intent": self._intent_label(spec),
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

        payload["intent"] = self._intent_label(spec)
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
            person = rec.get("person") or spec.person
            task = rec.get("task") or spec.task
            answer_mode = getattr(spec, "answer_mode", TaskAnswerMode.default)
            payload.update(
                self._build_intent_answer(
                    spec=spec,
                    rows=rows,
                    person=person,
                    task_val=task,
                    person_filters_active=False,
                    person_filter_values=[],
                    low_conf=False,
                    answer_mode=answer_mode,
                )
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
            base.setdefault("intent", self._intent_label(spec))
            base["nl_ir"] = spec.dict()
        else:
            base.setdefault("intent", "unknown")
            base["nl_ir"] = {"error": "missing_spec"}

        if not llm_settings.enabled or llm_settings.provider == "dummy":
            base["error"] = "text2sql_llm_disabled"
            base["answer"] = "Text2SQL pipeline requires a configured LLM provider."
            return base
        runtime = _resolve_text2sql_settings()
        runtime_provider = runtime.get("provider", llm_settings.provider)
        if runtime_provider not in {"ollama", "openai", "dashscope"}:
            base["error"] = "text2sql_llm_provider_unsupported"
            base["answer"] = (
                f"Text2SQL is not yet supported for provider {runtime_provider}."
            )
            return base

        dialect = _resolve_text2sql_dialect()
        prompt = _build_text2sql_prompt(q, spec, dialect=dialect)
        try:
            llm_result, llm_runtime = _call_text2sql_llm(prompt, dialect=dialect)
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

        hint = _make_text2sql_ir_hint(spec)

        for item in llm_result.queries[:TEXT2SQL_MAX_QUERIES]:
            try:
                rewritten_sql = _rewrite_text2sql_query(item.sql, hint, question=q)
                normalized_sql = _normalize_and_validate_text2sql_query(
                    rewritten_sql,
                    dialect=dialect,
                )
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

        if llm_runtime and llm_runtime.get("model"):
            base["text2sql_model"] = llm_runtime["model"]
            base["text2sql_provider"] = llm_runtime.get("provider")

        natural_answer: Optional[str] = None
        if primary_rows:
            try:
                natural_answer = _generate_text2sql_answer(q, primary_rows, llm_runtime)
            except Text2SQLGenerateError as exc:
                base["text2sql_answer_error"] = str(exc)

        if natural_answer:
            base["answer"] = natural_answer
        elif primary_rows:
            base["answer"] = _summarize_text2sql_rows(primary_rows)
        else:
            base["answer"] = "Text2SQL query returned no rows."

        return base


def _build_text2sql_prompt(
    question: str, spec: Optional[TaskQuerySpec], *, dialect: str = "sqlite"
) -> str:
    hint = _make_text2sql_ir_hint(spec)
    hint_json = json.dumps(hint, ensure_ascii=False, indent=2)
    schema = get_tasks_schema_config()
    latest_relation = schema.latest_relation
    history_relation = schema.history_relation
    dialect_norm = (dialect or "sqlite").strip().lower()
    if dialect_norm == "mssql":
        intro = (
            "Generate at most two SQL queries that answer the user's question using the "
            "SQL Server (T-SQL) schema below. SQL requirements:\n"
        )
        limit_rule = "- ALWAYS include a TOP (N) clause (N <= 100 rows). Do not use LIMIT.\n"
        hint_limit_rule = (
            "- Honor hint.limit when present so TOP matches the user's request; otherwise keep TOP <= 100.\n"
        )
        dialect_note = "- Do not use SQLite-only syntax; stick to T-SQL expressions only.\n"
    else:
        intro = (
            "Generate at most two SQL queries that answer the user's question using the "
            "SQLite schema below. SQL requirements:\n"
        )
        limit_rule = "- ALWAYS include a LIMIT clause (<= 100 rows).\n"
        hint_limit_rule = (
            "- Honor hint.limit when present so LIMIT matches the user's request; otherwise keep LIMIT <= 100.\n"
        )
        dialect_note = "- Do not use non-SQLite syntax; stick to SQLite expressions only.\n"

    parts = [
        intro,
        "- Only SELECT statements are allowed.\n",
        f"- Target the {latest_relation} or {history_relation} tables ({latest_relation} contains the latest row per person+task).\n",
        "- Always include an ORDER BY when the user cares about recency.\n",
        limit_rule,
        "- Do not invent tables or columns.\n",
        "- Do not use parameters; embed literal values directly in the SQL.\n",
        "- ?IR hint ???person/task/project ?????????? hint ??????????????????/????\n",
        "- If the IR hint lists multiple persons or tasks, use an IN (...) filter instead of re-parsing the question.\n",
        "- Map IR hint tags to tags LIKE filters; do not treat tag keywords as person/task names.\n",
        "- Translate time_range/due_range/created_range hints into comparisons on ts/due_ts/created_ts respectively, instead of inventing CURRENT_TIMESTAMP math.\n",
        "- Do not compare ts with created_ts/due_ts directly; rely on the IR-provided ranges for each column.\n",
        "- If a time/due/created range hint is missing, either omit that filter or choose an explicit literal (e.g., now-7d); never output placeholders or arbitrary constants.\n",
        hint_limit_rule,
        "- Do not emit parameter placeholders (?) or MySQL-specific functions such as DATE_SUB/CURDATE.\n",
        dialect_note,
        "- Return strictly valid JSON (no comments or explanations outside the JSON block, and ensure all quotes/brackets are balanced).\n",
        "- If you cannot obtain a literal value for a filter, drop that filter instead of inventing placeholders.\n",
        "\n",
        "Return your answer as pure JSON matching this shape (no extra commentary):\n",
        '{"queries":[{"sql":"SELECT ...","description":"short natural language summary"}]}\n',
        "\n",
        "### Database schema\n",
        f"{_text2sql_schema().strip()}\n\n",
        "### Natural language question\n",
        f"{question}\n\n",
        "### IR hint (may contain mistakes, but usually helpful)\n",
        f"{hint_json}\n",
    ]
    return "".join(parts)


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
        "raw_query": getattr(spec, "raw_query", None),
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


def _call_text2sql_llm(
    prompt: str, *, dialect: str = "sqlite"
) -> Tuple[Text2SQLResponseModel, Dict[str, str]]:
    if not llm_settings.enabled or llm_settings.provider == "dummy":
        raise Text2SQLGenerateError("LLM provider is not configured")

    system_prompt = _text2sql_system_prompt(dialect)
    runtime = _resolve_text2sql_settings()
    provider = runtime["provider"]
    if provider == "ollama":
        response = _call_text2sql_via_ollama(
            prompt, runtime["model"], runtime["base_url"], system_prompt
        )
        return response, runtime
    if provider in {"openai", "dashscope"}:
        response = _call_text2sql_via_openai(
            prompt,
            runtime["model"],
            runtime["base_url"],
            runtime["api_key"],
            system_prompt,
        )
        return response, runtime

    raise Text2SQLGenerateError(f"Provider {provider} is not supported for Text2SQL")

def _resolve_text2sql_settings() -> Dict[str, str]:
    provider = llm_settings.text2sql_provider or llm_settings.provider
    if not provider or provider == "dummy":
        provider = llm_settings.provider

    model = llm_settings.text2sql_model or llm_settings.model

    if provider == "ollama":
        base_url = (
            llm_settings.text2sql_ollama_base_url or llm_settings.ollama_base_url
        )
        return {
            "provider": "ollama",
            "model": model,
            "base_url": base_url,
            "api_key": "",
        }

    if provider in {"openai", "dashscope"}:
        base_url = (
            llm_settings.text2sql_openai_base_url or llm_settings.openai_base_url
        )
        api_key = llm_settings.text2sql_api_key or llm_settings.api_key
        return {
            "provider": provider,
            "model": model,
            "base_url": base_url,
            "api_key": api_key,
        }

    return {
        "provider": provider,
        "model": model,
        "base_url": llm_settings.ollama_base_url,
        "api_key": llm_settings.api_key,
    }

def _call_text2sql_via_ollama(
    prompt: str, model: str, base_url: str, system_prompt: str
) -> Text2SQLResponseModel:
    """Call Ollama's /api/chat endpoint for Text2SQL."""

    payload: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {
            "temperature": 0.0,
        },
    }

    url = f"{base_url.rstrip('/')}/api/chat"
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

    def _try_parse_raw_json(text: str) -> Optional[Dict[str, Any]]:
        candidate = (text or "").strip()
        if not candidate.startswith("{") or not candidate.endswith("}"):
            return None
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return None

    parsed: Optional[Dict[str, Any]] = _try_parse_raw_json(content)
    if parsed is None:
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


def _call_text2sql_via_openai(
    prompt: str, model: str, base_url: str, api_key: str, system_prompt: str
) -> Text2SQLResponseModel:
    """Call an OpenAI-compatible chat completion API for Text2SQL."""

    if not api_key:
        raise Text2SQLGenerateError("LLM_API_KEY is required for provider 'openai'")

    payload: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    url = f"{base_url.rstrip('/')}/chat/completions"
    try:
        resp = httpx.post(url, headers=headers, json=payload, timeout=60.0)
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise Text2SQLGenerateError(f"OpenAI-compatible request failed: {exc}") from exc

    response_text = resp.text
    try:
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
    except (ValueError, KeyError, TypeError, IndexError) as exc:
        raise Text2SQLGenerateError(
            "Invalid response format from OpenAI-compatible API", raw_response=response_text
        ) from exc

    parsed: Optional[Dict[str, Any]] = None
    if isinstance(content, dict):
        parsed = content
    elif isinstance(content, str):
        parsed = _try_parse_json_text(content)

    if parsed is None:
        raise Text2SQLGenerateError(
            "LLM output is not valid JSON", raw_response=response_text
        )

    try:
        return Text2SQLResponseModel.parse_obj(parsed)
    except ValidationError as exc:
        raise Text2SQLGenerateError(
            f"LLM JSON does not match expected schema: {exc}", raw_response=response_text
        ) from exc


def _generate_text2sql_answer(
    question: str,
    rows: List[Dict[str, Any]],
    runtime: Optional[Dict[str, str]],
) -> Optional[str]:
    if not rows or not runtime:
        return None
    preview = rows[:TEXT2SQL_ANSWER_MAX_ROWS]
    prompt = _build_text2sql_answer_prompt(question, preview)
    content = _call_text2sql_answer_llm(prompt, runtime)
    return content.strip() if content else None


def _build_text2sql_answer_prompt(
    question: str, rows: List[Dict[str, Any]]
) -> str:
    rows_json = json.dumps(rows, ensure_ascii=False, indent=2)
    return (
        "用户问题：\n"
        f"{question}\n\n"
        "SQL 查询返回的记录（仅展示前几条）：\n"
        f"{rows_json}\n\n"
        "请根据这些记录，给出简洁、准确的中文回答。如果记录为空或缺少答案所需信息，请明确说明未找到匹配的任务或数据不足。"
    )


def _call_text2sql_answer_llm(prompt: str, runtime: Dict[str, str]) -> str:
    provider = runtime.get("provider")
    model = runtime.get("model")
    if not provider or not model:
        raise Text2SQLGenerateError("Text2SQL answer LLM runtime is incomplete")
    if provider == "ollama":
        return _call_text2sql_answer_via_ollama(prompt, runtime)
    if provider in {"openai", "dashscope"}:
        return _call_text2sql_answer_via_openai(prompt, runtime)
    raise Text2SQLGenerateError(f"Unsupported provider for Text2SQL answer: {provider}")


def _call_text2sql_answer_via_ollama(
    prompt: str, runtime: Dict[str, str]
) -> str:
    base_url = runtime.get("base_url") or llm_settings.ollama_base_url
    payload = {
        "model": runtime.get("model"),
        "messages": [
            {"role": "system", "content": TEXT2SQL_ANSWER_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {"temperature": 0.0},
    }
    url = f"{base_url.rstrip('/')}/api/chat"
    try:
        resp = httpx.post(url, json=payload, timeout=60.0)
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise Text2SQLGenerateError(f"Ollama answer request failed: {exc}") from exc
    try:
        data = resp.json()
        return str(data["message"]["content"]).strip()
    except (ValueError, KeyError, TypeError) as exc:
        raise Text2SQLGenerateError("Invalid Ollama answer response") from exc


def _call_text2sql_answer_via_openai(
    prompt: str, runtime: Dict[str, str]
) -> str:
    base_url = runtime.get("base_url") or llm_settings.openai_base_url
    api_key = runtime.get("api_key") or llm_settings.api_key
    if not api_key:
        raise Text2SQLGenerateError("LLM_API_KEY is required for answer generation")
    payload = {
        "model": runtime.get("model"),
        "messages": [
            {"role": "system", "content": TEXT2SQL_ANSWER_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    url = f"{base_url.rstrip('/')}/chat/completions"
    try:
        resp = httpx.post(url, headers=headers, json=payload, timeout=60.0)
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise Text2SQLGenerateError(f"OpenAI-compatible answer request failed: {exc}") from exc
    try:
        data = resp.json()
        return str(data["choices"][0]["message"]["content"]).strip()
    except (ValueError, KeyError, TypeError, IndexError) as exc:
        raise Text2SQLGenerateError("Invalid OpenAI-compatible answer response") from exc


def _try_parse_json_text(text: str) -> Optional[Dict[str, Any]]:
    candidate = (text or "").strip()
    if not candidate:
        return None
    if candidate.startswith("{") and candidate.endswith("}"):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return None
    try:
        json_payload = _extract_json_payload(candidate)
        return json.loads(json_payload)
    except (ValueError, json.JSONDecodeError):
        return None


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
        s = inner.strip()

    def _extract_from(text: str, idx: int) -> Optional[str]:
        depth = 0
        in_string = False
        escape = False
        for pos in range(idx, len(text)):
            ch = text[pos]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[idx : pos + 1].strip()
        return None

    for marker in ('{"queries"', "{"):
        start = s.find(marker)
        if start == -1:
            continue
        result = _extract_from(s, start)
        if result:
            return result.strip()
    raise ValueError("no JSON object found")


def _format_range_literal(value: Optional[Any]) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value).strip()
    if not text:
        return None
    # try to resolve symbolic tokens like now-7d, now, start_of_week, etc.
    resolved = _resolve_symbolic_time(text)
    if resolved is not None:
        return str(int(resolved))
    if re.fullmatch(r"-?\d+(\.\d+)?", text):
        return text
    escaped = text.replace("'", "''")
    return f"'{escaped}'"


def _replace_symbolic_time_literals(sql: str) -> str:
    if not sql:
        return sql

    def _resolve_literal(token: str) -> Optional[int]:
        return _resolve_symbolic_time(token)

    quoted_pattern = re.compile(
        r"\b(ts|created_ts|due_ts)\s*(>=|<=|=)\s*'([^']+)'",
        re.IGNORECASE,
    )

    def _quoted_repl(match: re.Match) -> str:
        column = match.group(1)
        op = match.group(2)
        token = match.group(3).strip().lower()
        resolved = _resolve_literal(token)
        if resolved is None:
            return match.group(0)
        return f"{column} {op} {int(resolved)}"

    sql = quoted_pattern.sub(_quoted_repl, sql)

    bare_pattern = re.compile(
        r"\b(ts|created_ts|due_ts)\s*(>=|<=|=)\s*(now(?:[+-]\d+[dwm])?|start_of_week|end_of_week|start_of_month|end_of_month|next_week|next_month)\b",
        re.IGNORECASE,
    )

    def _bare_repl(match: re.Match) -> str:
        column = match.group(1)
        op = match.group(2)
        token = match.group(3).strip().lower()
        resolved = _resolve_literal(token)
        if resolved is None:
            return match.group(0)
        return f"{column} {op} {int(resolved)}"

    return bare_pattern.sub(_bare_repl, sql)


def _replace_tsql_datetime_math(sql: str) -> str:
    """Replace common T-SQL datetime expressions with epoch-millis literals."""
    if not sql:
        return sql
    now = datetime.now(timezone.utc)
    now_ms = int(now.timestamp() * 1000)

    def _days_ms(days: int) -> int:
        return int((now + timedelta(days=days)).timestamp() * 1000)

    patterns = [
        (re.compile(r"CAST\(\s*GETUTCDATE\(\)\s*AS\s*BIGINT\s*\)\s*\*\s*1000", re.IGNORECASE), now_ms),
        (re.compile(r"CAST\(\s*GETDATE\(\)\s*AS\s*BIGINT\s*\)\s*\*\s*1000", re.IGNORECASE), now_ms),
        (
            re.compile(
                r"CAST\(\s*DATEDIFF\(\s*SECOND\s*,\s*CAST\('1970-01-01'\s+AS\s+DATETIME2\)\s*,\s*CAST\(GETUTCDATE\(\)\s+AS\s+DATETIME2\)\s*\)\s*AS\s*BIGINT\s*\)\s*\*\s*1000",
                re.IGNORECASE,
            ),
            now_ms,
        ),
        (
            re.compile(
                r"CAST\(\s*DATEDIFF\(\s*SECOND\s*,\s*CAST\('1970-01-01'\s+AS\s+DATETIME2\)\s*,\s*CAST\(GETDATE\(\)\s+AS\s+DATETIME2\)\s*\)\s*AS\s*BIGINT\s*\)\s*\*\s*1000",
                re.IGNORECASE,
            ),
            now_ms,
        ),
    ]

    def _dateadd_repl(match: re.Match) -> str:
        days = int(match.group(1))
        return str(_days_ms(days))

    dateadd_patterns = [
        re.compile(
            r"CAST\(\s*DATEADD\(\s*DAY\s*,\s*([+-]?\d+)\s*,\s*GETUTCDATE\(\)\s*\)\s*AS\s*BIGINT\s*\)\s*\*\s*1000",
            re.IGNORECASE,
        ),
        re.compile(
            r"CAST\(\s*DATEADD\(\s*DAY\s*,\s*([+-]?\d+)\s*,\s*GETDATE\(\)\s*\)\s*AS\s*BIGINT\s*\)\s*\*\s*1000",
            re.IGNORECASE,
        ),
    ]

    updated = sql
    for pattern, value in patterns:
        updated = pattern.sub(str(value), updated)
    for pattern in dateadd_patterns:
        updated = pattern.sub(_dateadd_repl, updated)
    return updated


def _build_range_clause(column: str, start_literal: Optional[str], end_literal: Optional[str]) -> str:
    parts: List[str] = []
    if start_literal:
        parts.append(f"{column} >= {start_literal}")
    if end_literal:
        parts.append(f"{column} <= {end_literal}")
    return " AND ".join(parts)


def _apply_range_hint(sql: str, column: str, range_hint: Optional[Dict[str, Any]]) -> str:
    if not sql:
        return sql
    hint = range_hint or {}
    start_literal = _format_range_literal(hint.get("start"))
    end_literal = _format_range_literal(hint.get("end"))
    alias_pattern = re.compile(rf"\b[a-zA-Z0-9_]+\.(?={re.escape(column)}\b)", re.IGNORECASE)
    sql = alias_pattern.sub("", sql)

    def _replacement_clause(default: str) -> str:
        clause = _build_range_clause(column, start_literal, end_literal)
        return clause or default

    between_pattern = re.compile(
        rf"{column}\s*(?:>=|>)\s*(\?|\([^)]*now\(\)[^)]*\)|\d+)\s+and\s*{column}\s*(?:<=|<)\s*(\?|\([^)]*now\(\)[^)]*\)|\d+)",
        re.IGNORECASE,
    )
    sql = between_pattern.sub(lambda _: _replacement_clause("1=1"), sql)

    placeholder_pattern = re.compile(
        rf"{column}\s*(?:>=|>|<=|<)\s*(\?|\([^)]*now\(\)[^)]*\))",
        re.IGNORECASE,
    )
    sql = placeholder_pattern.sub(lambda match: _replacement_clause("1=1"), sql)

    interval_pattern = re.compile(
        rf"{column}\s*(?:>=|>|<=|<)\s*now\(\)\s*-\s*\d+\s*[a-z]+",
        re.IGNORECASE,
    )
    sql = interval_pattern.sub(_replacement_clause("1=1"), sql)

    symbolic_pattern = re.compile(
        rf"{column}\s*(?:>=|>|<=|<)\s*'[^']*'",
        re.IGNORECASE,
    )
    sql = symbolic_pattern.sub(_replacement_clause("1=1"), sql)

    literal_pattern = re.compile(
        rf"{column}\s*(>=|>|<=|<)\s*(\d+)",
        re.IGNORECASE,
    )

    def _literal_repl(match: re.Match) -> str:
        op = match.group(1)
        value = match.group(2)
        try:
            num = int(value)
        except ValueError:
            return match.group(0)
        if num >= TEXT2SQL_SUSPICIOUS_LITERAL_THRESHOLD:
            return match.group(0)
        if op in (">=", ">") and start_literal:
            return f"{column} {op} {start_literal}"
        if op in ("<=", "<") and end_literal:
            return f"{column} {op} {end_literal}"
        return "1=1"

    sql = literal_pattern.sub(_literal_repl, sql)
    return sql


def _ensure_tag_filters(sql: str, tags: Optional[List[str]], *, column: str = "tags") -> str:
    def _clean(tag: str) -> str:
        cleaned = str(tag or "").strip()
        cleaned = cleaned.strip(" ，,;；、")
        cleaned = re.sub(r"\s+", "", cleaned)
        return cleaned

    tag_values = [_clean(tag) for tag in (tags or []) if _clean(tag)]
    if not tag_values:
        return sql
    lowered = sql.lower()
    col = (column or "tags").strip()
    if not col:
        return sql
    col_norm = col.lower()
    if re.search(rf"\b{re.escape(col_norm)}\s+(?:like|=|in)\b", lowered):
        return sql

    def _escape(tag: str) -> str:
        return tag.replace("'", "''")
    seen: List[str] = []
    for tag in tag_values:
        if tag not in seen:
            seen.append(tag)
        if len(seen) >= 2:
            break
    clause_parts = [f"{col} LIKE '%{_escape(tag)}%'" for tag in seen]
    clause = " AND ".join(clause_parts)
    where_idx = lowered.find(" where ")
    if where_idx != -1:
        insert_pos = where_idx + len(" where ")
        end_idx = len(sql)
        for keyword in (" order by ", " limit "):
            idx = lowered.find(keyword, insert_pos)
            if idx != -1 and idx < end_idx:
                end_idx = idx
        existing = sql[insert_pos:end_idx].strip()
        suffix = sql[end_idx:]
        if existing:
            sql = f"{sql[:insert_pos]}({clause}) AND ({existing}){suffix}"
        else:
            sql = f"{sql[:insert_pos]}({clause}){suffix}"
        return sql

    order_idx = lowered.find(" order by ")
    limit_idx = lowered.find(" limit ")
    insert_pos = len(sql)
    for idx in (order_idx, limit_idx):
        if idx != -1 and idx < insert_pos:
            insert_pos = idx
    suffix = sql[insert_pos:]
    prefix = sql[:insert_pos]
    if suffix.strip():
        sql = f"{prefix} WHERE ({clause}) {suffix.lstrip()}"
    else:
        sql = f"{prefix} WHERE ({clause})"
    return sql


def _cleanup_invalid_order_tokens(sql: str) -> str:
    if not sql:
        return sql
    pattern = re.compile(r"(\b(?:and|where)\b)\s+[A-Za-z_][\w.]*\s+(?:asc|desc)\b", re.IGNORECASE)
    def _repl(match: re.Match) -> str:
        lead = match.group(1)
        return " " if lead.lower() == "and" else f"{lead} "
    return pattern.sub(_repl, sql)


def _strip_semicolons(sql: str) -> str:
    if not sql:
        return sql
    return sql.replace(";", " ")


def _parse_sql_expression(sql: str, dialect: str = "sqlite") -> exp.Expression:
    try:
        return parse_one(sql, read=dialect)
    except Exception as exc:  # pragma: no cover - defensive
        raise Text2SQLValidationError(f"Failed to parse SQL: {exc}") from exc


def _unwrap_select_expression(expression: exp.Expression) -> exp.Expression:
    if isinstance(expression, exp.With):
        return _unwrap_select_expression(expression.this)
    if isinstance(expression, exp.Subquery):
        return _unwrap_select_expression(expression.this)
    paren_cls = getattr(exp, "Paren", None)
    if paren_cls is not None and isinstance(expression, paren_cls):
        return _unwrap_select_expression(expression.this)
    return expression


def _validate_select_expression(expr: exp.Expression) -> exp.Query:
    target = _unwrap_select_expression(expr)
    if not isinstance(target, exp.Query):
        raise Text2SQLValidationError("Only SELECT statements are permitted.")
    return target


def _validate_allowed_tables(expression: exp.Expression) -> None:
    allowed = _allowed_tables()
    for table in expression.find_all(exp.Table):
        name = table.name
        if not name:
            raise Text2SQLValidationError("Query references an unnamed table.")
        normalized = _normalize_relation_name(name)
        if normalized not in allowed:
            raise Text2SQLValidationError(
                f"Query references table '{name}' which is not allowed."
            )


def _ensure_limit_ast(select_expr: exp.Query, *, max_rows: int = 100) -> None:
    limit = select_expr.args.get("limit")
    capped_literal = exp.Literal.number(max_rows)
    if limit is None:
        select_expr.set("limit", exp.Limit(expression=capped_literal))
        return
    literal = getattr(limit, "expression", None)
    if isinstance(literal, exp.Literal) and literal.is_number:
        try:
            value = int(literal.name)
        except (TypeError, ValueError):
            limit.set("expression", capped_literal)
            return
        if value > max_rows:
            limit.set("expression", capped_literal)
    else:
        limit.set("expression", capped_literal)


def _rewrite_text2sql_query(sql: str, hint: Optional[Dict[str, Any]], *, question: str = "") -> str:
    if not sql:
        return sql
    updated = sql
    hint = hint or {}

    # Align time-related literals with IR hint (ts/created_ts/due_ts).
    schema = get_tasks_schema_config()
    ts_col = schema.translate_field("ts") or "ts"
    created_col = schema.translate_field("created_ts") or "created_ts"
    due_col = schema.translate_field("due_ts") or "due_ts"
    tags_col = schema.translate_field("tags") or "tags"
    person_col = schema.translate_field("person") or "person"
    project_col = schema.translate_field("project") or "project"

    updated = _apply_range_hint(updated, ts_col, hint.get("time_range"))
    updated = _apply_range_hint(updated, created_col, hint.get("created_range"))
    updated = _apply_range_hint(updated, due_col, hint.get("due_range"))

    # Ensure tag filters reflect IR/KG tags when LLM forgot to use them.
    updated = _ensure_tag_filters(updated, hint.get("tags"), column=tags_col)

    # Normalize person/project literals to canonical values from IR/KG.
    updated = _apply_scalar_hint(updated, person_col, hint.get("person"))
    updated = _apply_scalar_hint(updated, project_col, hint.get("project"))

    domain = get_tasks_domain()
    updated = domain.rewrite_text2sql(
        updated,
        hint=hint,
        question=question or "",
        schema=schema,
    )

    # Cleanup and safety-normalization.
    updated = _cleanup_invalid_order_tokens(updated)
    updated = _strip_semicolons(updated)
    return updated


def _normalize_and_validate_text2sql_query(sql: str, *, dialect: str = "sqlite") -> str:
    if not sql or not sql.strip():
        raise Text2SQLValidationError("SQL query is empty.")
    normalized = sql.strip().rstrip(";").strip()
    normalized = _replace_symbolic_time_literals(normalized)
    normalized = _replace_tsql_datetime_math(normalized)
    dialect_norm = (dialect or "sqlite").strip().lower()
    sqlglot_dialect = _sqlglot_dialect(dialect_norm)
    ast = _parse_sql_expression(normalized, dialect=sqlglot_dialect)
    _validate_allowed_tables(ast)
    target = _validate_select_expression(ast)
    _ensure_limit_ast(target, max_rows=100)
    normalized = ast.sql(dialect=sqlglot_dialect)
    if dialect_norm == "mssql":
        normalized = _tsql_prefix_unicode_literals(normalized)
    lowered = normalized.lower()
    schema = get_tasks_schema_config()
    if "?" in normalized:
        raise Text2SQLValidationError("Parameter placeholders are not allowed.")
    if re.search(r":(?!/)[A-Za-z_]\w*", normalized):
        raise Text2SQLValidationError("Named parameter placeholders are not allowed.")
    for func in TEXT2SQL_DISALLOWED_FUNCTIONS:
        if re.search(rf"\b{re.escape(func)}\b", lowered):
            raise Text2SQLValidationError(f"Unsupported SQL function detected: {func}")
    for comparison in _text2sql_disallowed_comparisons(schema):
        if comparison in lowered:
            raise Text2SQLValidationError(f"Unsupported column comparison detected: {comparison}")
    literal_cols = _text2sql_literal_columns(schema)
    if literal_cols:
        col_pattern = "|".join(re.escape(col) for col in literal_cols)
        literal_pattern = re.compile(
            rf"(?<!\w)({col_pattern})(?!\w)\s*(>=|<=|=)\s*(\d+)",
            re.IGNORECASE,
        )
        for column, _, value in literal_pattern.findall(normalized):
            try:
                literal = int(value)
            except ValueError:
                continue
            if literal < TEXT2SQL_SUSPICIOUS_LITERAL_THRESHOLD:
                raise Text2SQLValidationError(
                    f"Suspicious literal detected for {column}: {value}"
                )
    for keyword in TEXT2SQL_FORBIDDEN_KEYWORDS:
        if re.search(rf"\b{re.escape(keyword)}\b", lowered):
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


def _apply_scalar_hint(sql: str, column: str, value: Optional[Any]) -> str:
    """Rewrite simple column literals (person/project) to canonical hint value.

    This is intentionally conservative and only touches common patterns like:
      - column = '...'
      - column IN ('a', 'b', ...)
    so that Text2SQL output is nudged towards IR/KG without heavy AST surgery.
    """
    if not sql or value in (None, ""):
        return sql

    val = str(value)
    # basic SQL string escaping
    val_escaped = val.replace("'", "''")

    lowered = sql.lower()
    if column.lower() not in lowered:
        return sql

    # Pattern 1: column = '...'
    pattern_eq = re.compile(
        rf"(\b{re.escape(column)}\b\s*=\s*)'[^']*'", re.IGNORECASE
    )
    if pattern_eq.search(sql):
        return pattern_eq.sub(rf"\1'{val_escaped}'", sql)

    # Pattern 2: column IN ('a', 'b', ...)
    pattern_in = re.compile(
        rf"(\b{re.escape(column)}\b\s+in\s*\()([^)]+)(\))", re.IGNORECASE
    )

    def _repl_in(match: re.Match) -> str:
        prefix = match.group(1)
        suffix = match.group(3)
        return f"{prefix}'{val_escaped}'{suffix}"

    return pattern_in.sub(_repl_in, sql)
