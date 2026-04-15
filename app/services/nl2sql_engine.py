from __future__ import annotations

from enum import Enum
from os import getenv
from typing import Any, Dict, List, Optional, Set
import logging
import re

from pydantic import BaseModel, Field

from app.services.llm_client import get_llm_client
from app.services import kg_lite
from app.tasks_schema import get_tasks_schema_config


class TaskQueryIntent(str, Enum):
    """Semantic intent type for task queries.

    The enum is intentionally fine-grained so that the NL parser / LLM
    can choose the most appropriate shape; downstream components may map
    several of these into coarser buckets (e.g. "status_query").
    """

    # Single latest status for one (person, task) pair.
    task_status_single = "task_status_single"
    # Multiple status rows, typically for one task or person.
    task_status_list = "task_status_list"
    # All tasks for a given person (optionally filtered by status).
    task_list_by_person = "task_list_by_person"
    # Status history for a specific (person, task) pair.
    task_history = "task_history"
    # Aggregate view for a person (e.g. counts by status).
    person_summary = "person_summary"
    # Fallback when intent cannot be reliably determined.
    unknown = "unknown"


class TaskAnswerMode(str, Enum):
    """Controls how TaskQueryEngine should format the final answer."""

    default = "default"
    completion_time_latest = "completion_time_latest"
    task_count_by_status = "task_count_by_status"
    person_summary_by_project = "person_summary_by_project"
    overdue_count_by_person = "overdue_count_by_person"


class TaskStatus(str, Enum):
    """Task status enum corresponding to tasks.status."""

    DONE = "DONE"
    TODO = "TODO"
    IN_PROGRESS = "IN_PROGRESS"
    BLOCKED = "BLOCKED"
    ANY = "ANY"  # no restriction (only used in IR)


class OrderByDirection(str, Enum):
    asc = "asc"
    desc = "desc"


class TimeRange(BaseModel):
    """Time range (kept simple for now)."""

    start: Optional[str] = Field(
        None,
        description="Start time (ISO-8601 or natural language fragment), inclusive.",
    )
    end: Optional[str] = Field(
        None,
        description="End time (ISO-8601 or natural language fragment), inclusive.",
    )


class OrderBySpec(BaseModel):
    """Order-by field and direction for later NL→SQL mapping."""

    field: str = Field(..., description="Order field, e.g. ts, id, person.")
    direction: OrderByDirection = Field(
        OrderByDirection.desc, description="Order direction (default desc)."
    )


_SQL_PARAM_SCALAR_TYPES = (str, int, float, bytes, bool)
_SAFE_FIELD_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ALLOWED_FILTER_OPS = {"eq", "in", "like", "gte", "lte", "between"}
_PRIORITY_TOKEN_RE = re.compile(r"^P(?P<num>[1-9][0-9]*)$", re.IGNORECASE)


def _coerce_sql_param_scalar(value: Any) -> Any:
    """Coerce a value into a safe SQLite bind parameter (or None if impossible).

    This prevents sqlite3 binding failures when upstream structured outputs
    accidentally emit dict/list payloads (e.g., {"tag": "..."}).
    """

    if value is None:
        return None
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, _SQL_PARAM_SCALAR_TYPES):
        return value
    if isinstance(value, dict):
        for key in ("tag", "value", "name", "id", "code"):
            if key in value:
                coerced = _coerce_sql_param_scalar(value.get(key))
                if coerced is not None:
                    return coerced
        if len(value) == 1:
            coerced = _coerce_sql_param_scalar(next(iter(value.values())))
            if coerced is not None:
                return coerced
        return None
    return None


def _validate_filter_field(field: Any) -> str:
    name = str(field or "").strip()
    if not name or not _SAFE_FIELD_RE.match(name):
        raise ValueError(f"unsafe filter field: {field!r}")
    return name


def _normalize_filter_op(op: Any) -> str:
    normalized = str(op or "eq").lower().strip()
    if normalized not in _ALLOWED_FILTER_OPS:
        return "eq"
    return normalized


def _coerce_priority_value(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.isdigit():
            return int(text)
        match = _PRIORITY_TOKEN_RE.match(text)
        if match:
            return int(match.group("num"))
        return None
    if isinstance(value, dict):
        for key in value.keys():
            key_text = str(key or "").strip()
            match = _PRIORITY_TOKEN_RE.match(key_text)
            if match:
                return int(match.group("num"))
        for key in ("priority", "p", "level"):
            if key in value:
                return _coerce_priority_value(value.get(key))
        return None
    return None


class QueryFilter(BaseModel):
    """Flexible filter definition for advanced queries."""

    field: str = Field(..., description="Column name or logical field.")
    op: str = Field("eq", description="Operator such as eq, in, like, gte.")
    value: Optional[Any] = Field(
        None, description="Single value used by most operators."
    )
    values: Optional[List[Any]] = Field(
        None, description="Multi-value payload for IN/BETWEEN clauses."
    )

    def to_plan_filter(self) -> Dict[str, Any]:
        field = _validate_filter_field(self.field)
        op = _normalize_filter_op(self.op)
        payload: Dict[str, Any] = {"field": field, "op": op}

        if op == "in":
            raw_vals: List[Any] = []
            if self.values:
                raw_vals = list(self.values)
            elif self.value is not None:
                raw_vals = [self.value]
            vals: List[Any] = []
            for item in raw_vals:
                if field == "priority":
                    coerced = _coerce_priority_value(item)
                else:
                    coerced = _coerce_sql_param_scalar(item)
                if coerced is not None:
                    vals.append(coerced)
            if not vals:
                raise ValueError("empty/invalid IN values")
            payload["value"] = vals
            return payload

        if op == "between":
            raw_vals = list(self.values or [])
            if len(raw_vals) < 2 and self.value is not None:
                raw_vals.append(self.value)
            while len(raw_vals) < 2:
                raw_vals.append(None)
            start = _coerce_sql_param_scalar(raw_vals[0])
            end = _coerce_sql_param_scalar(raw_vals[1])
            if start is None and end is None and self.value is not None:
                raise ValueError("invalid BETWEEN values")
            payload["value"] = [start, end]
            return payload

        if field == "priority":
            coerced = _coerce_priority_value(self.value)
        else:
            coerced = _coerce_sql_param_scalar(self.value)
        if coerced is None and self.value is not None:
            raise ValueError("invalid scalar value")
        payload["value"] = coerced
        return payload


class TaskQuerySpec(BaseModel):
    """Intermediate representation for task query semantics."""

    intent: TaskQueryIntent = Field(
        TaskQueryIntent.task_status_single, description="Recognized query intent."
    )
    raw_query: str = Field(..., description="Original natural language query.")
    is_supported: Optional[bool] = Field(
        None,
        description="LLM hint describing whether the IR fast path should handle this query.",
    )
    intent_confidence: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="LLM confidence score (0~1) for the predicted intent.",
    )
    raw_intent_nl: Optional[str] = Field(
        None,
        description="Free-form natural language summary of the detected intent (LLM provided).",
    )
    answer_mode: TaskAnswerMode = Field(
        TaskAnswerMode.default, description="Optional hint for downstream answer formatter."
    )

    person: Optional[str] = Field(
        None, description="Person involved in the task, if recognized."
    )
    task: Optional[str] = Field(
        None, description="Task name (best guess), if recognized."
    )
    task_keywords: List[str] = Field(
        default_factory=list,
        description="Keywords extracted from task description for fuzzy matching.",
    )
    project: Optional[str] = Field(
        None, description="Project identifier/name for filtering."
    )
    tags: List[str] = Field(
        default_factory=list,
        description="Tags/labels from task description for filtering.",
    )
    priority: Optional[int] = Field(
        None, description="Priority (1 highest)."
    )
    status: List[TaskStatus] = Field(
        default_factory=list,
        description="Required task status filters; empty means no restriction.",
    )

    time_range: Optional[TimeRange] = Field(
        None, description="Time range for the query, e.g. recent week/month (status timestamp)."
    )
    due_range: Optional[TimeRange] = Field(
        None, description="Due time range filter (epoch ms or ISO)."
    )
    created_range: Optional[TimeRange] = Field(
        None, description="Created time range filter (epoch ms or ISO)."
    )
    order_by: List[OrderBySpec] = Field(
        default_factory=list, description="Order-by fields."
    )
    limit: Optional[int] = Field(
        10, ge=1, le=200, description="Max number of rows to return (default 10)."
    )
    filters: List[QueryFilter] = Field(
        default_factory=list,
        description="Additional filters that cannot be expressed via dedicated fields.",
    )

    extra: Dict[str, Any] = Field(
        default_factory=dict,
        description="Reserved for debug info, model scores, parse details, etc.",
    )


def is_complex_by_text(raw_query: Optional[str]) -> bool:
    """Simple keyword sniffing to flag obviously complex/BI-style questions."""

    text = (raw_query or "").strip()
    if not text:
        return False
    lower = text.lower()
    for kw in _COMPLEX_KEYWORDS:
        if not kw:
            continue
        if kw in text or kw.lower() in lower:
            return True
    return False


def _collect_filter_values(spec: TaskQuerySpec, field_name: str) -> List[str]:
    values: List[str] = []
    filters = getattr(spec, "filters", None) or []
    normalized = field_name.lower()
    for flt in filters:
        if not isinstance(flt, QueryFilter):
            continue
        raw_field = str(getattr(flt, "field", "") or "").lower()
        if raw_field != normalized:
            continue
        op = str(getattr(flt, "op", "eq") or "eq").lower()
        if op == "in":
            for val in getattr(flt, "values", None) or []:
                if val not in (None, ""):
                    values.append(str(val))
        else:
            val = getattr(flt, "value", None)
            if val not in (None, ""):
                values.append(str(val))
    return values


def too_many_entities(spec: TaskQuerySpec) -> bool:
    """Heuristic guardrail: multi-person/task/project/tag queries => complex."""

    def _count(values: List[str], extra: Optional[str]) -> int:
        uniq: Set[str] = {v for v in values if v}
        if extra:
            uniq.add(extra)
        return len(uniq)

    person_values = _collect_filter_values(spec, "person")
    task_values = _collect_filter_values(spec, "task")
    project_values = _collect_filter_values(spec, "project")

    if _count(person_values, getattr(spec, "person", None)) >= 2:
        return True
    if _count(task_values, getattr(spec, "task", None)) >= 2:
        return True
    if _count(project_values, getattr(spec, "project", None)) >= 2:
        return True

    status_list = [s for s in getattr(spec, "status", []) or []]
    if len(status_list) >= 3:
        return True

    tags = getattr(spec, "tags", []) or []
    if len(tags) >= 3:
        return True

    return False


def is_simple_intent(spec: Optional[TaskQuerySpec]) -> bool:
    """Return True only when the IR fast path should own the query.

    Phase 2 whitelist strategy:
    - Only allow very simple, well‑specified single‑task intents:
      * task_status_single  (person + task, latest status)
      * task_history        (person + task, status history / completion time)
    - All list / aggregation / multi‑entity queries are considered too
      complex or fragile for the IR fast path and should be delegated to
      downstream resolvers / Text2SQL.
    """

    if spec is None:
        return False

    if is_complex_by_text(getattr(spec, "raw_query", "")):
        return False

    if getattr(spec, "is_supported", None) is False:
        return False

    raw_intent = getattr(spec, "intent", TaskQueryIntent.unknown)
    if isinstance(raw_intent, TaskQueryIntent):
        intent = raw_intent
    else:
        try:
            intent = TaskQueryIntent(str(raw_intent))
        except Exception:
            intent = TaskQueryIntent.unknown
    # Whitelist: only single‑task intents go through fast path.
    allowed_intents = {TaskQueryIntent.task_status_single, TaskQueryIntent.task_history}
    if intent not in allowed_intents:
        return False

    def _has_single_entity(field: str) -> bool:
        value = getattr(spec, field, None)
        if value:
            return True
        filter_values = [
            v for v in _collect_filter_values(spec, field) if v not in (None, "")
        ]
        if len(filter_values) == 1:
            return True
        extra = getattr(spec, "extra", {}) or {}
        token_key = f"rule_{field}_tokens"
        tokens = extra.get(token_key)
        if isinstance(tokens, list):
            token_values = [t for t in tokens if t]
            if len(token_values) == 1:
                return True
        return False

    if intent in allowed_intents:
        if not _has_single_entity("person"):
            return False
        if not _has_single_entity("task"):
            return False
        # For single-task intents, we deliberately ignore too_many_entities()
        # and only rely on intent + single person/task checks.
        confidence = getattr(spec, "intent_confidence", None)
        if confidence is not None and confidence < 0.5:
            return False
        return True


_USE_LLM_FOR_NL2SQL = getenv("TASKS_NL2SQL_LLM", "0") == "1"
_TASK_STATUS_VALUES = {
    item.strip().upper()
    for item in getenv("TASKS_STATUS_VALUES", "DONE,TODO").split(",")
    if item.strip()
}

_PERSON_SEPARATOR_CHARS = "、，,/和及与"
_MULTI_ENTITY_SPLIT_RE = re.compile(f"[{_PERSON_SEPARATOR_CHARS}\+\s]+")
_PERSON_SUFFIXES = (
    "的任务",
    "的情况",
    "的状态",
    "的项目",
    "任务",
    "情况",
    "状态",
    "列表",
    "有哪些",
    "还有",
    "最近",
    "里",
    "在",
    "的",
)
_PERSON_STOPWORDS = (
    "\u5f53\u524d",  # 当前
    "\u6240\u6709",  # 所有
    "\u6700\u8fd1",  # 最近
    "\u672c\u5468",  # 本周
    "\u672c\u6708",  # 本月
    "\u8fd8\u6709",  # 还有
    "\u54ea\u4e9b",  # 哪些
    "\u4ec0\u4e48",  # 什么
    "\u60c5\u51b5",  # 情况
    "\u4efb\u52a1",  # 任务
    "\u72b6\u6001",  # 状态
    "\u5217\u8868",  # 列表
    "\u5168\u90e8",  # 全部
    "\u5176\u4ed6",  # 其他
    "\u6807\u7b7e",  # 标签
    "\u6807\u7b7e\u5305\u542b",  # 标签包含
    "\u6807\u7b7e\u91cc",  # 标签里
    "\u6807\u7b7e\u4e2d",  # 标签中
    "\u6807\u7b7e\u662f",  # 标签是
    "\u9879\u76ee",  # 项目
    "\u9879\u76ee\u91cc",  # 项目里
    "\u9879\u76ee\u4e2d",  # 项目中
    "\u9879\u76ee\u7684",  # 项目的
)

_TIME_RANGE_HINTS = [
    ("最近一周", ("now-7d", "now")),
    ("最近7天", ("now-7d", "now")),
    ("最近一个月", ("now-30d", "now")),
    ("最近30天", ("now-30d", "now")),
    ("最近三个月", ("now-90d", "now")),
    ("最近90天", ("now-90d", "now")),
    ("本周", ("start_of_week", "end_of_week")),
    ("本月", ("start_of_month", "end_of_month")),
]

_RECENT_RANGE_PATTERN = re.compile(r"(?:最近|过去)(\d{1,2})(天|日|周|月)")

_PROJECT_PATTERNS = [
    re.compile(r"(?P<name>[A-Za-z0-9\u4e00-\u9fff_-]+)\s*项目"),
    re.compile(r"项目\s*(?P<name>[A-Za-z0-9\u4e00-\u9fff_-]+)"),
]

_TAG_PATTERN = re.compile(r"#([\w\u4e00-\u9fff-]+)")
_TAG_PHRASE_PATTERN = re.compile(
    r"\u6807\u7b7e(?:\u5305\u542b|\u6709|\u5e26|\u542b|\u662f)?\s*([#\w\u4e00-\u9fff-]+)"
)

_LIMIT_PATTERN = re.compile(r"(?:前|最近)?(\d{1,3})(?:条|个|项|行)")

_PRIORITY_HINTS = [
    ("p1", 1),
    ("p2", 2),
    ("p3", 3),
    ("\u6700\u9ad8\u4f18\u5148\u7ea7", 1),  # 最高优先级
    ("\u9ad8\u4f18\u5148\u7ea7", 1),        # 高优先级
    ("\u9ad8\u4f18p1", 1),                 # 高优P1
    ("\u9ad8\u4f18", 1),                   # 高优
]

_ORDER_ASC_HINTS = ("最早", "时间升序", "按创建顺序")
_ORDER_DESC_HINTS = ("最新", "最近", "按更新时间", "按优先级")
_DUE_TIME_HINTS = ("截止", "到期", "ddl", "DDL", "计划完成", "期限")
_STATUS_TIME_HINTS = (
    "更新时间",
    "更新于",
    "最近更新",
    "状态时间",
    "完成时间",
    "什么时候完成",
    "何时完成",
)
_TASK_QUESTION_FRAGMENTS = (
    "任务有哪些",
    "有哪些任务",
    "哪些任务",
    "任务列表",
    "任务清单",
    "所有任务",
    "全部任务",
    "多少任务",
    "几个任务",
    "什么任务",
)
_PERSON_CLAUSE_MARKERS = (
    "还剩多少",
    "剩余多少",
    "剩下多少",
    "还有多少",
    "还剩",
    "剩余",
    "剩下",
    "都完成了哪些",
    "完成了哪些",
    "都有哪些",
    "有哪些",
    "哪些",
    "最近",
    "本周",
    "本月",
    "截止",
    "到期",
    "高优",
    "P1",
    "P2",
    "P3",
    "都完成了",
    "完成了",
    "都负责",
    "负责",
    "参与",
)
_PERSON_PREFIXES = ("列出", "查看", "查询", "给我看", "看一下", "统计")
_REMAINING_COUNT_HINTS = (
    "还剩多少任务",
    "剩余多少任务",
    "剩下多少任务",
    "还有多少任务",
    "多少任务未完成",
    "未完成多少任务",
)

_COMPLEX_KEYWORDS = [
    "\u7edf\u8ba1",
    "\u6bd4\u4f8b",
    "\u5360\u6bd4",
    "\u6548\u7387",
    "\u6700\u5fd9",
    "\u6700\u6162",
    "\u903e\u671f\u7387",
    "\u5e73\u5747",
    "\u6392\u540d",
    "\u5bf9\u6bd4",
    "\u5404\u90e8\u95e8",
    "\u6bcf\u4e2a\u90e8\u95e8",
    "\u6309\u9879\u76ee",
    "\u603b\u4f53\u60c5\u51b5",
    "\u6982\u51b5",
    "\u5206\u5e03",
    "ratio",
    "percent",
    "compare",
]

_PERSON_TASK_STATUS_RE = re.compile(
    r"(?P<person>[\u4e00-\u9fffA-Za-z0-9_]+)的(?P<task>.+?)(?:现在|目前)?(?:什么状态|状况|进度|进展)"
)
_PERSON_TASK_GENERIC_RE = re.compile(r"(?P<person>[\u4e00-\u9fffA-Za-z0-9_]+)的(?P<task>.+)")

_COMPLETION_TIME_HINTS = (
    "\u4ec0\u4e48\u65f6\u5019\u5b8c\u6210",  # 什么时候完成
    "\u4f55\u65f6\u5b8c\u6210",  # 何时完成
    "\u4ec0\u4e48\u65f6\u5019\u641e\u5b9a",  # 什么时候搞定
    "\u5b8c\u6210\u65f6\u95f4",  # 完成时间
    "\u4ec0\u4e48\u65f6\u5019\u7ed3\u675f",  # 什么时候结束
    "\u4f55\u65f6\u641e\u5b9a",  # 何时搞定
)






def _split_multi_values(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in _MULTI_ENTITY_SPLIT_RE.split(value) if item.strip()]


def _normalize_person_candidate(value: Optional[str]) -> str:
    token = (value or "").strip()
    if not token:
        return ""
    changed = True
    while changed:
        changed = False
        for suffix in _PERSON_SUFFIXES:
            if token.endswith(suffix):
                token = token[: -len(suffix)].strip() if len(token) > len(suffix) else ""
                changed = True
                break
    return token


def _person_has_stopword(token: str) -> bool:
    return any(stop in token for stop in _PERSON_STOPWORDS)


def _trim_person_clause(token: str) -> str:
    candidate = (token or "").strip()
    if not candidate:
        return ""
    for prefix in _PERSON_PREFIXES:
        if candidate.startswith(prefix) and len(candidate) > len(prefix):
            candidate = candidate[len(prefix) :].strip()
            break
    for marker in _PERSON_CLAUSE_MARKERS:
        if marker in candidate:
            candidate = candidate.split(marker, 1)[0].strip()
    return candidate


def _sanitize_person_value(value: Optional[str]) -> Optional[str]:
    token = _normalize_person_candidate(_trim_person_clause(value or ""))
    if not token:
        return None
    if _person_has_stopword(token):
        return None
    return token


def _extract_keywords(value: Optional[str]) -> List[str]:
    if not value:
        return []
    tokens = re.split(r"[\s,，/()（）:\-]+", value)
    return [tok.strip() for tok in tokens if tok.strip()]


def _detect_time_range(text: str) -> Optional[TimeRange]:
    for hint, (start, end) in _TIME_RANGE_HINTS:
        if hint in text:
            return TimeRange(start=start, end=end)
    m = _RECENT_RANGE_PATTERN.search(text)
    if m:
        amount = int(m.group(1))
        unit = m.group(2)
        if unit in ("天", "日"):
            start = f"now-{amount}d"
        elif unit == "周":
            start = f"now-{amount * 7}d"
        else:
            start = f"now-{amount * 30}d"
        return TimeRange(start=start, end="now")
    return None



def _detect_created_range(text: str) -> Optional[TimeRange]:
    lower = text.lower()
    if (
        "\u521b\u5efa" not in text
        and "\u65b0\u5efa" not in text
        and "create" not in lower
        and "created" not in lower
    ):
        return None
    return _detect_time_range(text)


def _detect_due_range(text: str) -> Optional[TimeRange]:
    due_kws = ("截止", "到期", "ddl")
    if not any(kw in text for kw in due_kws):
        return None
    if "本周" in text:
        return TimeRange(start="start_of_week", end="end_of_week")
    if "本月" in text:
        return TimeRange(start="start_of_month", end="end_of_month")
    m = _RECENT_RANGE_PATTERN.search(text)
    if m:
        amount = int(m.group(1))
        unit = m.group(2)
        if unit in ("天", "日"):
            start = f"now-{amount}d"
        elif unit == "周":
            start = f"now-{amount * 7}d"
        else:
            start = f"now-{amount * 30}d"
        return TimeRange(start=start, end="now")
    return None


def _mentions_due_time(text: str) -> bool:
    return any(kw in (text or "") for kw in _DUE_TIME_HINTS)


def _mentions_status_time(text: str) -> bool:
    return any(kw in (text or "") for kw in _STATUS_TIME_HINTS)


def _mentions_created_time(text: str) -> bool:
    lower = (text or "").lower()
    return any(kw in (text or "") for kw in ("创建", "新建")) or "create" in lower or "created" in lower


def _is_remaining_count_question(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    return any(hint in t for hint in _REMAINING_COUNT_HINTS) or (
        "多少" in t and "任务" in t and any(kw in t for kw in ("还剩", "剩余", "剩下", "未完成"))
    )


def _looks_like_task_question_fragment(value: Optional[str], raw_query: str = "") -> bool:
    token = (value or "").strip()
    if not token:
        return False
    raw = (raw_query or "").strip()
    if raw and token == raw and _is_list_style_question(raw):
        return True
    if any(fragment in token for fragment in _TASK_QUESTION_FRAGMENTS):
        return True
    if "?" in token or "？" in token:
        return True
    if len(token) > 30 and any(kw in token for kw in ("哪些", "什么", "多少")):
        return True
    return False


def _clean_spurious_task_entities(spec: TaskQuerySpec, text: str) -> None:
    list_style = _is_list_style_question(text)
    cleanups: List[str] = []

    task_value = (getattr(spec, "task", None) or "").strip()
    if task_value and (list_style or _looks_like_task_question_fragment(task_value, text)):
        spec.task = None
        cleanups.append("cleared_task_question_fragment")

    filters = list(getattr(spec, "filters", []) or [])
    if filters:
        kept: List[QueryFilter] = []
        removed_task_filter = False
        for flt in filters:
            field = str(getattr(flt, "field", "") or "").lower()
            if field != "task":
                kept.append(flt)
                continue
            op = str(getattr(flt, "op", "") or "eq").lower()
            if list_style:
                removed_task_filter = True
                continue
            if op == "in":
                values = [
                    val
                    for val in (getattr(flt, "values", None) or [])
                    if not _looks_like_task_question_fragment(str(val), text)
                    and not _looks_like_person(str(val))
                ]
                if values:
                    flt.values = values
                    kept.append(flt)
                else:
                    removed_task_filter = True
                continue
            value = getattr(flt, "value", None)
            if _looks_like_task_question_fragment(str(value or ""), text) or _looks_like_person(
                str(value or "")
            ):
                removed_task_filter = True
                continue
            kept.append(flt)
        if removed_task_filter:
            cleanups.append("removed_spurious_task_filter")
        spec.filters = kept

    if list_style and getattr(spec, "task_keywords", None):
        keywords = [
            kw
            for kw in (getattr(spec, "task_keywords", None) or [])
            if not _looks_like_task_question_fragment(str(kw), text)
            and str(kw).strip() not in {"任务", "有哪些", "哪些"}
        ]
        if len(keywords) != len(getattr(spec, "task_keywords", None) or []):
            cleanups.append("trimmed_task_keywords")
        spec.task_keywords = keywords

    if cleanups:
        extra = getattr(spec, "extra", None) or {}
        current = list(extra.get("ir_cleanups") or [])
        for item in cleanups:
            if item not in current:
                current.append(item)
        extra["ir_cleanups"] = current
        spec.extra = extra


def _clear_due_polluted_time_range(spec: TaskQuerySpec, text: str) -> None:
    if not getattr(spec, "due_range", None):
        return
    if not getattr(spec, "time_range", None):
        return
    if not _mentions_due_time(text) or _mentions_status_time(text):
        return
    spec.time_range = None
    extra = getattr(spec, "extra", None) or {}
    current = list(extra.get("ir_cleanups") or [])
    if "cleared_time_range_for_due_query" not in current:
        current.append("cleared_time_range_for_due_query")
    extra["ir_cleanups"] = current
    spec.extra = extra


def _normalize_due_range_from_text(spec: TaskQuerySpec, text: str) -> None:
    if not _mentions_due_time(text):
        return
    detected = _detect_due_range(text)
    if not detected:
        return
    current = getattr(spec, "due_range", None)
    if current and current.start == detected.start and current.end == detected.end:
        return
    spec.due_range = detected
    extra = getattr(spec, "extra", None) or {}
    current_cleanups = list(extra.get("ir_cleanups") or [])
    if "normalized_due_range_from_text" not in current_cleanups:
        current_cleanups.append("normalized_due_range_from_text")
    extra["ir_cleanups"] = current_cleanups
    spec.extra = extra


def _clear_unmentioned_created_range(spec: TaskQuerySpec, text: str) -> None:
    if not getattr(spec, "created_range", None):
        return
    if _mentions_created_time(text):
        return
    spec.created_range = None
    extra = getattr(spec, "extra", None) or {}
    current = list(extra.get("ir_cleanups") or [])
    if "cleared_unmentioned_created_range" not in current:
        current.append("cleared_unmentioned_created_range")
    extra["ir_cleanups"] = current
    spec.extra = extra


def _normalize_statuses_against_config(spec: TaskQuerySpec) -> None:
    if not _TASK_STATUS_VALUES or {"DONE", "TODO"} - _TASK_STATUS_VALUES:
        return
    normalized: List[TaskStatus] = []
    mapped: List[str] = []
    seen: Set[TaskStatus] = set()
    for status in getattr(spec, "status", None) or []:
        current = status if isinstance(status, TaskStatus) else None
        if current is None:
            try:
                current = TaskStatus(str(status))
            except ValueError:
                continue
        if current in (TaskStatus.IN_PROGRESS, TaskStatus.BLOCKED):
            mapped.append(f"{current.value}->TODO")
            current = TaskStatus.TODO
        if current != TaskStatus.ANY and current.value not in _TASK_STATUS_VALUES:
            continue
        if current not in seen:
            seen.add(current)
            normalized.append(current)
    if normalized != list(getattr(spec, "status", None) or []):
        spec.status = normalized
    if mapped:
        extra = getattr(spec, "extra", None) or {}
        current = list(extra.get("status_value_mappings") or [])
        for item in mapped:
            if item not in current:
                current.append(item)
        extra["status_value_mappings"] = current
        spec.extra = extra

def _looks_like_person(token: str) -> bool:
    candidate = _normalize_person_candidate(token)
    if not candidate:
        return False
    if _person_has_stopword(candidate):
        return False
    if len(candidate) < 2 or len(candidate) > 4:
        return False
    cjk_chars = sum(1 for ch in candidate if "\u4e00" <= ch <= "\u9fff")
    return cjk_chars >= len(candidate) - 1


def _extract_person_tokens_from_text(text: str) -> List[str]:
    persons: List[str] = []
    segments: List[str] = []
    for anchor in ("任务", "最近", "还有", "有哪些"):
        if anchor in text:
            segments.append(text.split(anchor, 1)[0])
    segments.append(text)
    trim_anchors = ("自己", "有关", "相关", "以及")
    for segment in segments:
        for token in _split_multi_values(segment):
            token = token.strip()
            for anchor in trim_anchors:
                if anchor in token:
                    token = token.split(anchor, 1)[0]
                    break
            normalized = _sanitize_person_value(token)
            if normalized and _looks_like_person(normalized) and normalized not in persons:
                persons.append(normalized)
        if len(persons) >= 2:
            break
    return persons


def _detect_limit(text: str) -> Optional[int]:
    m = _LIMIT_PATTERN.search(text)
    if not m:
        return None
    try:
        value = int(m.group(1))
    except ValueError:
        return None
    return max(1, min(value, 200))


def _detect_project(text: str) -> Optional[str]:
    for pattern in _PROJECT_PATTERNS:
        match = pattern.search(text)
        if match:
            name = match.group("name")
            if name:
                return name.strip()
    return None


def _detect_priority(text: str) -> Optional[int]:
    lower = text.lower()
    for hint, value in _PRIORITY_HINTS:
        if hint.lower() in lower:
            return value
    return None


def _normalize_tag_value(value: str) -> str:
    token = (value or "").strip()
    if not token:
        return ""
    token = token.lstrip("#")
    token = token.rstrip("\u6807\u7b7e").rstrip("\u7684")
    token = re.split(r"[，,。.;；、]+", token)[0]
    token = re.sub(r"\s+", "", token)
    return token.strip()


def _extract_tags(text: str) -> List[str]:
    tags: List[str] = [_normalize_tag_value(match.group(1)) for match in _TAG_PATTERN.finditer(text)]
    phrase_hits: List[str] = []
    for match in _TAG_PHRASE_PATTERN.finditer(text):
        candidate = _normalize_tag_value(match.group(1))
        if candidate:
            phrase_hits.append(candidate)
    ordered: List[str] = []
    for tag in tags + phrase_hits:
        if tag and tag not in ordered:
            ordered.append(tag)
    return ordered


def _build_multi_filter(field: str, values: List[str]) -> Optional[QueryFilter]:
    if not values:
        return None
    values = [v for v in values if v]
    if len(values) <= 1:
        return None
    return QueryFilter(field=field, op="in", values=values)


def _filter_value_text(value: Any) -> Optional[str]:
    coerced = _coerce_sql_param_scalar(value)
    if coerced in (None, ""):
        return None
    return str(coerced).strip() or None


def _normalize_filter_payloads(spec: TaskQuerySpec) -> None:
    normalized_filters: List[QueryFilter] = []
    for flt in list(getattr(spec, "filters", None) or []):
        field = str(getattr(flt, "field", "") or "").lower()
        op = str(getattr(flt, "op", "") or "eq").lower()
        values = list(getattr(flt, "values", None) or [])
        if values:
            scalar_values: List[Any] = []
            for item in values:
                if field == "priority":
                    priority_value = _coerce_priority_value(item)
                    if priority_value is not None:
                        scalar_values.append(priority_value)
                        continue
                text_value = _filter_value_text(item)
                if text_value is not None:
                    scalar_values.append(text_value)
            if op == "in":
                flt.values = scalar_values
            elif len(scalar_values) == 1 and getattr(flt, "value", None) in (None, ""):
                flt.value = scalar_values[0]
                flt.values = None
            elif len(scalar_values) > 1 and field in {"person", "task", "status", "project", "owner"}:
                flt.op = "in"
                flt.value = None
                flt.values = scalar_values
            elif scalar_values:
                flt.values = scalar_values
        if field == "priority" and getattr(flt, "value", None) not in (None, ""):
            priority_value = _coerce_priority_value(getattr(flt, "value", None))
            if priority_value is not None:
                flt.value = priority_value
        normalized_filters.append(flt)
    spec.filters = normalized_filters


def _extract_person_task(text: str) -> Dict[str, Optional[str]]:
    """Return parsed person/task strings from a natural language question."""
    match = _PERSON_TASK_STATUS_RE.search(text)
    if match:
        return {
            "person": _sanitize_person_value(match.group("person")),
            "task": match.group("task").strip(),
        }

    match = _PERSON_TASK_GENERIC_RE.search(text)
    if match:
        return {
            "person": _sanitize_person_value(match.group("person")),
            "task": match.group("task").strip(),
        }

    # Pattern: "<person>最近还有哪些任务"
    recent_tasks_hint = "\u6700\u8fd1\u8fd8\u6709\u54ea\u4e9b\u4efb\u52a1"
    if recent_tasks_hint in text:
        left, _, _ = text.partition(recent_tasks_hint)
        left = left.strip()
        if left:
            return {"person": _sanitize_person_value(left), "task": None}

    if "\u6807\u7b7e" in text or "#" in text:
        return {"person": None, "task": text}

    return {"person": None, "task": None}


def _range_is_empty(value: Optional[TimeRange]) -> bool:
    return value is None or (not value.start and not value.end)


def _is_list_style_question(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    # Keep it lightweight: detect "list tasks" queries.
    if "任务" not in t:
        return False
    return any(
        kw in t
        for kw in (
            "有哪些",
            "都有哪些",
            "哪些任务",
            "列出",
            "列出来",
            "给我",
            "任务列表",
            "都列出来",
        )
    )


def _prune_filters_by_field(spec: TaskQuerySpec, field_name: str) -> None:
    filters = list(getattr(spec, "filters", []) or [])
    if not filters:
        return
    normalized = field_name.lower()
    kept: List[QueryFilter] = []
    for flt in filters:
        try:
            flt_field = str(getattr(flt, "field", "") or "").lower()
        except Exception:
            flt_field = ""
        if flt_field == normalized:
            continue
        kept.append(flt)
    spec.filters = kept


def _ensure_person_filter_from_text(spec: TaskQuerySpec, text: str) -> None:
    persons = _extract_person_tokens_from_text(text)
    if len(persons) < 2:
        return
    filters = list(getattr(spec, "filters", []) or [])
    for flt in filters:
        if (
            str(getattr(flt, "field", "")).lower() == "person"
            and str(getattr(flt, "op", "")).lower() == "in"
        ):
            normalized = [
                text_value
                for item in (getattr(flt, "values", None) or [])
                for text_value in [_filter_value_text(item)]
                if text_value
            ]
            if not set(persons).issubset(set(normalized)):
                normalized = persons
            flt.values = normalized
            spec.filters = filters
            if not (getattr(spec, "person", None) or "").strip():
                spec.person = persons[0]
            return
    has_person_filter = any(
        str(getattr(f, "field", "")).lower() == "person"
        and getattr(f, "values", None)
        for f in filters
    )
    if has_person_filter:
        if not (getattr(spec, "person", None) or "").strip():
            spec.person = persons[0]
        return
    filters.append(QueryFilter(field="person", op="in", values=persons))
    spec.filters = filters
    existing = (getattr(spec, "person", None) or "").strip()
    if existing in {"??", "?", "unknown", "UNKNOWN"}:
        spec.person = None
        return
    if not existing:
        # Keep a representative person for downstream heuristics/UI, while the
        # actual SQL should use the IN filter for correctness.
        spec.person = persons[0]


def _post_process_intent(spec: TaskQuerySpec, text: str) -> None:
    """Apply lightweight intent heuristics on top of LLM/rule output."""
    t = (text or "").strip()
    t_lower = t.lower()
    if not t:
        return
    # Phase 3 heuristics freeze: keep this function limited to the generic
    # heuristics below (completion-time detection, status intent tweaks,
    # multi-person filters, time/due/priority hints, limit/order adjustments).
    # New domain-specific behaviors should be expressed via LLM IR fields /
    # answer_mode, not by adding more pattern matching here.

    raw_mode = getattr(spec, "answer_mode", TaskAnswerMode.default)
    if isinstance(raw_mode, TaskAnswerMode):
        answer_mode = raw_mode
    else:
        try:
            answer_mode = TaskAnswerMode(str(raw_mode))
        except Exception:
            answer_mode = TaskAnswerMode.default
    spec.answer_mode = answer_mode

    list_style = _is_list_style_question(t)
    enterprise_user_match = re.search(r"\bUser_\d+\b", t, flags=re.IGNORECASE)
    enterprise_user = enterprise_user_match.group(0) if enterprise_user_match else None
    enterprise_division_match = re.search(r"\bDivision_\d+\b", t, flags=re.IGNORECASE)
    enterprise_division = enterprise_division_match.group(0) if enterprise_division_match else None
    enterprise_org_match = re.search(r"\bOrg_\d+\b", t, flags=re.IGNORECASE)
    enterprise_org = enterprise_org_match.group(0) if enterprise_org_match else None
    enterprise_post_match = re.search(r"\bPost_\d+\b", t, flags=re.IGNORECASE)
    enterprise_post = enterprise_post_match.group(0) if enterprise_post_match else None

    if not getattr(spec, "tags", None):
        try:
            spec.tags = _extract_tags(t)
        except Exception:
            spec.tags = []
    tags_normalized = [tag.strip() for tag in (getattr(spec, "tags", None) or []) if tag]
    tag_set = set(tags_normalized)
    if tag_set:
        person_value = (getattr(spec, "person", None) or "").strip()
        if person_value and person_value in tag_set:
            spec.person = None
        filters = list(getattr(spec, "filters", []) or [])
        cleaned_filters: List[QueryFilter] = []
        for flt in filters:
            try:
                field = str(getattr(flt, "field", "") or "").lower()
            except Exception:
                field = ""
            if field != "person":
                cleaned_filters.append(flt)
                continue
            op = str(getattr(flt, "op", "") or "").lower()
            if op == "in":
                values = getattr(flt, "values", None) or []
                remaining = [v for v in values if str(v).strip() not in tag_set]
                if remaining:
                    flt.values = remaining
                    cleaned_filters.append(flt)
                continue
            value = (getattr(flt, "value", None) or "").strip()
            if value and value in tag_set:
                continue
            cleaned_filters.append(flt)
        spec.filters = cleaned_filters

    if not getattr(spec, "tags", None):
        try:
            spec.tags = _extract_tags(t)
        except Exception:
            pass

    # KG-lite: track whether any KG resolution is applied for this spec.
    kg_used = False

    # KG-lite person normalization
    if getattr(spec, "person", None):
        original_person = str(spec.person)
        resolved_person = kg_lite.resolve_person(original_person)
        if resolved_person and resolved_person != original_person:
            spec.extra.setdefault("kg_person_source", original_person)
            spec.person = resolved_person
            kg_used = True

    # KG-lite project normalization (from project field or full text).
    original_project = getattr(spec, "project", None)
    resolved_project = kg_lite.resolve_project(original_project, t)
    if resolved_project and resolved_project != original_project:
        if original_project:
            spec.extra.setdefault("kg_project_source", original_project)
        else:
            spec.extra.setdefault("kg_project_source", t)
        spec.project = resolved_project
        kg_used = True

    # KG-lite category -> tags expansion from text (even if tags already exist, we can enrich).
    kg_tags = kg_lite.resolve_category_tags(t)
    if kg_tags:
        existing_tags = getattr(spec, "tags", None) or []
        merged = list(existing_tags)
        for tag in kg_tags:
            if tag and tag not in merged:
                merged.append(tag)
        if merged != existing_tags:
            spec.tags = merged
            spec.extra.setdefault("kg_category_source", t)
            kg_used = True

    # KG-lite status normalization (LLM outputs may use synonyms).
    status_list = getattr(spec, "status", None) or []
    if status_list:
        normalized: List[TaskStatus] = []
        seen: set[TaskStatus] = set()
        for raw_status in status_list:
            if isinstance(raw_status, TaskStatus):
                candidate = raw_status.value
            else:
                candidate = str(raw_status or "")
            if not candidate:
                continue
            canonical = kg_lite.resolve_status_value(candidate) or candidate.upper()
            try:
                enum_value = TaskStatus(canonical)
            except ValueError:
                continue
            if enum_value not in seen:
                seen.add(enum_value)
                normalized.append(enum_value)
        if normalized:
            spec.status = normalized
            kg_used = True

    # KG-lite priority normalization.
    if getattr(spec, "priority", None) is not None:
        resolved_priority = kg_lite.resolve_priority_value(spec.priority)
        if resolved_priority is not None:
            if resolved_priority != spec.priority:
                kg_used = True
            spec.priority = resolved_priority
        else:
            spec.priority = None

    if kg_used:
        spec.extra["kg_enabled"] = True


    raw_text_lower = t_lower
    if "p1" in raw_text_lower and ("高优" in t or "高优先级" in t):
        if spec.priority is None:
            spec.priority = 1
        task_val = getattr(spec, "task", None)
        if isinstance(task_val, str) and "高优" in task_val and "p1" in task_val.lower():
            spec.task = None
        filters = list(getattr(spec, "filters", []) or [])
        cleaned_filters: List[QueryFilter] = []
        for flt in filters:
            try:
                field = str(getattr(flt, "field", "") or "").lower()
            except Exception:
                field = ""
            if field != "task":
                cleaned_filters.append(flt)
                continue
            op = str(getattr(flt, "op", "") or "").lower()
            if op == "in":
                values = getattr(flt, "values", None) or []
                remaining = [v for v in values if not (isinstance(v, str) and "高优" in v and "p1" in v.lower())]
                if remaining:
                    flt.values = remaining
                    cleaned_filters.append(flt)
                continue
            value = getattr(flt, "value", None)
            if isinstance(value, str) and "高优" in value and "p1" in value.lower():
                continue
            cleaned_filters.append(flt)
        spec.filters = cleaned_filters

    intent = getattr(spec, "intent", None)
    status_kws = ("已完成", "未完成", "done", "todo", "搞定", "结束")
    status_kws_lower = tuple(kw.lower() for kw in status_kws)

    if _is_remaining_count_question(t):
        spec.intent = TaskQueryIntent.task_status_list
        spec.answer_mode = TaskAnswerMode.task_count_by_status
        spec.status = [TaskStatus.TODO]
        spec.task = None
        _prune_filters_by_field(spec, "task")

    # 如果 LLM/规则没有给出任何 status，就根据文本自动推断一遍
    if not getattr(spec, "status", None):
        status_hints: list[TaskStatus] = []
        status_map: list[tuple[tuple[str, ...], TaskStatus]] = [
            (("阻塞", "卡住", "block", "blocked"), TaskStatus.BLOCKED),
            (("进行中", "在做", "in progress"), TaskStatus.IN_PROGRESS),
            (("已完成", "做完", "完成了", "done"), TaskStatus.DONE),
            (("未完成", "没做", "还没做", "todo"), TaskStatus.TODO),
        ]
        for kws, st in status_map:
            if any(kw.lower() in t_lower for kw in kws):
                status_hints.append(st)
        if status_hints:
            seen: set[TaskStatus] = set()
            normalized: list[TaskStatus] = []
            for st in status_hints:
                if st not in seen:
                    seen.add(st)
                    normalized.append(st)
            spec.status = normalized

    if intent == TaskQueryIntent.task_list_by_person:
        if not any(kw in t_lower for kw in status_kws_lower):
            spec.status = []

    if "现在有哪些任务" in t or "任务列表" in t:
        spec.intent = TaskQueryIntent.task_list_by_person
        if not spec.status:
            spec.status = [TaskStatus.DONE, TaskStatus.TODO]

    if "都是什么状态" in t or ("历史" in t and "状态" in t):
        spec.intent = TaskQueryIntent.task_history

    # List-style queries ("有哪些/列出...任务") should not be forced into
    # task_status_single by spurious LLM guesses for person/task.
    if list_style and getattr(spec, "intent", None) == TaskQueryIntent.task_status_single:
        person_tokens = _extract_person_tokens_from_text(t)
        if len(person_tokens) == 1 and ("有哪些" in t or "列表" in t or "列出" in t):
            spec.intent = TaskQueryIntent.task_list_by_person
        else:
            spec.intent = TaskQueryIntent.task_status_list

    # Enterprise hint: "User_14 ..." often refers to owner_name rather than executor.
    # Only apply when the token exists in the text and the question does not say "执行".
    if list_style and enterprise_user and "执行" not in t and "执行人" not in t:
        person_value = (getattr(spec, "person", None) or "").strip()
        if person_value and person_value.lower() == enterprise_user.lower():
            _prune_filters_by_field(spec, "person")
            spec.filters = list(getattr(spec, "filters", []) or []) + [
                QueryFilter(field="owner", op="eq", value=enterprise_user)
            ]
            spec.person = None

    # Enterprise hint: Org_/Division_/Post_ tokens map to organization fields, not project/person.
    if enterprise_division:
        project_value = (getattr(spec, "project", None) or "").strip()
        if project_value and project_value.lower() == enterprise_division.lower():
            spec.project = None
        spec.filters = list(getattr(spec, "filters", []) or []) + [
            QueryFilter(field="division_name", op="eq", value=enterprise_division)
        ]
    if enterprise_org:
        project_value = (getattr(spec, "project", None) or "").strip()
        if project_value and project_value.lower() == enterprise_org.lower():
            spec.project = None
        spec.filters = list(getattr(spec, "filters", []) or []) + [
            QueryFilter(field="org_name", op="eq", value=enterprise_org)
        ]
    if enterprise_post:
        project_value = (getattr(spec, "project", None) or "").strip()
        if project_value and project_value.lower() == enterprise_post.lower():
            spec.project = None
        spec.filters = list(getattr(spec, "filters", []) or []) + [
            QueryFilter(field="post_name", op="eq", value=enterprise_post)
        ]

    # Enterprise hint: read/delegated flags.
    if "阅读" in t or "已读" in t or "未读" in t:
        if "尚未阅读" in t or "未阅读" in t or "未读" in t:
            spec.filters = list(getattr(spec, "filters", []) or []) + [
                QueryFilter(field="is_read", op="eq", value=0)
            ]
        elif "已经被阅读" in t or "已阅读" in t or "已读" in t:
            spec.filters = list(getattr(spec, "filters", []) or []) + [
                QueryFilter(field="is_read", op="eq", value=1)
            ]
    if "委托" in t:
        if "未被委托" in t or "未委托" in t:
            spec.filters = list(getattr(spec, "filters", []) or []) + [
                QueryFilter(field="is_delegated", op="eq", value=0)
            ]
        elif "已经被委托" in t or "已被委托" in t or "已委托" in t:
            spec.filters = list(getattr(spec, "filters", []) or []) + [
                QueryFilter(field="is_delegated", op="eq", value=1)
            ]

    # If the text does not contain an explicit person/task mention, prefer
    # clearing guessed entities so the query can be answered as a list.
    if list_style and getattr(spec, "intent", None) in (
        TaskQueryIntent.task_status_list,
        TaskQueryIntent.task_list_by_person,
    ):
        person_tokens = _extract_person_tokens_from_text(t)
        kg_person_source = None
        try:
            kg_person_source = (getattr(spec, "extra", {}) or {}).get("kg_person_source")
        except Exception:
            kg_person_source = None
        person_value = (getattr(spec, "person", None) or "").strip()
        is_placeholder_person = person_value in {"??", "?", "unknown", "UNKNOWN"}
        if getattr(spec, "person", None) and not is_placeholder_person:
            if person_tokens and (
                str(spec.person) not in person_tokens
                and (not kg_person_source or str(kg_person_source) not in person_tokens)
            ):
                spec.person = None
                _prune_filters_by_field(spec, "person")
            elif not person_tokens:
                spec.person = None
                _prune_filters_by_field(spec, "person")
        if getattr(spec, "task", None) and str(spec.task) not in t:
            spec.task = None
            _prune_filters_by_field(spec, "task")

    completion_mode = any(kw in t for kw in _COMPLETION_TIME_HINTS)
    if completion_mode:
        spec.intent = TaskQueryIntent.task_history
        spec.answer_mode = TaskAnswerMode.completion_time_latest
        spec.status = [TaskStatus.DONE]
        spec.limit = 1
        if not spec.order_by:
            spec.order_by = [OrderBySpec(field="ts", direction=OrderByDirection.desc)]

    if (
        t.endswith("什么状态？")
        or t.endswith("什么状态?")
        or "现在什么状态" in t
        or "是什么状态" in t
    ):
        if spec.person and spec.task:
            spec.intent = TaskQueryIntent.task_status_single
            if not spec.status:
                spec.status = [TaskStatus.DONE, TaskStatus.TODO]
            else:
                spec.status = [
                    s
                    for s in spec.status
                    if not isinstance(s, TaskStatus) or s != TaskStatus.ANY
                ]
            if spec.limit is None or spec.limit > 1:
                spec.limit = 1

    _ensure_person_filter_from_text(spec, t)
    entity_guess = _extract_person_task(t)
    person_tokens_hint = _extract_person_tokens_from_text(t)
    guessed_person = entity_guess.get("person")
    if not person_tokens_hint and guessed_person:
        try:
            candidate = _sanitize_person_value(str(guessed_person))
        except Exception:
            candidate = None
        if candidate and _looks_like_person(candidate):
            person_tokens_hint = [candidate]

    has_person_scope_filter = any(
        str(getattr(f, "field", "") or "").lower() == "person"
        and str(getattr(f, "op", "") or "").lower() == "in"
        and getattr(f, "values", None)
        for f in (getattr(spec, "filters", None) or [])
        if isinstance(f, QueryFilter)
    )

    if (
        not getattr(spec, "person", None)
        and len(person_tokens_hint) == 1
        and not has_person_scope_filter
    ):
        spec.person = person_tokens_hint[0]
    if not getattr(spec, "task", None) and spec.intent in (
        TaskQueryIntent.task_status_single,
        TaskQueryIntent.task_history,
    ):
        guessed_task = entity_guess.get("task")
        if guessed_task:
            spec.task = guessed_task.strip()
        elif getattr(spec, "task_keywords", None):
            first_kw = next((kw for kw in spec.task_keywords if kw), None)
            if first_kw:
                spec.task = first_kw

    if getattr(spec, "person", None):
        sanitized_person = _sanitize_person_value(str(spec.person))
        spec.person = sanitized_person
    if getattr(spec, "task", None):
        spec.task = str(spec.task).strip()

    _normalize_filter_payloads(spec)
    _clean_spurious_task_entities(spec, t)

    if spec.intent == TaskQueryIntent.task_list_by_person and not spec.person:
        spec.is_supported = False
        spec.extra.setdefault("unsupported_reason", "list_by_person_missing_person")


    if _range_is_empty(spec.time_range):
        tr = _detect_time_range(t)
        if tr:
            spec.time_range = tr

    if _range_is_empty(spec.due_range):
        dr = _detect_due_range(t)
        if dr:
            spec.due_range = dr

    if _range_is_empty(spec.created_range):
        cr = _detect_created_range(t)
        if cr:
            spec.created_range = cr

    _normalize_due_range_from_text(spec, t)
    _clear_due_polluted_time_range(spec, t)
    _clear_unmentioned_created_range(spec, t)
    _normalize_statuses_against_config(spec)

    if spec.priority is None:
        pr = _detect_priority(t)
        if pr is not None:
            spec.priority = pr

    if spec.intent == TaskQueryIntent.task_list_by_person:
        if spec.limit is None or spec.limit < 1 or spec.limit > 200:
            spec.limit = 50
    else:
        try:
            limit_val = int(spec.limit) if spec.limit is not None else None
        except Exception:
            limit_val = None
        if limit_val is None or limit_val < 1 or limit_val > 200:
            spec.limit = 10
        else:
            spec.limit = limit_val


def parse_task_query_nl(q: str) -> TaskQuerySpec:
    """Entry point: parse natural language into TaskQuerySpec."""
    text = (q or "").strip()
    if not text:
        return TaskQuerySpec(intent=TaskQueryIntent.unknown, raw_query=q)

    llm_error: Optional[Exception] = None

    # 优先尝试 LLM 解析。
    if _USE_LLM_FOR_NL2SQL:
        try:
            client = get_llm_client()
            raw = client.generate_task_query_spec(text)
            spec = TaskQuerySpec.parse_obj(raw)
            if not spec.raw_query:
                spec.raw_query = q
            _post_process_intent(spec, text)
            spec.extra.setdefault("nl2sql_source", "llm")
            return spec
        except Exception as exc:
            llm_error = exc
            logging.exception(
                "TASKS_NL2SQL LLM parse failed; falling back to rule-based parser"
            )

    # 回退到规则解析。
    spec = _rule_based_parse_task_query_nl_v2(text)
    spec.raw_query = q
    _post_process_intent(spec, text)
    spec.extra.setdefault("nl2sql_source", "rules")
    if llm_error is not None:
        spec.extra.setdefault("nl2sql_llm_error", str(llm_error))
    return spec


def sanitize_task_query_spec_for_downstream(spec: TaskQuerySpec) -> TaskQuerySpec:
    """Return a cleaned copy of TaskQuerySpec for prompts or fallback SQL."""

    cleaned = spec.copy(deep=True)
    text = (getattr(cleaned, "raw_query", "") or "").strip()
    _normalize_filter_payloads(cleaned)
    _clean_spurious_task_entities(cleaned, text)
    _normalize_due_range_from_text(cleaned, text)
    _clear_due_polluted_time_range(cleaned, text)
    _clear_unmentioned_created_range(cleaned, text)
    _normalize_statuses_against_config(cleaned)
    return cleaned


def _rule_based_parse_task_query_nl(q: str) -> TaskQuerySpec:
    text = (q or "").strip()
    text_lower = text.lower()

    # 1) 粗略意图识别
    intent = TaskQueryIntent.task_status_single
    if any(kw in text for kw in ("列表", "有哪些", "所有", "全部")):
        intent = TaskQueryIntent.task_status_list
    if any(kw in text for kw in ("张三", "李四", "老王", "老张")) and "有哪些" in text:
        intent = TaskQueryIntent.task_list_by_person

    # 2) 粗略人物和任务抽取
    person: Optional[str] = None
    task: Optional[str] = None

    if "什么状态" in text:
        # 示例：张三的E3D接口联调现在什么状态？
        left, _, right = text.partition("什么状态")
        if left:
            person = left.strip()
        task = right.strip() or None
    else:
        task = text or None

    # 3) 粗略状态提示词
    status: List[TaskStatus] = []
    if any(
        kw.lower() in text_lower
        for kw in ("已完成", "已经完成", "搞定", "完成了", "done")
    ):
        status = [TaskStatus.DONE]
    elif any(
        kw.lower() in text_lower
        for kw in ("未完成", "没完成", "还没做", "待办", "todo")
    ):
        status = [TaskStatus.TODO]

    # 4) 默认排序和 limit
    order_by = [OrderBySpec(field="ts", direction=OrderByDirection.desc)]
    limit = 10

    return TaskQuerySpec(
        intent=intent,
        raw_query=q,
        person=person or None,
        task=(task or "").strip() or None,
        task_keywords=[],
        status=status,
        time_range=None,
        order_by=order_by,
        limit=limit,
        extra={},
    )


def _rule_based_parse_task_query_nl_v2(q: str) -> TaskQuerySpec:
    """Cleaner rule-based parser with expanded hints and filters."""
    text = (q or "").strip()

    text_lower = text.lower()
    list_style = _is_list_style_question(text)

    # 1) coarse intent guess
    intent = TaskQueryIntent.task_status_single
    answer_mode = TaskAnswerMode.default
    remaining_count = _is_remaining_count_question(text)
    if any(kw in text for kw in ("列表", "有哪些", "所有", "全部", "清单")):
        intent = TaskQueryIntent.task_status_list
    if "任务列表" in text or "有哪些" in text:
        intent = TaskQueryIntent.task_list_by_person
    if remaining_count:
        intent = TaskQueryIntent.task_status_list
        answer_mode = TaskAnswerMode.task_count_by_status

    completion_mode = any(kw in text for kw in _COMPLETION_TIME_HINTS)
    if completion_mode:
        intent = TaskQueryIntent.task_history

    # 2) coarse entity extraction
    entities = _extract_person_task(text)
    raw_person = entities.get("person")
    person = _sanitize_person_value(raw_person)
    task = entities.get("task")
    if list_style and task and _looks_like_task_question_fragment(task, text):
        task = None
    if remaining_count:
        task = None
    if not task and not list_style and not remaining_count:
        task = text or None

    person_tokens_raw = _split_multi_values(raw_person)
    text_person_tokens = _extract_person_tokens_from_text(text)
    if text_person_tokens:
        person_tokens_raw = text_person_tokens
    elif not person_tokens_raw and person:
        person_tokens_raw = [person]
    person_tokens: List[str] = []
    for token in person_tokens_raw:
        normalized = _sanitize_person_value(token)
        if normalized and _looks_like_person(normalized) and normalized not in person_tokens:
            person_tokens.append(normalized)
    task_tokens = [] if (list_style or remaining_count) else _split_multi_values(task)
    filters: List[QueryFilter] = []
    person_filter = _build_multi_filter("person", person_tokens)
    if person_filter:
        filters.append(person_filter)
    task_filter = _build_multi_filter("task", task_tokens)
    if task_filter:
        filters.append(task_filter)
    if not person and person_tokens:
        person = person_tokens[0]
    if not task and task_tokens:
        task = task_tokens[0]

    # 3) status hints (expanded, allow multiple)
    status: List[TaskStatus] = []
    status_map = [
        (("阻塞", "卡住", "block", "blocked"), TaskStatus.BLOCKED),
        (("进行", "进展", "跟进", "在做", "in progress"), TaskStatus.IN_PROGRESS),
        (("完成了吗", "完成了没", "搞定了没", "搞定没有", "done"), TaskStatus.DONE),
        (("未完成", "没完成", "还没", "待办", "todo"), TaskStatus.TODO),
    ]
    for kws, st in status_map:
        if any(kw.lower() in text_lower for kw in kws):
            status.append(st)
    if remaining_count:
        status = [TaskStatus.TODO]
    elif completion_mode:
        answer_mode = TaskAnswerMode.completion_time_latest
        status = [TaskStatus.DONE]
    elif not status and "状态" in text and intent == TaskQueryIntent.task_status_single:
        status = [TaskStatus.DONE, TaskStatus.TODO]

    project = _detect_project(text)
    if project:
        person_tokens = [tok for tok in person_tokens if tok and tok != project]
        if person == project:
            person = None
        filters.append(QueryFilter(field="project", op="eq", value=project))

    tags = _extract_tags(text)
    priority = _detect_priority(text)
    detected_limit = _detect_limit(text)
    due_range = _detect_due_range(text)
    time_range = None if due_range and not _mentions_status_time(text) else _detect_time_range(text)
    created_range = _detect_created_range(text)

    order_by = [
        OrderBySpec(field="ts", direction=OrderByDirection.desc),
        OrderBySpec(field="priority", direction=OrderByDirection.asc),
    ]
    limit = detected_limit or 10
    if completion_mode:
        order_by = [
            OrderBySpec(field="ts", direction=OrderByDirection.desc),
        ]
        limit = 1
    elif any(hint in text for hint in _ORDER_ASC_HINTS):
        order_by = [
            OrderBySpec(field="ts", direction=OrderByDirection.asc),
            OrderBySpec(field="priority", direction=OrderByDirection.asc),
        ]
    elif any(hint in text for hint in _ORDER_DESC_HINTS):
        order_by = [
            OrderBySpec(field="ts", direction=OrderByDirection.desc),
            OrderBySpec(field="priority", direction=OrderByDirection.asc),
        ]

    task_keywords = _extract_keywords(task)

    return TaskQuerySpec(
        intent=intent,
        raw_query=q,
        person=person or None,
        task=(task or "").strip() or None,
        task_keywords=task_keywords,
        project=project,
        tags=tags,
        priority=priority,
        status=status,
        time_range=time_range,
        due_range=due_range,
        created_range=created_range,
        order_by=order_by,
        limit=limit,
        filters=filters,
        answer_mode=answer_mode,
        extra={
            "rule_person_tokens": person_tokens,
            "rule_task_tokens": task_tokens,
        },
    )

def build_task_query_plan(spec: TaskQuerySpec) -> Dict[str, Any]:
    """Convert TaskQuerySpec into a generic query plan dict."""
    intent = (
        spec.intent.value
        if isinstance(spec.intent, TaskQueryIntent)
        else str(spec.intent)
    )
    raw_answer_mode = getattr(spec, "answer_mode", TaskAnswerMode.default)
    if isinstance(raw_answer_mode, TaskAnswerMode):
        answer_mode = raw_answer_mode
    else:
        try:
            answer_mode = TaskAnswerMode(str(raw_answer_mode))
        except Exception:
            answer_mode = TaskAnswerMode.default

    schema = get_tasks_schema_config()
    table = schema.history_relation if spec.intent == TaskQueryIntent.task_history else schema.latest_relation
    target: Dict[str, Any] = {"table": table}

    def _translate_ident(value: str) -> str:
        name = (value or "").strip()
        if not name:
            return name
        if _SAFE_FIELD_RE.match(name):
            return schema.translate_field(name)
        return name

    def _translate_projection(expr: str) -> str:
        token = (expr or "").strip()
        if not token:
            return token
        # Keep aggregates/aliases intact (e.g. COUNT(*) AS task_count).
        if "(" in token or " " in token:
            return token
        return _translate_ident(token)

    filters: List[Dict[str, Any]] = []
    group_by: List[str] = []
    has_person_filter = any(
        str(getattr(flt, "field", "") or "").lower() == "person"
        for flt in (getattr(spec, "filters", None) or [])
        if isinstance(flt, QueryFilter)
    )
    if spec.person and not has_person_filter:
        filters.append({"field": schema.translate_field("person"), "op": "eq", "value": spec.person})
    if spec.task:
        filters.append({"field": schema.translate_field("task"), "op": "eq", "value": spec.task})
    if spec.project:
        filters.append({"field": schema.translate_field("project"), "op": "eq", "value": spec.project})
    if spec.priority is not None:
        filters.append({"field": schema.translate_field("priority"), "op": "eq", "value": spec.priority})
    if spec.tags:
        for tag in spec.tags:
            filters.append({"field": schema.translate_field("tags"), "op": "like", "value": f"%{tag}%"})
    if spec.status:
        concrete = [
            s for s in spec.status if not isinstance(s, TaskStatus) or s != TaskStatus.ANY
        ]
        if concrete:
            filters.append(
                {
                    "field": schema.translate_field("status"),
                    "op": "in",
                    "value": [
                        s.value if isinstance(s, TaskStatus) else str(s)
                        for s in concrete
                    ],
                }
            )
    if spec.time_range:
        if spec.time_range.start:
            filters.append(
                {"field": schema.translate_field("ts"), "op": "gte", "value": spec.time_range.start}
            )
        if spec.time_range.end:
            filters.append(
                {"field": schema.translate_field("ts"), "op": "lte", "value": spec.time_range.end}
            )
    if spec.due_range:
        if spec.due_range.start:
            filters.append(
                {"field": schema.translate_field("due_ts"), "op": "gte", "value": spec.due_range.start}
            )
        if spec.due_range.end:
            filters.append({"field": schema.translate_field("due_ts"), "op": "lte", "value": spec.due_range.end})
    if spec.created_range:
        if spec.created_range.start:
            filters.append(
                {"field": schema.translate_field("created_ts"), "op": "gte", "value": spec.created_range.start}
            )
        if spec.created_range.end:
            filters.append(
                {"field": schema.translate_field("created_ts"), "op": "lte", "value": spec.created_range.end}
            )
    if spec.filters:
        invalid_filters: List[Dict[str, Any]] = []
        for flt in spec.filters:
            try:
                normalized = flt.to_plan_filter()
            except Exception as exc:
                invalid_filters.append(
                    {
                        "field": getattr(flt, "field", None),
                        "op": getattr(flt, "op", None),
                        "value": getattr(flt, "value", None),
                        "values": getattr(flt, "values", None),
                        "error": str(exc),
                    }
                )
                continue
            field = _translate_ident(str(normalized.get("field") or ""))
            op = str(normalized.get("op") or "").lower()
            if field and op == "in" and field == schema.translate_field("tags") and isinstance(
                normalized.get("value"), (list, tuple)
            ):
                for tag in normalized.get("value") or []:
                    if tag is None:
                        continue
                    filters.append(
                        {"field": schema.translate_field("tags"), "op": "like", "value": f"%{tag}%"}
                    )
            elif normalized.get("field"):
                normalized["field"] = field
                filters.append(normalized)
        if invalid_filters:
            try:
                extra = spec.extra or {}
                extra.setdefault("invalid_filters", [])
                extra["invalid_filters"].extend(invalid_filters)
                spec.extra = extra
            except Exception:
                pass

    if answer_mode == TaskAnswerMode.task_count_by_status:
        projections: List[str] = [
            _translate_ident("status"),
            "COUNT(*) AS task_count",
        ]
        group_by = [_translate_ident("status")]
    elif answer_mode == TaskAnswerMode.person_summary_by_project:
        projections = [
            _translate_ident("project"),
            _translate_ident("person"),
            _translate_ident("status"),
            "COUNT(*) AS task_count",
        ]
        group_by = [_translate_ident("project"), _translate_ident("person"), _translate_ident("status")]
    elif answer_mode == TaskAnswerMode.overdue_count_by_person:
        projections = [
            _translate_ident("person"),
            "COUNT(*) AS overdue_count",
        ]
        group_by = [_translate_ident("person")]
    elif spec.intent in (
        TaskQueryIntent.task_status_single,
        TaskQueryIntent.task_status_list,
        TaskQueryIntent.task_list_by_person,
        TaskQueryIntent.task_history,
    ):
        projections = [
            _translate_ident("id"),
            _translate_ident("person"),
            _translate_ident("task"),
            _translate_ident("status"),
            _translate_ident("ts"),
            _translate_ident("project"),
            _translate_ident("tags"),
            _translate_ident("priority"),
            _translate_ident("due_ts"),
            _translate_ident("created_ts"),
            _translate_ident("updated_ts"),
            _translate_ident("status_note"),
        ]
    elif spec.intent == TaskQueryIntent.person_summary:
        projections = [
            _translate_ident("person"),
            _translate_ident("status"),
            "COUNT(*) AS task_count",
        ]
        group_by = [_translate_ident("person"), _translate_ident("status")]
    else:
        projections = ["*"]

    sort: List[Dict[str, Any]] = [
        {
            "field": _translate_ident(ob.field),
            "direction": (
                ob.direction.value
                if isinstance(ob.direction, OrderByDirection)
                else str(ob.direction)
            ).upper(),
        }
        for ob in spec.order_by
    ]
    if answer_mode == TaskAnswerMode.task_count_by_status:
        sort = [
            {"field": "task_count", "direction": "DESC"},
            {"field": _translate_ident("status"), "direction": "ASC"},
        ]
    elif answer_mode == TaskAnswerMode.person_summary_by_project:
        sort = [
            {"field": _translate_ident("project"), "direction": "ASC"},
            {"field": _translate_ident("person"), "direction": "ASC"},
            {"field": _translate_ident("status"), "direction": "ASC"},
        ]
    elif answer_mode == TaskAnswerMode.overdue_count_by_person:
        sort = [
            {"field": "overdue_count", "direction": "DESC"},
            {"field": _translate_ident("person"), "direction": "ASC"},
        ]
    elif not sort:
        if spec.intent == TaskQueryIntent.person_summary:
            sort = [
                {"field": "task_count", "direction": "DESC"},
                {"field": _translate_ident("person"), "direction": "ASC"},
            ]
        else:
            # Default to latest-first on ts, then id for deterministic ordering.
            sort = [
                {"field": _translate_ident("ts"), "direction": "DESC"},
                {"field": _translate_ident("priority"), "direction": "ASC"},
                {"field": _translate_ident("id"), "direction": "DESC"},
            ]

    limit = spec.limit or 10

    # De-duplicate filters to avoid accidental over-constraint from mixed
    # "dedicated fields" + "filters" (especially after LLM post-processing).
    deduped: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for flt in filters:
        try:
            field = str(flt.get("field") or "")
            op = str(flt.get("op") or "")
            value = flt.get("value")
            if isinstance(value, list):
                value_key = tuple(value)
            else:
                value_key = value
            key = repr((field, op, value_key))
        except Exception:
            key = repr(flt)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(flt)
    filters = deduped

    return {
        "intent": intent,
        "target": target,
        "filters": filters,
        "projections": projections,
        "group_by": group_by,
        "sort": sort,
        "limit": limit,
    }


def build_query_plan_v2(spec: TaskQuerySpec) -> Dict[str, Any]:
    """
    Experimental multi-table-ready query planner.

    For now this is a thin wrapper around `build_task_query_plan(...)` and
    produces the same single-table / single-view query-plan IR.

    In the future, this function will be the only place where we introduce
    multi-table / multi-view semantics (joins to `persons`, `projects`, `tags`,
    etc.). The TaskQuerySpec IR shape and the KG-lite resolution APIs are
    intentionally kept stable and should not depend on the physical schema.
    """
    return build_task_query_plan(spec)

