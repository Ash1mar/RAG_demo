from __future__ import annotations

from enum import Enum
from os import getenv
from typing import Any, Dict, List, Optional, Set
import logging
import re

from pydantic import BaseModel, Field

from app.services.llm_client import get_llm_client


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
        op = (self.op or "eq").lower()
        payload: Dict[str, Any] = {"field": self.field, "op": op}
        if op == "in":
            vals: List[Any] = []
            if self.values:
                vals = list(self.values)
            elif self.value is not None:
                vals = [self.value]
            payload["value"] = vals
        elif op == "between":
            vals = list(self.values or [])
            if len(vals) < 2 and self.value is not None:
                vals.append(self.value)
            while len(vals) < 2:
                vals.append(None)
            payload["value"] = vals[:2]
        else:
            payload["value"] = self.value
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

_PERSON_SEPARATOR_CHARS = "、，,/和及与"
_MULTI_ENTITY_SPLIT_RE = re.compile(f"[{_PERSON_SEPARATOR_CHARS}\+\s]+")

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

_LIMIT_PATTERN = re.compile(r"(?:前|最近)?(\d{1,3})(?:条|个|项|行)")

_PRIORITY_HINTS = [
    ("p1", 1),
    ("p2", 2),
    ("p3", 3),
    ("最高优先级", 1),
    ("高优先级", 1),
    ("高优", 1),
    ("中优", 2),
    ("低优", 3),
]

_ORDER_ASC_HINTS = ("最早", "时间升序", "按创建顺序")
_ORDER_DESC_HINTS = ("最新", "最近", "按更新时间", "按优先级")

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

def _looks_like_person(token: str) -> bool:
    if not token:
        return False
    if len(token) < 2 or len(token) > 4:
        return False
    cjk_chars = sum(1 for ch in token if "\u4e00" <= ch <= "\u9fff")
    return cjk_chars >= len(token) - 1


def _extract_person_tokens_from_text(text: str) -> List[str]:
    persons: List[str] = []
    segments: List[str] = []
    for anchor in ("任务", "最近", "还有", "有哪些"):
        if anchor in text:
            segments.append(text.split(anchor, 1)[0])
    segments.append(text)
    for segment in segments:
        for token in _split_multi_values(segment):
            token = token.strip()
            for anchor in ("最近", "任务", "还有", "以及"):
                if anchor in token:
                    token = token.split(anchor, 1)[0]
                    break
            if _looks_like_person(token) and token not in persons:
                persons.append(token)
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


def _extract_tags(text: str) -> List[str]:
    return [match.group(1) for match in _TAG_PATTERN.finditer(text)]


def _build_multi_filter(field: str, values: List[str]) -> Optional[QueryFilter]:
    if not values:
        return None
    values = [v for v in values if v]
    if len(values) <= 1:
        return None
    return QueryFilter(field=field, op="in", values=values)


def _extract_person_task(text: str) -> Dict[str, Optional[str]]:
    """Return parsed person/task strings from a natural language question."""
    match = _PERSON_TASK_STATUS_RE.search(text)
    if match:
        return {
            "person": match.group("person").strip(),
            "task": match.group("task").strip(),
        }

    match = _PERSON_TASK_GENERIC_RE.search(text)
    if match:
        return {
            "person": match.group("person").strip(),
            "task": match.group("task").strip(),
        }

    # Pattern: "<person>现在有哪些任务"
    if "有哪些任务" in text:
        left, _, _ = text.partition("有哪些任务")
        left = left.strip()
        if left:
            return {"person": left, "task": None}

    return {"person": None, "task": None}


def _range_is_empty(value: Optional[TimeRange]) -> bool:
    return value is None or (not value.start and not value.end)


def _ensure_person_filter_from_text(spec: TaskQuerySpec, text: str) -> None:
    persons = _extract_person_tokens_from_text(text)
    if len(persons) < 2:
        return
    filters = list(getattr(spec, "filters", []) or [])
    has_person_filter = any(
        str(getattr(f, "field", "")).lower() == "person"
        and str(getattr(f, "op", "")).lower() == "in"
        and getattr(f, "values", None)
        for f in filters
    )
    if has_person_filter:
        return
    filters.append(QueryFilter(field="person", op="in", values=persons))
    spec.filters = filters
    spec.person = None


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

    intent = getattr(spec, "intent", None)
    status_kws = ("已完成", "未完成", "done", "todo", "搞定", "结束")
    status_kws_lower = tuple(kw.lower() for kw in status_kws)

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
    if not person_tokens_hint and entity_guess.get("person"):
        person_tokens_hint = [entity_guess["person"]]
    if not getattr(spec, "person", None) and len(person_tokens_hint) == 1:
        spec.person = person_tokens_hint[0]
    if not getattr(spec, "task", None):
        guessed_task = entity_guess.get("task")
        if guessed_task:
            spec.task = guessed_task.strip()
        elif getattr(spec, "task_keywords", None):
            first_kw = next((kw for kw in spec.task_keywords if kw), None)
            if first_kw:
                spec.task = first_kw

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

    # 1) coarse intent guess
    intent = TaskQueryIntent.task_status_single
    answer_mode = TaskAnswerMode.default
    if any(kw in text for kw in ("列表", "有哪些", "所有", "全部", "清单")):
        intent = TaskQueryIntent.task_status_list
    if "任务列表" in text or "有哪些" in text:
        intent = TaskQueryIntent.task_list_by_person

    completion_mode = any(kw in text for kw in _COMPLETION_TIME_HINTS)
    if completion_mode:
        intent = TaskQueryIntent.task_history

    # 2) coarse entity extraction
    entities = _extract_person_task(text)
    person = entities.get("person")
    task = entities.get("task") or text or None

    person_tokens = _split_multi_values(person)
    task_tokens = _split_multi_values(task)
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
    if completion_mode:
        answer_mode = TaskAnswerMode.completion_time_latest
        status = [TaskStatus.DONE]
    elif not status and "状态" in text and intent == TaskQueryIntent.task_status_single:
        status = [TaskStatus.DONE, TaskStatus.TODO]

    project = _detect_project(text)
    if project:
        filters.append(QueryFilter(field="project", op="eq", value=project))

    tags = _extract_tags(text)
    priority = _detect_priority(text)
    detected_limit = _detect_limit(text)
    time_range = _detect_time_range(text)

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

    table = "tasks" if spec.intent == TaskQueryIntent.task_history else "task_latest"
    target: Dict[str, Any] = {"table": table}

    filters: List[Dict[str, Any]] = []
    group_by: List[str] = []
    if spec.person:
        filters.append({"field": "person", "op": "eq", "value": spec.person})
    if spec.task:
        filters.append({"field": "task", "op": "eq", "value": spec.task})
    if spec.project:
        filters.append({"field": "project", "op": "eq", "value": spec.project})
    if spec.priority is not None:
        filters.append({"field": "priority", "op": "eq", "value": spec.priority})
    if spec.tags:
        for tag in spec.tags:
            filters.append({"field": "tags", "op": "like", "value": f"%{tag}%"})
    if spec.status:
        concrete = [
            s for s in spec.status if not isinstance(s, TaskStatus) or s != TaskStatus.ANY
        ]
        if concrete:
            filters.append(
                {
                    "field": "status",
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
                {"field": "ts", "op": "gte", "value": spec.time_range.start}
            )
        if spec.time_range.end:
            filters.append(
                {"field": "ts", "op": "lte", "value": spec.time_range.end}
            )
    if spec.due_range:
        if spec.due_range.start:
            filters.append(
                {"field": "due_ts", "op": "gte", "value": spec.due_range.start}
            )
        if spec.due_range.end:
            filters.append({"field": "due_ts", "op": "lte", "value": spec.due_range.end})
    if spec.created_range:
        if spec.created_range.start:
            filters.append(
                {"field": "created_ts", "op": "gte", "value": spec.created_range.start}
            )
        if spec.created_range.end:
            filters.append(
                {"field": "created_ts", "op": "lte", "value": spec.created_range.end}
            )
    if spec.filters:
        for flt in spec.filters:
            try:
                normalized = flt.to_plan_filter()
            except Exception:
                continue
            if normalized.get("field"):
                filters.append(normalized)

    if answer_mode == TaskAnswerMode.task_count_by_status:
        projections: List[str] = [
            "status",
            "COUNT(*) AS task_count",
        ]
        group_by = ["status"]
    elif answer_mode == TaskAnswerMode.person_summary_by_project:
        projections = [
            "project",
            "person",
            "status",
            "COUNT(*) AS task_count",
        ]
        group_by = ["project", "person", "status"]
    elif answer_mode == TaskAnswerMode.overdue_count_by_person:
        projections = [
            "person",
            "COUNT(*) AS overdue_count",
        ]
        group_by = ["person"]
    elif spec.intent in (
        TaskQueryIntent.task_status_single,
        TaskQueryIntent.task_status_list,
        TaskQueryIntent.task_list_by_person,
        TaskQueryIntent.task_history,
    ):
        projections = [
            "id",
            "person",
            "task",
            "status",
            "ts",
            "project",
            "tags",
            "priority",
            "due_ts",
            "created_ts",
            "updated_ts",
            "status_note",
        ]
    elif spec.intent == TaskQueryIntent.person_summary:
        projections = [
            "person",
            "status",
            "COUNT(*) AS task_count",
        ]
        group_by = ["person", "status"]
    else:
        projections = ["*"]

    sort: List[Dict[str, Any]] = [
        {
            "field": ob.field,
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
            {"field": "status", "direction": "ASC"},
        ]
    elif answer_mode == TaskAnswerMode.person_summary_by_project:
        sort = [
            {"field": "project", "direction": "ASC"},
            {"field": "person", "direction": "ASC"},
            {"field": "status", "direction": "ASC"},
        ]
    elif answer_mode == TaskAnswerMode.overdue_count_by_person:
        sort = [
            {"field": "overdue_count", "direction": "DESC"},
            {"field": "person", "direction": "ASC"},
        ]
    elif not sort:
        if spec.intent == TaskQueryIntent.person_summary:
            sort = [
                {"field": "task_count", "direction": "DESC"},
                {"field": "person", "direction": "ASC"},
            ]
        else:
            # Default to latest-first on ts, then id for deterministic ordering.
            sort = [
                {"field": "ts", "direction": "DESC"},
                {"field": "priority", "direction": "ASC"},
                {"field": "id", "direction": "DESC"},
            ]

    limit = spec.limit or 10

    return {
        "intent": intent,
        "target": target,
        "filters": filters,
        "projections": projections,
        "group_by": group_by,
        "sort": sort,
        "limit": limit,
    }




