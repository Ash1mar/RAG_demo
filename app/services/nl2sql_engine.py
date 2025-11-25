from __future__ import annotations

from enum import Enum
from os import getenv
from typing import Any, Dict, List, Optional
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




_TASK_COUNT_QUESTION_HINTS = (
    "\u8fd8\u6709\u591a\u5c11",
    "\u8fd8\u5269\u591a\u5c11",
    "\u5269\u4e0b\u591a\u5c11",
    "\u8fd8\u5269\u51e0",
    "\u8fd8\u6709\u51e0",
    "\u5269\u51e0",
    "\u6709\u591a\u5c11\u4efb\u52a1",
    "\u5269\u4f59\u591a\u5c11",
)

_TASK_COUNT_STATUS_HINTS = [
    (
        (
            "\u672a\u5b8c\u6210",
            "\u672a\u5b8c",
            "\u6ca1\u5b8c\u6210",
            "\u6ca1\u641e\u5b9a",
            "\u672a\u7ed3\u675f",
            "\u5f85\u529e",
            "todo",
        ),
        TaskStatus.TODO,
    ),
    (
        (
            "\u8fdb\u884c\u4e2d",
            "\u5728\u505a",
            "\u5904\u7406\u4e2d",
            "in progress",
        ),
        TaskStatus.IN_PROGRESS,
    ),
    (
        (
            "\u5361\u4f4f",
            "\u88ab\u5361",
            "blocked",
        ),
        TaskStatus.BLOCKED,
    ),
    (
        (
            "\u5df2\u5b8c\u6210",
            "\u5b8c\u6210\u4e86\u591a\u5c11",
            "done",
        ),
        TaskStatus.DONE,
    ),
]

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




def _detect_task_count_mode(text: str) -> Optional[List[TaskStatus]]:
    if not text:
        return None
    if not any(kw in text for kw in _TASK_COUNT_QUESTION_HINTS):
        return None
    statuses: List[TaskStatus] = []
    for hints, status in _TASK_COUNT_STATUS_HINTS:
        if any(h in text for h in hints):
            if status not in statuses:
                statuses.append(status)
    if not statuses:
        statuses = [TaskStatus.TODO, TaskStatus.IN_PROGRESS]
    return statuses


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
    if not t:
        return

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
    status_kws = ("完成", "未完成", "done", "todo", "搞定", "结束")

    if intent == TaskQueryIntent.task_list_by_person:
        if not any(kw in t for kw in status_kws):
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

    count_statuses = _detect_task_count_mode(t)
    if count_statuses:
        spec.intent = TaskQueryIntent.task_status_list
        spec.answer_mode = TaskAnswerMode.task_count_by_status
        existing_statuses = []
        if getattr(spec, "status", None):
            existing_statuses = list(spec.status)
        merged: List[TaskStatus] = []
        seen = set()
        for item in existing_statuses + count_statuses:
            try:
                enum_item = item if isinstance(item, TaskStatus) else TaskStatus(str(item))
            except Exception:
                continue
            if enum_item not in seen:
                merged.append(enum_item)
                seen.add(enum_item)
        spec.status = merged or count_statuses
        if spec.limit is None:
            spec.limit = max(len(count_statuses), len(TaskStatus.__members__), 4)
        else:
            try:
                limit_val = int(spec.limit)
            except Exception:
                limit_val = 0
            min_limit = max(len(count_statuses), len(TaskStatus.__members__), 4)
            if limit_val < min_limit:
                spec.limit = min_limit
        spec.order_by = []

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
    if any(kw in text for kw in ("完成了吗", "完成了没", "搞定了没", "搞定没有", "done")):
        status = [TaskStatus.DONE]
    elif any(kw in text for kw in ("未完成", "没完成", "还没", "待办", "todo")):
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
        if any(kw in text for kw in kws):
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




