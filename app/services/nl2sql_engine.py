from __future__ import annotations

from enum import Enum
from os import getenv
from typing import Any, Dict, List, Optional
import logging

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


class TaskStatus(str, Enum):
    """Task status enum corresponding to tasks.status."""

    DONE = "DONE"
    TODO = "TODO"
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


class TaskQuerySpec(BaseModel):
    """Intermediate representation for task query semantics."""

    intent: TaskQueryIntent = Field(
        TaskQueryIntent.task_status_single, description="Recognized query intent."
    )
    raw_query: str = Field(..., description="Original natural language query.")

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
    status: List[TaskStatus] = Field(
        default_factory=list,
        description="Required task status filters; empty means no restriction.",
    )

    time_range: Optional[TimeRange] = Field(
        None, description="Time range for the query, e.g. recent week/month."
    )
    order_by: List[OrderBySpec] = Field(
        default_factory=list, description="Order-by fields."
    )
    limit: Optional[int] = Field(
        10, ge=1, le=200, description="Max number of rows to return (default 10)."
    )

    extra: Dict[str, Any] = Field(
        default_factory=dict,
        description="Reserved for debug info, model scores, parse details, etc.",
    )


_USE_LLM_FOR_NL2SQL = getenv("TASKS_NL2SQL_LLM", "0") == "1"


def _post_process_intent(spec: TaskQuerySpec, text: str) -> None:
    """Apply lightweight intent heuristics on top of LLM/rule output.

    This is deliberately simple and only covers very common patterns in
    Chinese NL queries used in this demo.
    """
    t = text.strip()

    # 强信号：按人查任务列表。
    if "有哪些任务" in t or "任务列表" in t:
        spec.intent = TaskQueryIntent.task_list_by_person
        # 列表类问题，通常关注 DONE 和 TODO。
        if not spec.status:
            spec.status = [TaskStatus.DONE, TaskStatus.TODO]

    # 任务历史（目前保留，将来可用）。
    if "都是什么状态" in t or ("历史" in t and "状态" in t):
        spec.intent = TaskQueryIntent.task_history

    # “什么状态”结尾，且句子里能抽到人和任务 → 单条状态查询。
    if (
        t.endswith("什么状态？")
        or t.endswith("什么状态?")
        or "现在什么状态" in t
        or "是什么状态" in t
    ):
        if spec.person and spec.task:
            spec.intent = TaskQueryIntent.task_status_single
            # 单条状态查询聚焦 DONE/TODO；去掉 ANY。
            if not spec.status:
                spec.status = [TaskStatus.DONE, TaskStatus.TODO]
            else:
                spec.status = [
                    s
                    for s in spec.status
                    if not isinstance(s, TaskStatus) or s != TaskStatus.ANY
                ]
            # 鼓励单行语义。
            if spec.limit is None or spec.limit > 1:
                spec.limit = 1


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
    spec = _rule_based_parse_task_query_nl(text)
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


def build_task_query_plan(spec: TaskQuerySpec) -> Dict[str, Any]:
    """Convert TaskQuerySpec into a generic query plan dict."""
    intent = (
        spec.intent.value
        if isinstance(spec.intent, TaskQueryIntent)
        else str(spec.intent)
    )

    target: Dict[str, Any] = {"table": "tasks"}

    filters: List[Dict[str, Any]] = []
    if spec.person:
        filters.append({"field": "person", "op": "eq", "value": spec.person})
    if spec.task:
        filters.append({"field": "task", "op": "eq", "value": spec.task})
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

    if spec.intent in (
        TaskQueryIntent.task_status_single,
        TaskQueryIntent.task_status_list,
        TaskQueryIntent.task_list_by_person,
        TaskQueryIntent.task_history,
    ):
        projections: List[str] = ["id", "person", "task", "status", "ts"]
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
    if not sort:
        # Default to latest-first on ts, then id for deterministic ordering.
        sort = [
            {"field": "ts", "direction": "DESC"},
            {"field": "id", "direction": "DESC"},
        ]

    limit = spec.limit or 10

    return {
        "intent": intent,
        "target": target,
        "filters": filters,
        "projections": projections,
        "sort": sort,
        "limit": limit,
    }
