from __future__ import annotations

from enum import Enum
from os import getenv
from typing import Any, Dict, List, Optional
import logging

from pydantic import BaseModel, Field

from app.services.llm_client import get_llm_client


class TaskQueryIntent(str, Enum):
    """Semantic intent type for task queries."""

    task_status_single = "task_status_single"
    task_status_list = "task_status_list"
    task_list_by_person = "task_list_by_person"
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


def parse_task_query_nl(q: str) -> TaskQuerySpec:
    """Parse a natural-language task query into TaskQuerySpec.

    Behavior:
    - By default uses a light-weight rule-based parser (no DB / no SQL).
    - When TASKS_NL2SQL_LLM=1, it first tries to use LLM via get_llm_client().
      On any failure it falls back to the rule-based parser.
    """

    text = (q or "").strip()
    if not text:
        return TaskQuerySpec(intent=TaskQueryIntent.unknown, raw_query=q)

    llm_error: Optional[Exception] = None

    if _USE_LLM_FOR_NL2SQL:
        try:
            client = get_llm_client()
            raw = client.generate_task_query_spec(text)
            spec = TaskQuerySpec.parse_obj(raw)
            if not spec.raw_query:
                spec.raw_query = q
            spec.extra.setdefault("nl2sql_source", "llm")
            return spec
        except Exception as exc:
            llm_error = exc
            logging.exception(
                "TASKS_NL2SQL LLM parse failed; falling back to rule-based parser"
            )

    spec = _rule_based_parse_task_query_nl(text)
    spec.raw_query = q
    spec.extra.setdefault("nl2sql_source", "rules")
    if llm_error is not None:
        spec.extra.setdefault("nl2sql_llm_error", str(llm_error))
    return spec


def _rule_based_parse_task_query_nl(q: str) -> TaskQuerySpec:
    """Very simple rule-based NL→JSON parser for bootstrapping."""

    text = q.strip()

    # 1) Rough intent detection
    intent = TaskQueryIntent.task_status_single
    if any(kw in text for kw in ("列表", "有哪些", "所有", "全部")):
        intent = TaskQueryIntent.task_status_list
    if any(kw in text for kw in ("张三", "李四", "老王", "老张")) and "有哪些" in text:
        intent = TaskQueryIntent.task_list_by_person

    # 2) Rough person & task extraction
    person: Optional[str] = None
    task: Optional[str] = None

    if "的" in text:
        # Example: 张三的E3D接口联调现在什么状态？
        left, _, right = text.partition("的")
        if left:
            person = left.strip()
        task = right.strip() or None
    else:
        task = text or None

    # 3) Rough status hints
    status: List[TaskStatus] = []
    if any(kw in text for kw in ("完成了吗", "完成了没", "搞定了没", "搞定没有", "done")):
        status = [TaskStatus.DONE]
    elif any(kw in text for kw in ("未完成", "没完成", "还没", "待办", "todo")):
        status = [TaskStatus.TODO]

    # 4) Default order & limit
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

