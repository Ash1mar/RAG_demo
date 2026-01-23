from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Protocol

from app.services.nl2sql_engine import TaskAnswerMode, TaskQuerySpec


@dataclass
class AnswerContext:
    spec: TaskQuerySpec
    rows: List[Dict[str, Any]]
    person: Optional[str]
    task: Optional[str]
    person_filters_active: bool
    person_filter_values: List[str]
    low_conf: bool
    answer_mode: TaskAnswerMode
    format_ts: Callable[[int], str]


class TaskIntentHandler(Protocol):
    name: str

    def matches(self, spec: TaskQuerySpec, answer_mode: TaskAnswerMode) -> bool:
        ...

    def label(self, spec: TaskQuerySpec) -> Optional[str]:
        ...

    def build_answer(self, ctx: AnswerContext) -> Dict[str, Any]:
        ...
