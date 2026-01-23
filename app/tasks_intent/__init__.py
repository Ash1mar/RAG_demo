from __future__ import annotations

from typing import Optional

from app.services.nl2sql_engine import TaskAnswerMode, TaskQuerySpec
from app.tasks_intent.base import TaskIntentHandler
from app.tasks_intent.handlers import HANDLERS


def get_intent_handler(spec: TaskQuerySpec, answer_mode: TaskAnswerMode) -> TaskIntentHandler:
    for handler in HANDLERS:
        if handler.matches(spec, answer_mode):
            return handler
    return HANDLERS[-1]


def intent_label(spec: Optional[TaskQuerySpec]) -> str:
    if spec is None:
        return "unknown"
    for handler in HANDLERS:
        label = handler.label(spec)
        if label:
            return label
    return "unknown"
