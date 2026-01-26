from __future__ import annotations

import importlib
import os
from typing import Optional

from app.services.nl2sql_engine import TaskAnswerMode, TaskQuerySpec
from app.tasks_intent.base import TaskIntentHandler
from app.tasks_intent.handlers import HANDLERS as BUILTIN_HANDLERS


def _parse_csv_list(value: str) -> list[str]:
    parts: list[str] = []
    for token in (value or "").split(","):
        token = token.strip()
        if token:
            parts.append(token)
    return parts


def _load_handlers_from_module(module_name: str) -> list[TaskIntentHandler]:
    mod = importlib.import_module(module_name)
    handlers = getattr(mod, "HANDLERS", None)
    if isinstance(handlers, list) and handlers:
        return handlers
    get_handlers = getattr(mod, "get_handlers", None)
    if callable(get_handlers):
        loaded = get_handlers()
        if isinstance(loaded, list) and loaded:
            return loaded
    return []


def _get_registered_handlers() -> list[TaskIntentHandler]:
    handlers: list[TaskIntentHandler] = []

    pack = (os.getenv("TASKS_INTENT_PACK", "tasks") or "tasks").strip().lower()
    if pack and pack != "tasks":
        module_name = pack if "." in pack else f"app.tasks_intent.{pack}"
        try:
            handlers.extend(_load_handlers_from_module(module_name))
        except ModuleNotFoundError:
            pass

    extra_modules = _parse_csv_list(os.getenv("TASKS_INTENT_MODULES", ""))
    for module_name in extra_modules:
        try:
            handlers.extend(_load_handlers_from_module(module_name))
        except ModuleNotFoundError:
            continue

    handlers.extend(BUILTIN_HANDLERS)
    return handlers


def get_intent_handler(spec: TaskQuerySpec, answer_mode: TaskAnswerMode) -> TaskIntentHandler:
    handlers = _get_registered_handlers()
    for handler in handlers:
        if handler.matches(spec, answer_mode):
            return handler
    return handlers[-1] if handlers else BUILTIN_HANDLERS[-1]


def intent_label(spec: Optional[TaskQuerySpec]) -> str:
    if spec is None:
        return "unknown"
    for handler in _get_registered_handlers():
        label = handler.label(spec)
        if label:
            return label
    return "unknown"
