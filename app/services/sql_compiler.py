from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import os
import re
from typing import Any, Dict, List, Tuple, Optional

from app.services.nl2sql_engine import TaskQuerySpec, TaskQueryIntent, TaskStatus, build_task_query_plan
from app.sql_builder import build_sql_from_ir


class TaskSqlCompileError(Exception):
    """Raised when a TaskQuerySpec cannot be compiled into a safe SQL."""


@dataclass
class CompiledSql:
    sql: str
    params: Tuple[Any, ...]


def _resolve_symbolic_time_param(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    token = value.strip().lower()
    if not token:
        return value
    now = datetime.now(timezone.utc)
    if token == "now":
        return int(now.timestamp() * 1000)
    if token == "start_of_week":
        start = now - timedelta(days=now.weekday())
        return int(start.replace(hour=0, minute=0, second=0, microsecond=0).timestamp() * 1000)
    if token == "end_of_week":
        start = now - timedelta(days=now.weekday())
        end = start.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=7, seconds=-1)
        return int(end.timestamp() * 1000)
    if token == "start_of_month":
        return int(now.replace(day=1, hour=0, minute=0, second=0, microsecond=0).timestamp() * 1000)
    if token == "end_of_month":
        if now.month == 12:
            next_month = now.replace(year=now.year + 1, month=1, day=1)
        else:
            next_month = now.replace(month=now.month + 1, day=1)
        end = next_month.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(seconds=1)
        return int(end.timestamp() * 1000)
    match = re.fullmatch(r"now-(\d+)([dwm])", token)
    if match:
        amount = int(match.group(1))
        unit = match.group(2)
        days = amount * 7 if unit == "w" else amount * 30 if unit == "m" else amount
        return int((now - timedelta(days=days)).timestamp() * 1000)
    match = re.fullmatch(r"now\+(\d+)([dwm])", token)
    if match:
        amount = int(match.group(1))
        unit = match.group(2)
        days = amount * 7 if unit == "w" else amount * 30 if unit == "m" else amount
        return int((now + timedelta(days=days)).timestamp() * 1000)
    return value


def _build_params_from_plan(ir: Dict[str, Any]) -> Tuple[Any, ...]:
    """Derive positional parameters for a query-plan IR.

    The logic mirrors how `build_sql_from_ir` expands filters and LIMIT:
    - eq/like/gte/lte -> single value
    - in -> one value per list element
    - between -> two values (start, end)
    - LIMIT -> a single integer (clamped by caller)
    """
    params: List[Any] = []

    filters = ir.get("filters") or []
    for f in filters:
        op = str(f.get("op", "eq")).lower()
        value = f.get("value")
        if op == "between":
            if isinstance(value, (list, tuple)) and len(value) == 2:
                params.extend(_resolve_symbolic_time_param(item) for item in value)
        elif op == "in":
            if isinstance(value, (list, tuple)):
                params.extend(value)
        elif op == "exists":
            continue
        else:
            params.append(_resolve_symbolic_time_param(value))

    # LIMIT is always represented as a positional parameter when present.
    limit = ir.get("limit")
    if limit is not None:
        params.append(limit)

    return tuple(params)


def _spec_has_field_filter(spec: TaskQuerySpec, field: str) -> bool:
    if getattr(spec, field, None):
        return True
    filters = getattr(spec, "filters", None) or []
    for flt in filters:
        try:
            flt_field = str(getattr(flt, "field", "")).lower()
        except AttributeError:
            continue
        if flt_field == field:
            return True
    return False


def _default_dialect() -> str:
    raw = os.getenv("TASKS_DIALECT")
    if raw:
        return raw.strip().lower() or "sqlite"
    backend = os.getenv("TASKS_BACKEND", "sqlite").strip().lower()
    return "mssql" if backend == "mssql" else "sqlite"


def compile_tasks_sql(spec: TaskQuerySpec, *, dialect: Optional[str] = None) -> CompiledSql:
    """Compile TaskQuerySpec (NL semantic IR) into a safe, read‑only SQL for `tasks` table.

    Notes:
    - Only generates `SELECT ... FROM tasks ...` queries; no writes, no cross‑table access.
    - Uses positional parameters (`?`) to avoid SQL injection; caller must pass params as a tuple.
    - Currently supports:
        * task_status_single: lookup latest row by person + task.
        * task_status_list: filter by any combination of person / task / status, with ORDER BY + LIMIT.
        * task_list_by_person: list tasks by person (internally类似于 task_status_list).
    - If the spec is obviously incomplete or intent is unknown, raises TaskSqlCompileError.
    """

    intent = spec.intent
    dialect_norm = (dialect or _default_dialect()).strip().lower()

    # ---- Single task status: person + task -> latest row ----
    if intent == TaskQueryIntent.task_status_single:
        person = (spec.person or "").strip()
        task = (spec.task or "").strip()
        if not person or not task:
            raise TaskSqlCompileError("task_status_single requires both person and task")
        # Force single-row semantics via limit=1 on the plan.
        spec.limit = 1
        ir = build_task_query_plan(spec)
        sql = build_sql_from_ir(ir, dialect=dialect_norm)
        params = list(_build_params_from_plan(ir))
        # Clamp limit param to 1 explicitly.
        if params:
            params[-1] = 1
        return CompiledSql(sql=sql, params=tuple(params))

    # ---- Task list by person / general list ----
    if intent in (
        TaskQueryIntent.task_status_list,
        TaskQueryIntent.task_list_by_person,
        TaskQueryIntent.task_history,
    ):
        # Respect spec.limit when present; otherwise use a conservative default.
        default_limit = 100 if intent != TaskQueryIntent.task_history else 200
        raw_limit = spec.limit if spec.limit is not None else default_limit
        try:
            limit_int = max(1, min(int(raw_limit), 1000))
        except Exception as exc:  # pragma: no cover - defensive
            raise TaskSqlCompileError("invalid limit in TaskQuerySpec") from exc

        spec.limit = limit_int
        ir = build_task_query_plan(spec)
        sql = build_sql_from_ir(ir, dialect=dialect_norm)
        params = list(_build_params_from_plan(ir))
        # Last param is the LIMIT positional value; ensure it matches the clamped limit.
        if params:
            params[-1] = limit_int
        return CompiledSql(sql=sql, params=tuple(params))

    if intent == TaskQueryIntent.person_summary:
        if not _spec_has_field_filter(spec, "person"):
            raise TaskSqlCompileError("person_summary requires at least one person scope")
        raw_limit = spec.limit if spec.limit is not None else 100
        try:
            limit_int = max(1, min(int(raw_limit), 500))
        except Exception as exc:
            raise TaskSqlCompileError("invalid limit in TaskQuerySpec") from exc
        spec.limit = limit_int
        ir = build_task_query_plan(spec)
        sql = build_sql_from_ir(ir, dialect=dialect_norm)
        params = list(_build_params_from_plan(ir))
        if params:
            params[-1] = limit_int
        return CompiledSql(sql=sql, params=tuple(params))

    # ---- Unknown or unsupported intent ----
    raise TaskSqlCompileError(f"unsupported intent for tasks SQL compile: {intent}")


def compile_tasks_sql_v2(spec: TaskQuerySpec, *, dialect: Optional[str] = None) -> CompiledSql:
    """
    Multi-table-ready compiler entry point.

    Today this is a thin wrapper around `compile_tasks_sql(...)` and therefore
    generates the same single-table / single-view SQL (against `tasks` and
    `task_latest`).

    In the future, this function will be the place where we plug in a richer
    IR→plan→SQL pipeline that can:
    - join `tasks` with dimension tables such as `persons`, `projects`, `tags`, etc.;
    - keep the TaskQuerySpec IR and KG-lite resolution logic stable;
    - evolve the physical schema without touching the NL→IR layer.

    Callers that want to opt into the multi-table / multi-view world should use
    this v2 entry point instead of `compile_tasks_sql(...)`.
    """
    # For now, delegate to the v1 single-table implementation to avoid any
    # behavioral changes.
    return compile_tasks_sql(spec, dialect=dialect)
