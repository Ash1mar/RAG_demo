from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Sequence, Tuple

from app.services.nl2sql_engine import (
    TaskQuerySpec,
    TaskQueryIntent,
    TaskStatus,
    OrderBySpec,
)


class TaskSqlCompileError(Exception):
    """Raised when a TaskQuerySpec cannot be compiled into a safe SQL."""


@dataclass
class CompiledSql:
    sql: str
    params: Tuple[Any, ...]


def _build_status_clause(status_list: Sequence[TaskStatus], params: List[Any]) -> str:
    if not status_list:
        return ""
    # Map ANY -> no filter; others to concrete status values.
    concrete = [s for s in status_list if s != TaskStatus.ANY]
    if not concrete:
        return ""
    if len(concrete) == 1:
        params.append(concrete[0].value)
        return "status = ?"
    placeholders = ", ".join("?" for _ in concrete)
    params.extend(s.value for s in concrete)
    return f"status IN ({placeholders})"


def _build_order_by(order_by: Sequence[OrderBySpec]) -> str:
    if not order_by:
        return "ORDER BY ts DESC, id DESC"
    pieces: List[str] = []
    allowed_fields = {"id", "person", "task", "status", "ts"}
    for spec in order_by:
        field = spec.field.strip()
        if field not in allowed_fields:
            continue
        direction = spec.direction.value.lower()
        if direction not in ("asc", "desc"):
            direction = "desc"
        pieces.append(f"{field} {direction.upper()}")
    if not pieces:
        return "ORDER BY ts DESC, id DESC"
    return "ORDER BY " + ", ".join(pieces)


def compile_tasks_sql(spec: TaskQuerySpec) -> CompiledSql:
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
    params: List[Any] = []

    # ---- Single task status: person + task -> latest row ----
    if intent == TaskQueryIntent.task_status_single:
        person = (spec.person or "").strip()
        task = (spec.task or "").strip()
        if not person or not task:
            raise TaskSqlCompileError("task_status_single requires both person and task")
        sql = (
            "SELECT id, person, task, status, ts "
            "FROM tasks WHERE person = ? AND task = ? "
            "ORDER BY ts DESC, id DESC LIMIT 1"
        )
        params.extend([person, task])
        return CompiledSql(sql=sql, params=tuple(params))

    # ---- Task list by person / general list ----
    if intent in (TaskQueryIntent.task_status_list, TaskQueryIntent.task_list_by_person):
        clauses: List[str] = []

        if spec.person:
            clauses.append("person = ?")
            params.append(spec.person.strip())

        if spec.task:
            # For now keep equality; later we can extend to LIKE / token‑based matching.
            clauses.append("task = ?")
            params.append(spec.task.strip())

        status_clause = _build_status_clause(spec.status, params)
        if status_clause:
            clauses.append(status_clause)

        where = ""
        if clauses:
            where = " WHERE " + " AND ".join(clauses)

        order_by = _build_order_by(spec.order_by)

        limit = spec.limit if spec.limit is not None else 100
        try:
            limit_int = max(1, min(int(limit), 1000))
        except Exception as exc:  # pragma: no cover - defensive
            raise TaskSqlCompileError("invalid limit in TaskQuerySpec") from exc

        sql = (
            "SELECT id, person, task, status, ts "
            f"FROM tasks{where} "
            f"{order_by} "
            "LIMIT ?"
        )
        params.append(limit_int)
        return CompiledSql(sql=sql, params=tuple(params))

    # ---- Unknown or unsupported intent ----
    raise TaskSqlCompileError(f"unsupported intent for tasks SQL compile: {intent}")

