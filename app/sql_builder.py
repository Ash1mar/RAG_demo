from __future__ import annotations

from typing import Any, Dict, List, Tuple


def _render_filter(field: str, op: str, value: Any) -> str:
    """Render a single filter to a SQL WHERE fragment with placeholders.

    This helper keeps only the SQL *shape* here (e.g. "field = ?"); the
    concrete parameter values are expected to be bound by the caller.
    """
    op = op.lower()
    if op == "eq":
        return f"{field} = ?"
    if op == "like":
        return f"{field} LIKE ?"
    if op == "gte":
        return f"{field} >= ?"
    if op == "lte":
        return f"{field} <= ?"
    if op == "between":
        # treated as two positional params in the caller
        return f"{field} BETWEEN ? AND ?"
    if op == "in":
        # value is expected to be an iterable; we only care about length
        if not isinstance(value, (list, tuple)) or not value:
            # fall back to a tautology when IN would be empty
            return "1=1"
        placeholders = ", ".join("?" for _ in value)
        return f"{field} IN ({placeholders})"
    # Fallback: raw op with a single placeholder
    return f"{field} {op} ?"


def _render_where(filters: List[Dict[str, Any]]) -> str:
    clauses: List[str] = []
    for f in filters:
        field = f.get("field")
        op = f.get("op", "eq")
        value = f.get("value")
        if not field:
            continue
        clauses.append(_render_filter(field, op, value))
    if not clauses:
        return ""
    return " WHERE " + " AND ".join(clauses)


def _render_order_by(sort: List[Dict[str, Any]]) -> str:
    if not sort:
        return ""
    parts: List[str] = []
    for s in sort:
        field = s.get("field")
        direction = str(s.get("direction", "")).upper() or "DESC"
        if not field:
            continue
        if direction not in ("ASC", "DESC"):
            direction = "DESC"
        parts.append(f"{field} {direction}")
    if not parts:
        return ""
    return " ORDER BY " + ", ".join(parts)


def build_sql_from_ir(ir: Dict[str, Any]) -> str:
    """Build a SQL string from a normalized query-plan IR.

    The IR is expected to follow the shape produced by
    `build_task_query_plan`:

    {
        "intent": "...",
        "target": {"table": "tasks"},
        "filters": [...],
        "projections": [...],
        "sort": [...],
        "limit": 10
    }

    For now the SQL is tailored to the SQLite `tasks` table and only
    covers simple read-only queries for task status / task list flows.
    """

    intent = str(ir.get("intent", "") or "")
    target = ir.get("target") or {}
    table = target.get("table", "tasks")
    filters = ir.get("filters") or []
    projections = ir.get("projections") or []
    sort = ir.get("sort") or []
    group_by = ir.get("group_by") or []
    limit = ir.get("limit")

    if not projections:
        projections = ["id", "person", "task", "status", "ts"]

    cols = ", ".join(projections)

    where_sql = _render_where(filters)
    group_sql = ""
    if group_by:
        group_sql = " GROUP BY " + ", ".join(group_by)
    order_sql = _render_order_by(sort)

    # Intent-level branching is intentionally minimal; both status queries
    # and task lists share the same basic SELECT/WHERE/ORDER/LIMIT shape.
    base = f"SELECT {cols} FROM {table}{where_sql}{group_sql}{order_sql}"

    # Always use a positional placeholder for LIMIT when present so that
    # callers can clamp and bind the concrete value themselves.
    if limit is not None:
        base += " LIMIT ?"

    return base
