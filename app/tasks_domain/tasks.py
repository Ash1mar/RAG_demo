from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from app.tasks_domain.base import TaskDomain
from app.tasks_schema import TasksSchemaConfig


class TasksDomain(TaskDomain):
    name = "tasks"

    def rewrite_text2sql(
        self,
        sql: str,
        *,
        hint: Optional[Dict[str, Any]],
        question: str,
        schema: TasksSchemaConfig,
    ) -> str:
        if not sql:
            return sql

        updated = sql
        hint = hint or {}

        project_col = schema.translate_field("project") or "project"
        tags_col = schema.translate_field("tags") or "tags"
        status_col = schema.translate_field("status") or "status"
        priority_col = schema.translate_field("priority") or "priority"
        task_col = schema.translate_field("task") or "task"

        flow_value = _extract_flow_value(question or "")
        project_hint = hint.get("project")
        if not project_hint and flow_value:
            project_hint = flow_value

        updated = _rewrite_flow_filter(
            updated,
            question=question or "",
            project_hint=project_hint,
            project_column=project_col,
            tags_column=tags_col,
        )
        updated = _rewrite_status_overconstraint(
            updated,
            question=question or "",
            status_hint=hint.get("status"),
            status_column=status_col,
        )
        updated = _ensure_priority_filter(
            updated,
            hint.get("priority"),
            hint.get("task"),
            priority_column=priority_col,
            task_column=task_col,
        )
        return updated


def _ensure_priority_filter(
    sql: str,
    priority: Optional[Any],
    task_hint: Optional[Any],
    *,
    priority_column: str = "priority",
    task_column: str = "task",
) -> str:
    priority_col = (priority_column or "priority").strip()
    task_col = (task_column or "task").strip()
    if not priority_col:
        return sql

    def _normalize_priority_literals(text: str) -> Tuple[str, bool]:
        pattern = re.compile(
            rf"{re.escape(priority_col)}\s*(?:in\s*\([^\)]*\)|=\s*'[^']*')",
            re.IGNORECASE,
        )
        changed = False

        def _repl(match: re.Match) -> str:
            nonlocal changed
            chunk = match.group(0)
            if re.search(r"p\s*1|高优|高優", chunk, re.IGNORECASE):
                changed = True
                return f"{priority_col} = 1"
            return chunk

        new_text = pattern.sub(_repl, text)
        return new_text, changed

    sql, normalized_priority = _normalize_priority_literals(sql)
    lowered = sql.lower()

    if normalized_priority:
        return sql

    p_val: Optional[int] = None
    if priority is not None:
        try:
            p_val = int(priority)
        except (TypeError, ValueError):
            p_val = None
    if p_val is None and ("高优" in sql and "p1" in lowered):
        p_val = 1
    if p_val is None:
        return sql

    if re.search(rf"\b{re.escape(priority_col.lower())}\b", lowered):
        return sql
    clause = f"{priority_col} = {p_val}"

    if "高优" in sql and "p1" in lowered:
        if task_col:
            pattern_and = re.compile(
                rf"\s+and\s+{re.escape(task_col)}\s*(?:=|like)\s*'%[^']*高优[^']*p1[^']*%?'\s*",
                re.IGNORECASE,
            )
            sql = pattern_and.sub(" ", sql)

    lowered = sql.lower()
    where_idx = lowered.find(" where ")
    if where_idx != -1:
        insert_pos = where_idx + len(" where ")
        existing = sql[insert_pos:].strip()
        if existing:
            sql = f"{sql[:insert_pos]}({clause}) AND ({existing})"
        else:
            sql = f"{sql[:insert_pos]}({clause})"
        return sql

    order_idx = lowered.find(" order by ")
    limit_idx = lowered.find(" limit ")
    insert_pos = len(sql)
    for idx in (order_idx, limit_idx):
        if idx != -1 and idx < insert_pos:
            insert_pos = idx
    suffix = sql[insert_pos:]
    prefix = sql[:insert_pos]
    if suffix.strip():
        sql = f"{prefix} WHERE ({clause}) {suffix.lstrip()}"
    else:
        sql = f"{prefix} WHERE ({clause})"
    return sql


def _inject_where_clause(sql: str, clause: str) -> str:
    if not sql or not clause:
        return sql
    lowered = sql.lower()
    where_idx = lowered.find(" where ")
    if where_idx != -1:
        insert_pos = where_idx + len(" where ")
        end_idx = len(sql)
        for keyword in (" order by ", " limit "):
            idx = lowered.find(keyword, insert_pos)
            if idx != -1 and idx < end_idx:
                end_idx = idx
        existing = sql[insert_pos:end_idx].strip()
        suffix = sql[end_idx:]
        if existing:
            return f"{sql[:insert_pos]}({clause}) AND ({existing}){suffix}"
        return f"{sql[:insert_pos]}({clause}){suffix}"

    order_idx = lowered.find(" order by ")
    limit_idx = lowered.find(" limit ")
    insert_pos = len(sql)
    for idx in (order_idx, limit_idx):
        if idx != -1 and idx < insert_pos:
            insert_pos = idx
    suffix = sql[insert_pos:]
    prefix = sql[:insert_pos]
    if suffix.strip():
        return f"{prefix} WHERE ({clause}) {suffix.lstrip()}"
    return f"{prefix} WHERE ({clause})"


def _rewrite_flow_filter(
    sql: str,
    *,
    question: str,
    project_hint: Optional[str],
    project_column: str = "project",
    tags_column: str = "tags",
) -> str:
    if not sql:
        return sql
    project = (project_hint or "").strip()
    if not project:
        return sql
    if "flow_name" not in (question or "") and "流程" not in (question or ""):
        return sql

    lowered = sql.lower()
    project_col = (project_column or "project").strip()
    tags_col = (tags_column or "tags").strip()
    if not project_col or not tags_col:
        return sql
    if re.search(rf"\b{re.escape(project_col.lower())}\s*(=|like|in)\b", lowered):
        return sql

    escaped_project = project.replace("'", "''")
    escaped = re.escape(project)
    tags_like_pat = re.compile(
        rf"\b{re.escape(tags_col)}\s+like\s+(['\"])%{escaped}%\1", re.IGNORECASE
    )
    if tags_like_pat.search(sql):
        return tags_like_pat.sub(f"{project_col} = '{escaped_project}'", sql)

    return _inject_where_clause(sql, f"{project_col} = '{escaped_project}'")


def _remove_predicate(sql: str, predicate_pattern: re.Pattern[str]) -> str:
    if not sql:
        return sql
    updated = sql

    updated = re.sub(
        rf"\s+AND\s+\(?{predicate_pattern.pattern}\)?",
        "",
        updated,
        flags=re.IGNORECASE,
    )
    updated = re.sub(
        rf"\(?{predicate_pattern.pattern}\)?\s+AND\s+",
        "",
        updated,
        flags=re.IGNORECASE,
    )
    updated = re.sub(
        rf"\bWHERE\s+\(?{predicate_pattern.pattern}\)?\s*(?=(ORDER\s+BY|LIMIT|$))",
        "WHERE 1=1 ",
        updated,
        flags=re.IGNORECASE,
    )
    return updated


_FLOW_VALUE_PATTERN = re.compile(
    r"(?:流程|flow_name)\s*(?:（[^）]*）)?\s*(?:=|为)?\s*([^\s，。；;：:]+)",
    re.IGNORECASE,
)


def _extract_flow_value(question: str) -> Optional[str]:
    text = (question or "").strip()
    if not text:
        return None
    match = _FLOW_VALUE_PATTERN.search(text)
    if not match:
        return None
    value = (match.group(1) or "").strip()
    return value or None


def _rewrite_status_overconstraint(
    sql: str,
    *,
    question: str,
    status_hint: Optional[List[str]],
    status_column: str = "status",
) -> str:
    if not sql:
        return sql
    q = (question or "").lower()
    if "blocked" in q or "阻塞" in (question or ""):
        return sql
    status_col = (status_column or "status").strip()
    if not status_col:
        return sql
    pred = re.compile(rf"{re.escape(status_col)}\s*=\s*(['\"])BLOCKED\1", re.IGNORECASE)
    if not pred.search(sql):
        return sql
    return _remove_predicate(sql, pred)
