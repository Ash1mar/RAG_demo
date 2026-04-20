from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set


@dataclass(frozen=True)
class TasksSchemaConfig:
    """Configurable schema contract for the tasks SQLite backend.

    The codebase historically assumes two relations:
    - latest_relation: deduped latest row per (person, task)
    - history_relation: raw event/history rows

    To keep the project portable across business datasets, these relation names
    are configurable while the *logical* column contract stays stable
    (person/task/status/ts/project/tags/priority/due_ts/created_ts/...).
    """

    latest_relation: str = "task_latest"
    history_relation: str = "tasks"
    allowed_relations: Sequence[str] = ("task_latest", "tasks")
    field_map: Dict[str, str] = None  # logical_field -> physical_column

    def allowed_relation_set(self) -> Set[str]:
        return {str(r).strip() for r in self.allowed_relations if str(r).strip()}

    def translate_field(self, logical: str) -> str:
        """Translate a logical field name into a physical column name.

        The project aims to keep the *logical* contract stable (person/task/...)
        while allowing datasets to rename columns. Most deployments should keep
        using compatibility views so the defaults work without configuration.
        """

        name = (logical or "").strip()
        if not name:
            return name
        mapping = self.field_map or {}
        return (mapping.get(name) or name).strip() or name

    def logical_field_for(self, physical: str) -> str:
        """Translate a physical column name back to its logical field name."""
        name = (physical or "").strip()
        if not name:
            return name
        mapping = self.field_map or {}
        norm = _normalize_ident(name)
        for logical, mapped in mapping.items():
            if _normalize_ident(mapped) == norm:
                return logical
        return name


def _parse_csv_list(value: Optional[str]) -> List[str]:
    if not value:
        return []
    parts = []
    for token in value.split(","):
        t = token.strip()
        if t:
            parts.append(t)
    return parts


def _default_field_map() -> Dict[str, str]:
    # Logical contract used throughout the repo.
    return {
        "id": "id",
        "person": "person",  # executor / assignee
        "task": "task",
        "status": "status",
        "ts": "ts",
        "project": "project",
        "tags": "tags",
        "priority": "priority",
        "due_ts": "due_ts",
        "created_ts": "created_ts",
        "updated_ts": "updated_ts",
        "status_note": "status_note",
        "description": "description",
        # Enterprise extensions (optional).
        "owner": "owner",
        "owner_code": "owner_code",
        "owner_name": "owner_name",
        "created_by": "created_by",
        "created_by_name": "created_by_name",
        "created_by_org_code": "created_by_org_code",
        "org_name": "org_name",
        "division_name": "division_name",
        "division_code": "division_code",
        "post_name": "post_name",
        "post_code": "post_code",
        "is_read": "is_read",
        "is_delegated": "is_delegated",
        "task_id": "task_id",
    }


def _normalize_ident(name: Optional[str]) -> str:
    text = str(name or "").strip()
    if not text:
        return ""
    text = text.strip("[]")
    if "." in text:
        text = text.split(".")[-1]
    return text.strip("[]").lower()


def get_tasks_schema_config() -> TasksSchemaConfig:
    """Load schema config from environment variables.

    Defaults preserve existing behavior so old tests remain unchanged.
    """

    latest = os.getenv("TASKS_LATEST_RELATION", "task_latest").strip() or "task_latest"
    history = os.getenv("TASKS_HISTORY_RELATION", "tasks").strip() or "tasks"
    allowed_raw = os.getenv("TASKS_ALLOWED_RELATIONS", "").strip()

    allowed = _parse_csv_list(allowed_raw)
    if not allowed:
        allowed = [latest, history]

    # Ensure latest/history are included even if user forgot.
    allowed_set = {a for a in allowed if a}
    if latest not in allowed_set:
        allowed.append(latest)
    if history not in allowed_set:
        allowed.append(history)

    field_map = _default_field_map()
    # Optional per-field overrides, for portability across datasets without views.
    for logical_key in list(field_map.keys()):
        env_key = f"TASKS_COL_{logical_key.upper()}"
        raw = os.getenv(env_key, "").strip()
        if raw:
            field_map[logical_key] = raw

    return TasksSchemaConfig(
        latest_relation=latest,
        history_relation=history,
        allowed_relations=tuple(allowed),
        field_map=field_map,
    )
