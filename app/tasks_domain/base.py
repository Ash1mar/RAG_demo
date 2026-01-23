from __future__ import annotations

from typing import Any, Dict, Optional

from app.tasks_schema import TasksSchemaConfig


class TaskDomain:
    name = "tasks"

    def rewrite_text2sql(
        self,
        sql: str,
        *,
        hint: Optional[Dict[str, Any]],
        question: str,
        schema: TasksSchemaConfig,
    ) -> str:
        return sql
