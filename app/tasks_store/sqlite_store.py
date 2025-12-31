from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import sqlite3

from app.tasks_store.base import TasksStore
from app.tasks_schema import TasksSchemaConfig, get_tasks_schema_config


@dataclass
class SQLiteTasksConfig:
    db_path: str = "data/tasks.db"  # default location under project data/
    timeout_sec: float = 2.0         # read-only connect timeout
    latest_relation: Optional[str] = None
    history_relation: Optional[str] = None
    allowed_relations: Optional[Sequence[str]] = None


class SQLiteTasksStore(TasksStore):
    """Read-only SQLite-backed tasks store.

    Default schema (view `task_latest` -> latest status per person+task):
        id INTEGER PRIMARY KEY AUTOINCREMENT
        person TEXT NOT NULL
        task TEXT NOT NULL
        status TEXT NOT NULL -- expected values: 'DONE' | 'TODO' | 'IN_PROGRESS' | 'BLOCKED'
        ts INTEGER NOT NULL  -- epoch millis, status timestamp
        project TEXT
        tags TEXT
        priority INTEGER
        due_ts INTEGER
        created_ts INTEGER
        updated_ts INTEGER
        status_note TEXT
    """

    def __init__(self, config: Optional[SQLiteTasksConfig] = None) -> None:
        cfg = config or SQLiteTasksConfig()
        self._db_path = Path(cfg.db_path)
        self._timeout = float(getattr(cfg, "timeout_sec", 2.0))
        self._conn: Optional[sqlite3.Connection] = None
        self._schema = self._resolve_schema(cfg)

    # --- metadata helpers ---
    def _has_table_or_view(self, name: str) -> bool:
        conn = self._connect_ro()
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE (type='table' OR type='view') AND name=?",
            (name,),
        )
        return cur.fetchone() is not None

    def _resolve_schema(self, cfg: SQLiteTasksConfig) -> TasksSchemaConfig:
        base = get_tasks_schema_config()
        latest = (cfg.latest_relation or base.latest_relation).strip()
        history = (cfg.history_relation or base.history_relation).strip()
        allowed = cfg.allowed_relations or base.allowed_relations
        # Normalize
        allowed_norm = tuple(str(a).strip() for a in allowed if str(a).strip())
        if not allowed_norm:
            allowed_norm = (latest, history)
        return TasksSchemaConfig(
            latest_relation=latest,
            history_relation=history,
            allowed_relations=allowed_norm,
        )

    def _latest_table(self) -> str:
        # Prefer configured latest relation when present; fallback to the legacy names.
        if self._has_table_or_view(self._schema.latest_relation):
            return self._schema.latest_relation
        if self._has_table_or_view("task_latest"):
            return "task_latest"
        if self._has_table_or_view(self._schema.history_relation):
            return self._schema.history_relation
        return "tasks"

    # --- internal helpers ---
    def _connect_ro(self) -> sqlite3.Connection:
        if self._conn is not None:
            return self._conn
        # Use SQLite URI to open database in read-only mode
        # Convert to forward-slashes to be URI-friendly across OSes
        abs_path = self._db_path.resolve()
        uri = f"file:{abs_path.as_posix()}?mode=ro"
        self._conn = sqlite3.connect(uri, uri=True, check_same_thread=False, timeout=self._timeout)
        self._conn.row_factory = sqlite3.Row
        return self._conn

    # --- TasksStore API ---
    def ready(self) -> bool:
        try:
            _ = self._connect_ro()
            return self._has_table_or_view(self._schema.history_relation) or self._has_table_or_view(
                self._schema.latest_relation
            )
        except Exception:
            return False

    def get_latest_status(self, person: str, task: str) -> Optional[Dict[str, Any]]:
        conn = self._connect_ro()
        table = self._latest_table()
        sql = (
            f"SELECT * FROM {table} WHERE person = ? AND task = ? "
            "ORDER BY ts DESC, id DESC LIMIT 1"
        )
        row = conn.execute(sql, (person, task)).fetchone()
        if not row:
            return None
        return self._row_to_dict(row)

    def search(
        self,
        *,
        person: Optional[str] = None,
        task: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        conn = self._connect_ro()
        table = self._latest_table()
        clauses: List[str] = []
        params: List[Any] = []
        if person:
            clauses.append("person = ?")
            params.append(person)
        if task:
            clauses.append("task = ?")
            params.append(task)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = f"SELECT * FROM {table}{where} ORDER BY ts DESC, id DESC LIMIT ?"
        params.append(int(limit))
        rows = conn.execute(sql, params).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def list_persons(self) -> List[str]:
        conn = self._connect_ro()
        table = self._latest_table()
        rows = conn.execute(f"SELECT DISTINCT person FROM {table}").fetchall()
        return [str(r[0]) for r in rows]

    def list_tasks(self) -> List[str]:
        conn = self._connect_ro()
        table = self._latest_table()
        rows = conn.execute(f"SELECT DISTINCT task FROM {table}").fetchall()
        return [str(r[0]) for r in rows]

    # --- generic read‑only query helper for NL→SQL ---
    def query(self, sql: str, params: Tuple[Any, ...]) -> List[Dict[str, Any]]:
        """Execute a read‑only SELECT on the tasks DB.

        This helper is intended for use with the NL→SQL compiler, which should
        only emit queries of the form `SELECT ... FROM task_latest|tasks ...` using `?` placeholders.
        """
        s = (sql or "").strip().lower()
        if not s.startswith("select"):
            raise ValueError("only SELECT statements are allowed in SQLiteTasksStore.query")
        allowed_targets = tuple(f" from {name.lower()}" for name in self._schema.allowed_relations)
        if not any(t in s for t in allowed_targets):
            raise ValueError(f"query must target one of: {', '.join(self._schema.allowed_relations)}")

        conn = self._connect_ro()
        cur = conn.execute(sql, params)
        rows = cur.fetchall()
        return [self._row_to_dict(r) for r in rows]

    # --- internal mapping ---
    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        """Normalize SQLite row to a dict with optional extended fields."""
        keys = set(row.keys())
        base: Dict[str, Any] = {}
        base["id"] = int(row["id"]) if "id" in keys and row["id"] is not None else None
        if "person" in keys:
            base["person"] = row["person"]
        if "task" in keys:
            base["task"] = row["task"]
        if "status" in keys:
            base["status"] = row["status"]
        if "ts" in keys:
            try:
                base["ts"] = int(row["ts"]) if row["ts"] is not None else None
            except (TypeError, ValueError):
                base["ts"] = row["ts"]
        # optional extended fields (absent in older schemas)
        for field in (
            "project",
            "tags",
            "priority",
            "due_ts",
            "created_ts",
            "updated_ts",
            "status_note",
            "description",
            "person_id",
        ):
            if field in row.keys():
                base[field] = row[field]
        return base
