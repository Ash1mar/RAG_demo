from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import sqlite3

from app.tasks_store.base import TasksStore


@dataclass
class SQLiteTasksConfig:
    db_path: str = "data/tasks.db"  # default location under project data/


class SQLiteTasksStore(TasksStore):
    """Read-only SQLite-backed tasks store.

    Schema (table `tasks`):
        id INTEGER PRIMARY KEY AUTOINCREMENT
        person TEXT NOT NULL
        task TEXT NOT NULL
        status TEXT NOT NULL -- expected values: 'DONE' | 'TODO'
        ts INTEGER NOT NULL  -- epoch millis
    """

    def __init__(self, config: Optional[SQLiteTasksConfig] = None) -> None:
        cfg = config or SQLiteTasksConfig()
        self._db_path = Path(cfg.db_path)
        self._conn: Optional[sqlite3.Connection] = None

    # --- internal helpers ---
    def _connect_ro(self) -> sqlite3.Connection:
        if self._conn is not None:
            return self._conn
        # Use SQLite URI to open database in read-only mode
        # Convert to forward-slashes to be URI-friendly across OSes
        abs_path = self._db_path.resolve()
        uri = f"file:{abs_path.as_posix()}?mode=ro"
        self._conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        return self._conn

    # --- TasksStore API ---
    def ready(self) -> bool:
        try:
            conn = self._connect_ro()
            cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='tasks'")
            row = cur.fetchone()
            return bool(row)
        except Exception:
            return False

    def get_latest_status(self, person: str, task: str) -> Optional[Dict[str, Any]]:
        conn = self._connect_ro()
        sql = (
            "SELECT id, person, task, status, ts "
            "FROM tasks WHERE person = ? AND task = ? "
            "ORDER BY ts DESC, id DESC LIMIT 1"
        )
        row = conn.execute(sql, (person, task)).fetchone()
        if not row:
            return None
        return {
            "id": int(row["id"]),
            "person": row["person"],
            "task": row["task"],
            "status": row["status"],
            "ts": int(row["ts"]),
        }

    def search(self, *, person: Optional[str] = None, task: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        conn = self._connect_ro()
        clauses: List[str] = []
        params: List[Any] = []
        if person:
            clauses.append("person = ?")
            params.append(person)
        if task:
            clauses.append("task = ?")
            params.append(task)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = f"SELECT id, person, task, status, ts FROM tasks{where} ORDER BY ts DESC, id DESC LIMIT ?"
        params.append(int(limit))
        rows = conn.execute(sql, params).fetchall()
        return [
            {
                "id": int(r["id"]),
                "person": r["person"],
                "task": r["task"],
                "status": r["status"],
                "ts": int(r["ts"]),
            }
            for r in rows
        ]

    def list_persons(self) -> List[str]:
        conn = self._connect_ro()
        rows = conn.execute("SELECT DISTINCT person FROM tasks").fetchall()
        return [str(r[0]) for r in rows]

    def list_tasks(self) -> List[str]:
        conn = self._connect_ro()
        rows = conn.execute("SELECT DISTINCT task FROM tasks").fetchall()
        return [str(r[0]) for r in rows]
