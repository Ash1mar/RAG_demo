from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pyodbc

from app.tasks_store.base import TasksStore
from app.tasks_schema import TasksSchemaConfig, get_tasks_schema_config


@dataclass(frozen=True)
class MSSQLTasksConfig:
    server: str = "127.0.0.1,1433"
    database: str = "fact_tasks"
    user: str = "sa"
    password: str = ""
    driver: str = "ODBC Driver 18 for SQL Server"
    encrypt: bool = True
    trust_server_certificate: bool = True
    timeout_sec: float = 5.0
    latest_relation: Optional[str] = None
    history_relation: Optional[str] = None
    allowed_relations: Optional[Sequence[str]] = None


def _split_relation(name: str) -> Tuple[Optional[str], str]:
    text = (name or "").strip()
    if not text:
        return None, text
    if "." in text:
        schema, table = text.split(".", 1)
        return schema.strip("[]"), table.strip("[]")
    return None, text.strip("[]")


class MSSQLTasksStore(TasksStore):
    """Read-only SQL Server-backed tasks store."""

    def __init__(self, config: Optional[MSSQLTasksConfig] = None) -> None:
        cfg = config or MSSQLTasksConfig()
        self._cfg = cfg
        self._timeout = float(getattr(cfg, "timeout_sec", 5.0))
        self._conn: Optional[pyodbc.Connection] = None
        self._schema = self._resolve_schema(cfg)

    def _connect_ro(self) -> pyodbc.Connection:
        if self._conn is not None:
            return self._conn
        driver = (self._cfg.driver or "ODBC Driver 18 for SQL Server").strip()
        encrypt = "yes" if self._cfg.encrypt else "no"
        trust = "yes" if self._cfg.trust_server_certificate else "no"
        conn_str = (
            f"Driver={{{driver}}};"
            f"Server={self._cfg.server};"
            f"Database={self._cfg.database};"
            f"UID={self._cfg.user};"
            f"PWD={self._cfg.password};"
            f"Encrypt={encrypt};"
            f"TrustServerCertificate={trust};"
        )
        self._conn = pyodbc.connect(
            conn_str,
            timeout=int(self._timeout),
            autocommit=True,
        )
        return self._conn

    def _has_table_or_view(self, name: str) -> bool:
        conn = self._connect_ro()
        schema, table = _split_relation(name)
        if not table:
            return False
        if schema:
            sql = (
                "SELECT 1 FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ? "
                "UNION ALL "
                "SELECT 1 FROM INFORMATION_SCHEMA.VIEWS WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ?"
            )
            params = (schema, table, schema, table)
        else:
            sql = (
                "SELECT 1 FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = ? "
                "UNION ALL "
                "SELECT 1 FROM INFORMATION_SCHEMA.VIEWS WHERE TABLE_NAME = ?"
            )
            params = (table, table)
        row = conn.cursor().execute(sql, params).fetchone()
        return row is not None

    def _resolve_schema(self, cfg: MSSQLTasksConfig) -> TasksSchemaConfig:
        base = get_tasks_schema_config()
        latest = (cfg.latest_relation or base.latest_relation).strip()
        history = (cfg.history_relation or base.history_relation).strip()
        allowed = cfg.allowed_relations or base.allowed_relations
        allowed_norm = tuple(str(a).strip() for a in allowed if str(a).strip())
        if not allowed_norm:
            allowed_norm = (latest, history)
        return TasksSchemaConfig(
            latest_relation=latest,
            history_relation=history,
            allowed_relations=allowed_norm,
        )

    def _latest_table(self) -> str:
        if self._has_table_or_view(self._schema.latest_relation):
            return self._schema.latest_relation
        if self._has_table_or_view("task_latest"):
            return "task_latest"
        if self._has_table_or_view(self._schema.history_relation):
            return self._schema.history_relation
        return "tasks"

    def _exec_select(self, sql: str, params: Tuple[Any, ...]) -> List[Dict[str, Any]]:
        conn = self._connect_ro()
        cur = conn.cursor()
        cur.execute(sql, params)
        columns = [col[0] for col in cur.description]
        return [dict(zip(columns, row)) for row in cur.fetchall()]

    def _row_to_dict(self, row: Dict[str, Any]) -> Dict[str, Any]:
        keys = set(row.keys())
        base: Dict[str, Any] = {}
        if "id" in keys:
            base["id"] = int(row["id"]) if row["id"] is not None else None
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
            if field in keys:
                base[field] = row[field]
        return base

    def ready(self) -> bool:
        try:
            _ = self._connect_ro()
            return self._has_table_or_view(self._schema.history_relation) or self._has_table_or_view(
                self._schema.latest_relation
            )
        except Exception:
            return False

    def get_latest_status(self, person: str, task: str) -> Optional[Dict[str, Any]]:
        table = self._latest_table()
        sql = (
            f"SELECT * FROM {table} WHERE person = ? AND task = ? "
            "ORDER BY ts DESC, id DESC OFFSET 0 ROWS FETCH NEXT 1 ROWS ONLY"
        )
        rows = self._exec_select(sql, (person, task))
        if not rows:
            return None
        return self._row_to_dict(rows[0])

    def search(
        self,
        *,
        person: Optional[str] = None,
        task: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
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
        sql = (
            f"SELECT * FROM {table}{where} ORDER BY ts DESC, id DESC "
            "OFFSET 0 ROWS FETCH NEXT ? ROWS ONLY"
        )
        params.append(int(limit))
        rows = self._exec_select(sql, tuple(params))
        return [self._row_to_dict(r) for r in rows]

    def list_persons(self) -> List[str]:
        table = self._latest_table()
        rows = self._exec_select(f"SELECT DISTINCT person FROM {table}", ())
        return [str(r["person"]) for r in rows if r.get("person") is not None]

    def list_tasks(self) -> List[str]:
        table = self._latest_table()
        rows = self._exec_select(f"SELECT DISTINCT task FROM {table}", ())
        return [str(r["task"]) for r in rows if r.get("task") is not None]

    def query(self, sql: str, params: Tuple[Any, ...]) -> List[Dict[str, Any]]:
        s = (sql or "").strip().lower()
        if not s.startswith("select"):
            raise ValueError("only SELECT statements are allowed in MSSQLTasksStore.query")
        allowed_targets = tuple(f" from {name.lower()}" for name in self._schema.allowed_relations)
        if not any(t in s for t in allowed_targets):
            raise ValueError(f"query must target one of: {', '.join(self._schema.allowed_relations)}")
        rows = self._exec_select(sql, params)
        return [self._row_to_dict(r) for r in rows]
