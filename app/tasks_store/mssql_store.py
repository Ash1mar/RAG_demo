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
            field_map=base.field_map,
        )

    def _col(self, logical: str) -> str:
        return self._schema.translate_field(logical)

    def _pick_row_key(self, row: Dict[str, Any], logical: str) -> Optional[str]:
        if logical in row:
            return logical
        physical = self._col(logical)
        if physical in row:
            return physical
        key_map = {str(k).lower(): k for k in row.keys()}
        logical_key = str(logical).lower()
        if logical_key in key_map:
            return key_map[logical_key]
        physical_key = str(physical).lower()
        return key_map.get(physical_key)

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
        base: Dict[str, Any] = dict(row)
        id_key = self._pick_row_key(row, "id")
        if id_key is not None:
            base["id"] = int(row[id_key]) if row[id_key] is not None else None
        person_key = self._pick_row_key(row, "person")
        if person_key is not None:
            base["person"] = row[person_key]
        task_key = self._pick_row_key(row, "task")
        if task_key is not None:
            base["task"] = row[task_key]
        status_key = self._pick_row_key(row, "status")
        if status_key is not None:
            base["status"] = row[status_key]
        ts_key = self._pick_row_key(row, "ts")
        if ts_key is not None:
            try:
                base["ts"] = int(row[ts_key]) if row[ts_key] is not None else None
            except (TypeError, ValueError):
                base["ts"] = row[ts_key]
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
            "owner",
            "org_name",
            "division_name",
            "post_name",
            "is_read",
            "is_delegated",
        ):
            key = self._pick_row_key(row, field)
            if key is not None:
                base[field] = row[key]
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
        person_col = self._col("person")
        task_col = self._col("task")
        ts_col = self._col("ts")
        id_col = self._col("id")
        sql = (
            f"SELECT * FROM {table} WHERE {person_col} = ? AND {task_col} = ? "
            f"ORDER BY {ts_col} DESC, {id_col} DESC OFFSET 0 ROWS FETCH NEXT 1 ROWS ONLY"
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
        person_col = self._col("person")
        task_col = self._col("task")
        ts_col = self._col("ts")
        id_col = self._col("id")
        clauses: List[str] = []
        params: List[Any] = []
        if person:
            clauses.append(f"{person_col} = ?")
            params.append(person)
        if task:
            clauses.append(f"{task_col} = ?")
            params.append(task)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = (
            f"SELECT * FROM {table}{where} ORDER BY {ts_col} DESC, {id_col} DESC "
            "OFFSET 0 ROWS FETCH NEXT ? ROWS ONLY"
        )
        params.append(int(limit))
        rows = self._exec_select(sql, tuple(params))
        return [self._row_to_dict(r) for r in rows]

    def list_persons(self) -> List[str]:
        table = self._latest_table()
        col = self._col("person")
        rows = self._exec_select(f"SELECT DISTINCT {col} AS person FROM {table}", ())
        return [str(r.get("person")) for r in rows if r.get("person") is not None]

    def list_tasks(self) -> List[str]:
        table = self._latest_table()
        col = self._col("task")
        rows = self._exec_select(f"SELECT DISTINCT {col} AS task FROM {table}", ())
        return [str(r.get("task")) for r in rows if r.get("task") is not None]

    def query(self, sql: str, params: Tuple[Any, ...]) -> List[Dict[str, Any]]:
        s = (sql or "").strip().lower()
        if not s.startswith("select"):
            raise ValueError("only SELECT statements are allowed in MSSQLTasksStore.query")
        allowed_targets = tuple(f" from {name.lower()}" for name in self._schema.allowed_relations)
        if not any(t in s for t in allowed_targets):
            raise ValueError(f"query must target one of: {', '.join(self._schema.allowed_relations)}")
        rows = self._exec_select(sql, params)
        return [self._row_to_dict(r) for r in rows]
