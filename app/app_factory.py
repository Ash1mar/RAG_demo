from __future__ import annotations

from os import getenv
from typing import Tuple

from app.services.embeddings import Embedder
from app.services.task_query import TaskQueryEngine
from app.tasks_store.base import TasksStore
from app.tasks_store.sqlite_store import SQLiteTasksConfig, SQLiteTasksStore


def create_tasks_store() -> Tuple[str, TasksStore]:
    backend = getenv("TASKS_BACKEND", "sqlite").lower()
    if backend == "sqlite":
        return backend, SQLiteTasksStore(SQLiteTasksConfig(db_path=getenv("TASKS_DB", "data/tasks.db")))
    if backend == "mssql":
        try:
            from app.tasks_store.mssql_store import MSSQLTasksConfig, MSSQLTasksStore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "MSSQL backend requested but pyodbc is unavailable. Install pyodbc and the SQL Server ODBC driver."
            ) from exc
        store = MSSQLTasksStore(
            MSSQLTasksConfig(
                server=getenv("TASKS_MSSQL_SERVER", "127.0.0.1,1433"),
                database=getenv("TASKS_MSSQL_DATABASE", "fact_tasks"),
                user=getenv("TASKS_MSSQL_USER", "sa"),
                password=getenv("TASKS_MSSQL_PASSWORD", ""),
                driver=getenv("TASKS_MSSQL_DRIVER", "ODBC Driver 18 for SQL Server"),
                encrypt=(getenv("TASKS_MSSQL_ENCRYPT", "yes").lower() != "no"),
                trust_server_certificate=(
                    getenv("TASKS_MSSQL_TRUST_CERT", "yes").lower() != "no"
                ),
                timeout_sec=float(getenv("TASKS_MSSQL_TIMEOUT_SEC", "5")),
            )
        )
        return backend, store
    # Placeholder for future backends (e.g., KG). Keep API stable.
    return "sqlite", SQLiteTasksStore(SQLiteTasksConfig(db_path=getenv("TASKS_DB", "data/tasks.db")))


def create_task_query_engine(
    tasks_store: TasksStore, embedder: Embedder
) -> Tuple[str, TaskQueryEngine]:
    resolver_mode = getenv("RESOLVER", "hybrid").lower()
    engine = TaskQueryEngine(tasks_store=tasks_store, embedder=embedder, resolver_mode=resolver_mode)
    return resolver_mode, engine
