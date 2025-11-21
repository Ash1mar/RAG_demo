from __future__ import annotations

from typing import Any, Dict

from fastapi.testclient import TestClient

from app.demo_app import app, TASKS
from app.services.nl2sql_engine import TaskQueryIntent, parse_task_query_nl, TaskQuerySpec
from app.services.sql_compiler import compile_tasks_sql, TaskSqlCompileError, CompiledSql


client = TestClient(app)


def _ensure_demo_tasks() -> None:
    """Ensure demo tasks DB is initialized.

    Tests assume scripts/init_tasks_sqlite.py has been run once, but for
    robustness we simply check that the tasks table is ready.
    """
    assert TASKS.ready(), "tasks store is not ready; run scripts/init_tasks_sqlite.py first"


def test_parse_task_query_single_status() -> None:
    q = "张三的E3D接口联调现在什么状态？"
    spec = parse_task_query_nl(q)
    assert isinstance(spec, TaskQuerySpec)
    assert spec.intent == TaskQueryIntent.task_status_single
    assert spec.person is not None and "张三" in spec.person
    assert spec.task is not None
    # default limit and order_by should be populated
    assert spec.limit is not None and spec.limit > 0
    assert spec.order_by, "order_by should not be empty"


def test_compile_sql_single_status() -> None:
    spec = TaskQuerySpec(
        intent=TaskQueryIntent.task_status_single,
        raw_query="张三的E3D接口联调现在什么状态？",
        person="张三",
        task="E3D接口联调",
    )
    compiled = compile_tasks_sql(spec)
    assert isinstance(compiled, CompiledSql)
    sql_lower = compiled.sql.lower()
    assert sql_lower.startswith("select")
    assert (" from task_latest" in sql_lower) or (" from tasks" in sql_lower)
    assert "where person = ?" in sql_lower
    assert "and task = ?" in sql_lower
    assert "order by" in sql_lower
    assert "limit 1" in sql_lower or "limit ?" in sql_lower
    assert compiled.params[:2] == ("张三", "E3D接口联调")


def test_compile_sql_requires_person_and_task() -> None:
    spec = TaskQuerySpec(
        intent=TaskQueryIntent.task_status_single,
        raw_query="张三的E3D接口联调现在什么状态？",
        person="",
        task="E3D接口联调",
    )
    try:
        compile_tasks_sql(spec)
        assert False, "expected TaskSqlCompileError for missing person"
    except TaskSqlCompileError:
        pass


def test_db_ask_single_status_roundtrip() -> None:
    _ensure_demo_tasks()
    q = "张三的E3D接口联调现在什么状态？"
    resp = client.get("/db/ask", params={"q": q})
    assert resp.status_code == 200
    data: Dict[str, Any] = resp.json()
    assert data["query"] == q
    assert "ir" in data and isinstance(data["ir"], dict)
    assert "sql" in data and isinstance(data["sql"], str)
    assert "params" in data
    assert "rows" in data and isinstance(data["rows"], list)
    # For the demo DB we expect at most a handful of rows; at least 0.
    assert len(data["rows"]) >= 0


def test_db_ask_invalid_query_returns_4xx() -> None:
    # For an obviously incomplete query, IR may not compile into SQL.
    resp = client.get("/db/ask", params={"q": ""})
    assert resp.status_code == 400
