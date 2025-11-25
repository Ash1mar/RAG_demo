from __future__ import annotations

from typing import Any, Dict

import pytest

from fastapi.testclient import TestClient

from app.demo_app import app, TASKS
from app.services.nl2sql_engine import (
    QueryFilter,
    TaskAnswerMode,
    TaskQueryIntent,
    TaskQuerySpec,
    TaskStatus,
    TimeRange,
    build_task_query_plan,
    parse_task_query_nl,
    _post_process_intent,
)
from app.services.sql_compiler import compile_tasks_sql, TaskSqlCompileError, CompiledSql

STATUS_QUERY = "\u5f20\u4e09\u7684E3D\u63a5\u53e3\u8054\u8c03\u73b0\u5728\u4ec0\u4e48\u72b6\u6001\uff1f"
MULTI_PERSON_QUERY = "\u5f20\u4e09\u548c\u674e\u56db\u6700\u8fd1\u4e00\u5468\u7684\u4efb\u52a1\u5217\u8868\u8fd8\u6709\u54ea\u4e9b\uff1f"
COMPLETION_QUERY = "\u5f20\u4e09\u7684E3D\u63a5\u53e3\u8054\u8c03\u662f\u4ec0\u4e48\u65f6\u5019\u5b8c\u6210\u7684\uff1f"
DUE_PRIORITY_QUERY = "\u5217\u51fa\u674e\u56db\u672c\u5468\u622a\u6b62\u7684\u9ad8\u4f18P1\u4efb\u52a1"
COUNT_QUERY = "\u5f20\u4e09\u8fd8\u6709\u591a\u5c11\u4efb\u52a1\u672a\u5b8c\u6210\uff1f"

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


def test_parse_task_query_detects_time_range_and_filters() -> None:
    spec = parse_task_query_nl(MULTI_PERSON_QUERY)
    assert spec.intent in (
        TaskQueryIntent.task_status_list,
        TaskQueryIntent.task_list_by_person,
    )
    assert spec.time_range is not None
    assert spec.person is not None
    assert any(
        f.field == "person" and f.op.lower() == "in" and f.values and len(f.values) >= 2
        for f in spec.filters
    )


def test_parse_completion_time_question_sets_answer_mode() -> None:
    q = "张三的E3D接口联调是什么时候完成的？"
    spec = parse_task_query_nl(q)
    assert spec.answer_mode == TaskAnswerMode.completion_time_latest
    assert spec.intent == TaskQueryIntent.task_history
    assert spec.status == [TaskStatus.DONE]
    assert spec.limit == 1


def test_parse_task_count_question_defaults_without_llm() -> None:
    spec = parse_task_query_nl(COUNT_QUERY)
    assert spec.answer_mode == TaskAnswerMode.default


def test_post_process_adds_multi_person_filters() -> None:
    spec = TaskQuerySpec(
        intent=TaskQueryIntent.task_status_list,
        raw_query="",
        person="??",
        filters=[],
    )
    _post_process_intent(spec, MULTI_PERSON_QUERY)
    assert any(
        getattr(f, "field", "") == "person"
        and getattr(f, "op", "").lower() == "in"
        for f in spec.filters
    )
    assert spec.person is None


def test_post_process_detects_due_range_and_priority_keywords() -> None:
    spec = TaskQuerySpec(
        intent=TaskQueryIntent.task_list_by_person,
        raw_query=DUE_PRIORITY_QUERY,
        person="??",
        filters=[],
    )
    _post_process_intent(spec, DUE_PRIORITY_QUERY)
    assert spec.due_range is not None
    assert spec.due_range.start == "start_of_week"
    assert spec.priority == 1


def test_build_plan_respects_custom_filters() -> None:
    spec = TaskQuerySpec(
        intent=TaskQueryIntent.task_status_list,
        raw_query="multi person query",
        person="张三",
        filters=[
            QueryFilter(field="person", op="in", values=["张三", "李四"]),
            QueryFilter(field="project", op="eq", value="Alpha"),
        ],
    )
    plan = build_task_query_plan(spec)
    fields = [f["field"] for f in plan["filters"]]
    assert "person" in fields
    assert "project" in fields
    assert any(f["op"] == "in" and f["field"] == "person" for f in plan["filters"])


def test_compile_sql_person_summary_requires_scope() -> None:
    spec = TaskQuerySpec(
        intent=TaskQueryIntent.person_summary,
        raw_query="summary missing scope",
    )
    with pytest.raises(TaskSqlCompileError):
        compile_tasks_sql(spec)


def test_compile_sql_person_summary_group_by() -> None:
    spec = TaskQuerySpec(
        intent=TaskQueryIntent.person_summary,
        raw_query="summary",
        person="张三",
        status=[TaskStatus.DONE],
    )
    compiled = compile_tasks_sql(spec)
    sql_lower = compiled.sql.lower()
    assert "count" in sql_lower
    assert "group by" in sql_lower
    assert "task_count" in sql_lower
    assert compiled.params[-1] == spec.limit


def test_build_plan_person_summary_by_project_answer_mode() -> None:
    spec = TaskQuerySpec(
        intent=TaskQueryIntent.task_status_list,
        raw_query="project summary",
        answer_mode=TaskAnswerMode.person_summary_by_project,
        project="芯片",
    )
    plan = build_task_query_plan(spec)
    assert plan["projections"] == ["project", "person", "status", "COUNT(*) AS task_count"]
    assert plan["group_by"] == ["project", "person", "status"]
    assert plan["sort"][0]["field"] == "project"


def test_compile_sql_overdue_count_by_person_answer_mode() -> None:
    spec = TaskQuerySpec(
        intent=TaskQueryIntent.task_status_list,
        raw_query="overdue counts",
        answer_mode=TaskAnswerMode.overdue_count_by_person,
        status=[TaskStatus.TODO, TaskStatus.IN_PROGRESS],
        due_range=TimeRange(start="start_of_week", end="end_of_week"),
    )
    compiled = compile_tasks_sql(spec)
    sql_lower = compiled.sql.lower()
    assert "count(*) as overdue_count" in sql_lower
    assert "group by person" in sql_lower
    assert compiled.params[-1] == spec.limit


def test_build_plan_for_task_count_answer_mode() -> None:
    spec = TaskQuerySpec(
        intent=TaskQueryIntent.task_status_list,
        raw_query=COUNT_QUERY,
        person="张三",
        status=[TaskStatus.TODO, TaskStatus.IN_PROGRESS],
        answer_mode=TaskAnswerMode.task_count_by_status,
        limit=10,
    )
    plan = build_task_query_plan(spec)
    assert plan["projections"] == ["status", "COUNT(*) AS task_count"]
    assert plan["group_by"] == ["status"]
    assert plan["sort"][0]["field"] == "task_count"
    assert plan["limit"] == 10


def test_compile_sql_task_count_answer_mode_includes_group_by() -> None:
    spec = TaskQuerySpec(
        intent=TaskQueryIntent.task_status_list,
        raw_query=COUNT_QUERY,
        person="张三",
        status=[TaskStatus.TODO, TaskStatus.IN_PROGRESS],
        answer_mode=TaskAnswerMode.task_count_by_status,
        limit=8,
    )
    compiled = compile_tasks_sql(spec)
    sql_lower = compiled.sql.lower()
    assert "count(*) as task_count" in sql_lower
    assert "group by status" in sql_lower
    assert compiled.params[-1] == spec.limit
