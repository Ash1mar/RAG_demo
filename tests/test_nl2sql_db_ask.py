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
from app.services.llm_client import _normalize_task_query_spec_payload
from app.services.sql_compiler import compile_tasks_sql, TaskSqlCompileError, CompiledSql
from app.services.task_query import (
    TaskQueryEngine,
    Text2SQLQueryModel,
    Text2SQLResponseModel,
    _extract_model_thinking,
    _make_text2sql_ir_hint,
    _rewrite_text2sql_query,
    _strip_model_thinking,
)
from app.services import task_query as task_query_module

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
    assert spec.answer_mode == TaskAnswerMode.task_count_by_status
    assert spec.status == [TaskStatus.TODO]


def test_remaining_count_question_sets_person_todo_count_mode() -> None:
    q = "\u5f20\u624b\u7434\u8fd8\u5269\u591a\u5c11\u4efb\u52a1"
    spec = parse_task_query_nl(q)
    assert spec.intent == TaskQueryIntent.task_status_list
    assert spec.answer_mode == TaskAnswerMode.task_count_by_status
    assert spec.person == "\u5f20\u624b\u7434"
    assert spec.task is None
    assert spec.status == [TaskStatus.TODO]


def test_llm_payload_moves_answer_mode_out_of_intent() -> None:
    payload = _normalize_task_query_spec_payload(
        {
            "intent": "task_count_by_status",
            "raw_query": "\u5f20\u624b\u7434\u8fd8\u5269\u591a\u5c11\u4efb\u52a1",
            "person": "\u5f20\u624b\u7434",
            "status": ["TODO"],
        },
        raw_query="\u5f20\u624b\u7434\u8fd8\u5269\u591a\u5c11\u4efb\u52a1",
    )
    assert payload["intent"] == "task_status_list"
    assert payload["answer_mode"] == "task_count_by_status"


def test_text2sql_ir_fallback_uses_count_answer_handler() -> None:
    class FakeStore:
        def query(self, sql, params):
            return [{"status": "TODO", "task_count": 9}]

    engine = TaskQueryEngine(tasks_store=FakeStore(), embedder=None, resolver_mode="text2sql")
    spec = TaskQuerySpec(
        intent=TaskQueryIntent.task_status_list,
        raw_query="\u5f20\u624b\u7434\u8fd8\u5269\u591a\u5c11\u4efb\u52a1",
        person="\u5f20\u624b\u7434",
        status=[TaskStatus.TODO],
        answer_mode=TaskAnswerMode.task_count_by_status,
        filters=[QueryFilter(field="person", op="eq", value="\u5f20\u624b\u7434")],
    )
    payload = engine._try_text2sql_ir_fallback(spec, {"debug_trace": []}, debug_enabled=True)
    assert payload is not None
    assert payload["answer"] == "\u5f20\u624b\u7434\u7684\u4efb\u52a1\u6309\u72b6\u6001\u7edf\u8ba1\uff1a\u672a\u5b8c\u6210=9\uff08\u603b\u8ba1 9\uff09\u3002"
    assert payload["status_counts"] == [{"status": "TODO", "count": 9}]
    assert payload["total_tasks"] == 9


def test_text2sql_db_failure_falls_back_to_clean_ir_count(monkeypatch) -> None:
    class FlakyStore:
        def __init__(self):
            self.calls = 0

        def query(self, sql, params):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("ORDER BY invalid for aggregate query")
            return [{"status": "TODO", "task_count": 9}]

    def fake_text2sql_llm(prompt, *, dialect="sqlite"):
        return (
            Text2SQLResponseModel(
                queries=[
                    Text2SQLQueryModel(
                        sql=(
                            "SELECT COUNT(*) AS unfinished_task_count "
                            "FROM task_latest WHERE person = '张手琴' "
                            "AND status = 'TODO' ORDER BY ts DESC"
                        ),
                        description="bad aggregate SQL",
                    )
                ]
            ),
            {"provider": "openai", "model": "fake"},
            {"thinking": "draft sql"},
        )

    monkeypatch.setattr(task_query_module.llm_settings, "enabled", True)
    monkeypatch.setattr(task_query_module.llm_settings, "provider", "openai")
    monkeypatch.setattr(task_query_module, "_call_text2sql_llm", fake_text2sql_llm)

    store = FlakyStore()
    engine = TaskQueryEngine(tasks_store=store, embedder=None, resolver_mode="text2sql")
    spec = TaskQuerySpec(
        intent=TaskQueryIntent.task_status_list,
        raw_query="张手琴还有多少未完成的任务？",
        person="张手琴",
        status=[TaskStatus.TODO],
        answer_mode=TaskAnswerMode.task_count_by_status,
        filters=[QueryFilter(field="person", op="eq", value="张手琴")],
    )

    payload = engine._answer_via_text2sql(
        "张手琴还有多少未完成的任务？",
        spec,
        {"debug_trace": []},
    )

    assert store.calls == 2
    assert payload["resolver_mode"] == "text2sql_ir_fallback"
    assert payload["answer"] == "\u5f20\u624b\u7434\u7684\u4efb\u52a1\u6309\u72b6\u6001\u7edf\u8ba1\uff1a\u672a\u5b8c\u6210=9\uff08\u603b\u8ba1 9\uff09\u3002"
    assert payload["text2sql_fallback"] == "cleaned_ir"
    assert payload["debug_thinking"] == "draft sql"
    assert "text2sql_db_error" in payload
    assert "ORDER BY ts DESC" in payload["text2sql_failed_sql"]


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
    assert spec.time_range is None
    assert spec.priority == 1


def test_list_query_clears_question_fragment_task() -> None:
    q = "\u5f20\u624b\u7434\u7684\u4efb\u52a1\u6709\u54ea\u4e9b"
    spec = parse_task_query_nl(q)
    assert spec.person == "\u5f20\u624b\u7434"
    assert spec.task is None
    assert not any(getattr(f, "field", "") == "task" for f in spec.filters)


def test_multi_person_list_query_uses_person_filter_not_task_filter() -> None:
    q = "\u5f20\u624b\u7434\u548c\u5218\u536b\u6c11\u90fd\u5b8c\u6210\u4e86\u54ea\u4e9b\u4efb\u52a1"
    spec = parse_task_query_nl(q)
    assert spec.task is None
    assert any(
        getattr(f, "field", "") == "person"
        and getattr(f, "op", "").lower() == "in"
        and getattr(f, "values", None) == ["\u5f20\u624b\u7434", "\u5218\u536b\u6c11"]
        for f in spec.filters
    )
    assert not any(getattr(f, "field", "") == "task" for f in spec.filters)


def test_low_trust_rules_ir_hint_drops_entity_fields() -> None:
    q = "\u5f20\u624b\u7434\u548c\u5218\u536b\u6c11\u90fd\u5b8c\u6210\u4e86\u54ea\u4e9b\u4efb\u52a1"
    spec = TaskQuerySpec(
        intent=TaskQueryIntent.task_status_list,
        raw_query=q,
        task=q,
        filters=[QueryFilter(field="task", op="in", values=["\u5f20\u624b\u7434", "\u5218\u536b\u6c11\u90fd\u5b8c\u6210\u4e86\u54ea\u4e9b\u4efb\u52a1"])],
        extra={"nl2sql_source": "rules", "nl2sql_llm_error": "timeout"},
    )
    hint = _make_text2sql_ir_hint(spec)
    assert hint["hint_quality"]["trusted"] is False
    assert "task" not in hint
    assert "filters" not in hint


def test_post_process_normalizes_single_value_filter_payload() -> None:
    q = "\u5217\u51fa\u5f20\u624b\u7434\u672c\u5468\u622a\u6b62\u7684\u4efb\u52a1"
    spec = TaskQuerySpec(
        intent=TaskQueryIntent.task_list_by_person,
        raw_query=q,
        filters=[QueryFilter(field="person", op="eq", values=[{"name": "\u5f20\u624b\u7434"}])],
    )
    _post_process_intent(spec, q)
    plan = build_task_query_plan(spec)
    assert any(
        f["field"] == "person" and f["op"] == "eq" and f["value"] == "\u5f20\u624b\u7434"
        for f in plan["filters"]
    )
    assert not any(f["field"] == "person" and f.get("value") is None for f in plan["filters"])


def test_strip_model_thinking_removes_reasoning_prefix() -> None:
    assert _strip_model_thinking("<think>draft</think>\u7b54\u6848") == "\u7b54\u6848"
    assert _strip_model_thinking("draft</think>\u7b54\u6848") == "\u7b54\u6848"


def test_extract_model_thinking_for_debug() -> None:
    assert _extract_model_thinking("<think>draft</think>\u7b54\u6848") == "draft"
    assert _extract_model_thinking("draft</think>\u7b54\u6848") == "draft"
    assert _extract_model_thinking("\u7b54\u6848") is None


def test_post_process_list_style_query_does_not_force_single_task() -> None:
    q = "E3D系统中安全专项相关的高优P1任务有哪些？"
    spec = TaskQuerySpec(
        intent=TaskQueryIntent.task_status_single,
        raw_query=q,
        person="张三",
        task="E3D接口联调",
        project="E3D",
        priority=1,
        tags=["整改", "安全整改"],
        filters=[
            QueryFilter(field="tags", op="in", values=[{"tag": "安全专项"}]),
            QueryFilter(field="priority", op="eq", value={"P1": "high"}),
        ],
        limit=50,
    )
    _post_process_intent(spec, q)
    assert spec.intent in (
        TaskQueryIntent.task_status_list,
        TaskQueryIntent.task_list_by_person,
    )
    assert spec.task is None
    assert spec.person is None


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


def test_compile_sql_sanitizes_structured_filter_values() -> None:
    spec = TaskQuerySpec(
        intent=TaskQueryIntent.task_status_list,
        raw_query="structured filters from llm",
        project="E3D",
        priority=1,
        tags=["整改", "安全整改"],
        filters=[
            QueryFilter(field="tags", op="in", values=[{"tag": "安全专项"}]),
            QueryFilter(field="priority", op="eq", value={"P1": "high"}),
        ],
        limit=50,
    )
    compiled = compile_tasks_sql(spec)
    assert isinstance(compiled, CompiledSql)
    assert all(not isinstance(p, dict) for p in compiled.params)


def test_text2sql_rewrite_flow_uses_project_not_tags() -> None:
    q = "流程（flow_name）为 部门任务流程 下有哪些任务？"
    sql = "SELECT task, status, ts FROM task_latest WHERE tags LIKE '%部门任务流程%' ORDER BY ts DESC LIMIT 10"
    hint = {"project": "部门任务流程"}
    rewritten = _rewrite_text2sql_query(sql, hint, question=q)
    lowered = rewritten.lower()
    assert "project = '部门任务流程'".lower() in lowered
    assert "tags like '%部门任务流程%'" not in lowered


def test_text2sql_rewrite_flow_can_infer_from_question() -> None:
    q = "流程（flow_name）为 部门任务流程 下有哪些任务？"
    sql = "SELECT task, status, ts FROM task_latest WHERE tags LIKE '%部门任务流程%' ORDER BY ts DESC LIMIT 10"
    rewritten = _rewrite_text2sql_query(sql, {}, question=q)
    lowered = rewritten.lower()
    assert "project = '部门任务流程'".lower() in lowered
    assert "tags like '%部门任务流程%'" not in lowered


def test_text2sql_rewrite_drops_spurious_blocked_status() -> None:
    q = "流程（flow_name）为 部门任务流程 中即将到期的任务有哪些？"
    sql = "SELECT * FROM task_latest WHERE project = '部门任务流程' AND status = 'BLOCKED' AND due_ts < 9999999999999 LIMIT 10"
    hint = {"project": "部门任务流程", "status": []}
    rewritten = _rewrite_text2sql_query(sql, hint, question=q)
    assert "BLOCKED" not in rewritten


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
