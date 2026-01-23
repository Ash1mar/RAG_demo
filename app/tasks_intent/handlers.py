from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.services.nl2sql_engine import TaskAnswerMode, TaskQueryIntent, TaskQuerySpec, TaskStatus
from app.tasks_intent.base import AnswerContext, TaskIntentHandler


class CompletionTimeLatestHandler:
    name = "completion_time_latest"

    def matches(self, spec: TaskQuerySpec, answer_mode: TaskAnswerMode) -> bool:
        return answer_mode == TaskAnswerMode.completion_time_latest

    def label(self, spec: TaskQuerySpec) -> Optional[str]:
        return None

    def build_answer(self, ctx: AnswerContext) -> Dict[str, Any]:
        done_row = next(
            (
                rec
                for rec in ctx.rows
                if str(rec.get("status", "")).upper() == TaskStatus.DONE.value
            ),
            ctx.rows[0],
        )
        ts = int(done_row.get("ts", -1))
        ts_str = ctx.format_ts(ts) if ts >= 0 else "unknown time"
        payload = {
            "answer": f'{ctx.person} / "{ctx.task or ctx.spec.task}" was completed at {ts_str}.',
            "person": ctx.person,
            "task": ctx.task or ctx.spec.task,
            "status": str(done_row.get("status", "")).upper(),
            "ts": ts,
        }
        if ctx.low_conf:
            payload["answer"] = str(payload.get("answer", "")) + " (low confidence)"
        return payload


class TaskCountByStatusHandler:
    name = "task_count_by_status"

    def matches(self, spec: TaskQuerySpec, answer_mode: TaskAnswerMode) -> bool:
        return answer_mode == TaskAnswerMode.task_count_by_status

    def label(self, spec: TaskQuerySpec) -> Optional[str]:
        return None

    def build_answer(self, ctx: AnswerContext) -> Dict[str, Any]:
        counts_map: Dict[str, int] = {}
        for rec in ctx.rows:
            status = str(rec.get("status", "")).upper() or "UNKNOWN"
            raw_count = rec.get("task_count")
            try:
                cnt = int(raw_count)
            except (TypeError, ValueError):
                cnt = 1
            if cnt < 0:
                cnt = 0
            counts_map[status] = counts_map.get(status, 0) + cnt
        counts = [
            {"status": status, "count": counts_map[status]}
            for status in sorted(counts_map.keys(), key=lambda s: (-counts_map[s], s))
        ]
        total = sum(item["count"] for item in counts)
        stats_str = ", ".join(f"{item['status']}={item['count']}" for item in counts) or "none"
        if ctx.person_filters_active and ctx.person_filter_values:
            subject_label = ", ".join(ctx.person_filter_values)
        elif ctx.person:
            subject_label = str(ctx.person)
        else:
            subject_label = "Tasks"
        scope_bits: List[str] = []
        time_range = getattr(ctx.spec, "time_range", None)
        if time_range:
            scope_bits.append(
                f"time_range={getattr(time_range, 'start', None) or '*'}~{getattr(time_range, 'end', None) or '*'}"
            )
        due_range = getattr(ctx.spec, "due_range", None)
        if due_range:
            scope_bits.append(
                f"due_range={getattr(due_range, 'start', None) or '*'}~{getattr(due_range, 'end', None) or '*'}"
            )
        scope_suffix = f" within {', '.join(scope_bits)}" if scope_bits else ""
        if subject_label == "Tasks":
            answer_prefix = "Tasks by status"
        else:
            answer_prefix = f"{subject_label} tasks by status"
        payload = {
            "answer": f"{answer_prefix}{scope_suffix}: {stats_str} (total {total}).",
            "person": None if ctx.person_filters_active else ctx.person,
            "persons": ctx.person_filter_values if ctx.person_filters_active else ([ctx.person] if ctx.person else []),
            "task": None,
            "status_counts": counts,
            "total_tasks": total,
        }
        if ctx.low_conf:
            payload["answer"] = str(payload.get("answer", "")) + " (low confidence)"
        return payload


class PersonSummaryByProjectHandler:
    name = "person_summary_by_project"

    def matches(self, spec: TaskQuerySpec, answer_mode: TaskAnswerMode) -> bool:
        return answer_mode == TaskAnswerMode.person_summary_by_project

    def label(self, spec: TaskQuerySpec) -> Optional[str]:
        return None

    def build_answer(self, ctx: AnswerContext) -> Dict[str, Any]:
        summary: Dict[str, Dict[str, Dict[str, int]]] = {}
        for rec in ctx.rows:
            project = str(rec.get("project", "") or "Unspecified")
            person_name = str(rec.get("person", "") or "Unknown")
            status_val = str(rec.get("status", "") or "UNKNOWN").upper()
            count_val = rec.get("task_count")
            try:
                cnt = int(count_val)
            except (TypeError, ValueError):
                cnt = 0
            summary.setdefault(project, {}).setdefault(person_name, {})[status_val] = cnt

        parts: List[str] = []
        for project, people in summary.items():
            person_bits: List[str] = []
            for person_name, status_map in people.items():
                status_bits = [f"{status}={count}" for status, count in status_map.items()]
                person_bits.append(f"{person_name}({', '.join(status_bits)})")
            project_summary = "; ".join(person_bits) if person_bits else "no data"
            parts.append(f"{project}: {project_summary}")
        answer = " | ".join(parts) if parts else "No summary data."
        payload = {
            "answer": f"Project/person status summary: {answer}",
            "project_summary": summary,
            "person": None,
            "persons": [],
            "task": None,
        }
        if ctx.low_conf:
            payload["answer"] = str(payload.get("answer", "")) + " (low confidence)"
        return payload


class OverdueCountByPersonHandler:
    name = "overdue_count_by_person"

    def matches(self, spec: TaskQuerySpec, answer_mode: TaskAnswerMode) -> bool:
        return answer_mode == TaskAnswerMode.overdue_count_by_person

    def label(self, spec: TaskQuerySpec) -> Optional[str]:
        return None

    def build_answer(self, ctx: AnswerContext) -> Dict[str, Any]:
        rows_summary: List[Dict[str, Any]] = []
        for rec in ctx.rows:
            person_name = str(rec.get("person", "") or "Unknown")
            raw_count = rec.get("overdue_count")
            try:
                cnt = int(raw_count)
            except (TypeError, ValueError):
                cnt = 0
            rows_summary.append({"person": person_name, "count": cnt})
        rows_summary.sort(key=lambda item: (-item["count"], item["person"]))
        scope_bits: List[str] = []
        time_range = getattr(ctx.spec, "time_range", None)
        if time_range:
            scope_bits.append(
                f"time_range={getattr(time_range, 'start', None) or '*'}~{getattr(time_range, 'end', None) or '*'}"
            )
        due_range = getattr(ctx.spec, "due_range", None)
        if due_range:
            scope_bits.append(
                f"due_range={getattr(due_range, 'start', None) or '*'}~{getattr(due_range, 'end', None) or '*'}"
            )
        scope_suffix = f" within {', '.join(scope_bits)}" if scope_bits else ""
        summary_str = ", ".join(f"{item['person']}={item['count']}" for item in rows_summary) or "none"
        payload = {
            "answer": f"Overdue tasks per person{scope_suffix}: {summary_str}.",
            "overdue_counts": rows_summary,
            "person": None,
            "persons": [item["person"] for item in rows_summary],
            "task": None,
        }
        if ctx.low_conf:
            payload["answer"] = str(payload.get("answer", "")) + " (low confidence)"
        return payload


class TaskListByPersonHandler:
    name = "task_list_by_person"

    def matches(self, spec: TaskQuerySpec, answer_mode: TaskAnswerMode) -> bool:
        return spec.intent == TaskQueryIntent.task_list_by_person

    def label(self, spec: TaskQuerySpec) -> Optional[str]:
        return "task_list"

    def build_answer(self, ctx: AnswerContext) -> Dict[str, Any]:
        count = len(ctx.rows)
        preview_tasks: List[str] = []
        for rec in ctx.rows[:5]:
            t_name = str(rec.get("task", ""))
            t_status = str(rec.get("status", "")).upper()
            rec_person = str(rec.get("person", ""))
            if ctx.person_filters_active and rec_person:
                preview_tasks.append(f"{rec_person}:{t_name}({t_status})")
            else:
                preview_tasks.append(f"{t_name}({t_status})")
        preview = ", ".join(preview_tasks) if preview_tasks else "none"
        if ctx.person_filters_active:
            names = ", ".join(ctx.person_filter_values)
            payload = {
                "answer": f"Tasks for {names}: {preview}",
                "person": None,
                "persons": ctx.person_filter_values,
                "task": None,
            }
        else:
            payload = {
                "answer": f"{ctx.person} has {count} tasks: {preview}",
                "person": ctx.person,
                "task": None,
            }
        if ctx.low_conf:
            payload["answer"] = str(payload.get("answer", "")) + " (low confidence)"
        return payload


class TaskStatusListHandler:
    name = "task_status_list"

    def matches(self, spec: TaskQuerySpec, answer_mode: TaskAnswerMode) -> bool:
        return spec.intent == TaskQueryIntent.task_status_list

    def label(self, spec: TaskQuerySpec) -> Optional[str]:
        return "status_query"

    def build_answer(self, ctx: AnswerContext) -> Dict[str, Any]:
        count = len(ctx.rows)
        preview: List[str] = []
        for rec in ctx.rows[:5]:
            t_name = str(rec.get("task", ""))
            t_status = str(rec.get("status", "")).upper()
            rec_person = str(rec.get("person", ""))
            if ctx.person_filters_active and rec_person:
                preview.append(f"{rec_person}:{t_name}({t_status})")
            else:
                preview.append(f"{t_name}({t_status})")
        preview_str = ", ".join(preview) if preview else "none"
        if ctx.person_filters_active:
            names = ", ".join(ctx.person_filter_values)
            payload = {
                "answer": f"{names} have {count} task status records: {preview_str}",
                "person": None,
                "persons": ctx.person_filter_values,
                "task": None,
            }
        else:
            payload = {
                "answer": f"{ctx.person} has {count} task status records: {preview_str}",
                "person": ctx.person,
                "task": None,
            }
        if ctx.low_conf:
            payload["answer"] = str(payload.get("answer", "")) + " (low confidence)"
        return payload


class TaskHistoryHandler:
    name = "task_history"

    def matches(self, spec: TaskQuerySpec, answer_mode: TaskAnswerMode) -> bool:
        return spec.intent == TaskQueryIntent.task_history

    def label(self, spec: TaskQuerySpec) -> Optional[str]:
        return "task_history"

    def build_answer(self, ctx: AnswerContext) -> Dict[str, Any]:
        count = len(ctx.rows)
        rec = ctx.rows[0]
        status = str(rec.get("status", "")).upper()
        ts = int(rec.get("ts", -1))
        ts_str = ctx.format_ts(ts) if ts >= 0 else "unknown time"
        payload = {
            "answer": f'{ctx.person} / "{ctx.task}" has {count} status records; latest is {status} at {ts_str}.',
            "person": ctx.person,
            "task": ctx.task,
            "status": status,
            "ts": ts,
        }
        if ctx.low_conf:
            payload["answer"] = str(payload.get("answer", "")) + " (low confidence)"
        return payload


class PersonSummaryHandler:
    name = "person_summary"

    def matches(self, spec: TaskQuerySpec, answer_mode: TaskAnswerMode) -> bool:
        return spec.intent == TaskQueryIntent.person_summary

    def label(self, spec: TaskQuerySpec) -> Optional[str]:
        return "person_summary"

    def build_answer(self, ctx: AnswerContext) -> Dict[str, Any]:
        summary: Dict[str, List[str]] = {}
        for rec in ctx.rows:
            p_name = str(rec.get("person", ""))
            status = str(rec.get("status", "")).upper()
            count_val = rec.get("task_count")
            try:
                cnt = int(count_val)
            except (TypeError, ValueError):
                cnt = count_val
            summary.setdefault(p_name, []).append(f"{status}={cnt}")
        parts = []
        for p_name, stats in summary.items():
            stats_str = ", ".join(stats)
            parts.append(f"{p_name}: {stats_str}")
        answer = "; ".join(parts) if parts else "No summary data."
        payload = {
            "answer": answer,
            "person": None if ctx.person_filters_active else ctx.person,
            "persons": ctx.person_filter_values if ctx.person_filters_active else ([ctx.person] if ctx.person else []),
            "task": None,
        }
        if ctx.low_conf:
            payload["answer"] = str(payload.get("answer", "")) + " (low confidence)"
        return payload


class DefaultStatusHandler:
    name = "default_status"

    def matches(self, spec: TaskQuerySpec, answer_mode: TaskAnswerMode) -> bool:
        return True

    def label(self, spec: TaskQuerySpec) -> Optional[str]:
        if spec.intent == TaskQueryIntent.task_list_by_person:
            return "task_list"
        if spec.intent == TaskQueryIntent.task_history:
            return "task_history"
        if spec.intent in (
            TaskQueryIntent.task_status_single,
            TaskQueryIntent.task_status_list,
        ):
            return "status_query"
        if spec.intent == TaskQueryIntent.person_summary:
            return "person_summary"
        return "unknown"

    def build_answer(self, ctx: AnswerContext) -> Dict[str, Any]:
        rec = ctx.rows[0]
        status = str(rec.get("status", "")).upper()
        ts = int(rec.get("ts", -1))
        ts_str = ctx.format_ts(ts) if ts >= 0 else "unknown time"
        payload = {
            "answer": f'{ctx.person} / "{ctx.task}" is {"completed" if status == "DONE" else status.lower()} (latest update: {ts_str}).',
            "person": ctx.person,
            "task": ctx.task,
            "status": status,
            "ts": ts,
        }
        if ctx.low_conf:
            payload["answer"] = str(payload.get("answer", "")) + " (low confidence)"
        return payload


HANDLERS: List[TaskIntentHandler] = [
    CompletionTimeLatestHandler(),
    TaskCountByStatusHandler(),
    PersonSummaryByProjectHandler(),
    OverdueCountByPersonHandler(),
    TaskListByPersonHandler(),
    TaskStatusListHandler(),
    TaskHistoryHandler(),
    PersonSummaryHandler(),
    DefaultStatusHandler(),
]
