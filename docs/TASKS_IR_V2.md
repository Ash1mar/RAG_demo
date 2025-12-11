# Task Query IR v2 (`TaskQuerySpec`)

This document describes the semantic intermediate representation (IR) used by
the tasks Q&A subsystem: `TaskQuerySpec`. It is the *single source of truth*
for how natural-language questions are mapped into structured intent and
filters, independently of the underlying physical database schema
(single-table vs. multi-table).

The key design principle is:

> Natural language → `TaskQuerySpec` (IR) → query plan → SQL.

When we later introduce multi-table / multi-view schemas (joins to `persons`,
`projects`, `tags`, etc.), we will **only** change the “query plan → SQL”
layer. The IR (`TaskQuerySpec`) and KG-lite resolution remain stable.

---

## 1. Goals

- Provide a easy-to-read, schema-agnostic semantic shape for task queries.
- Separate:
  - **Conceptual fields** (person, project, status, tags, time ranges, etc.)
    from
  - **Physical layout** (tables, views, joins, indices).
- Make it easy to add multi-table / multi-view support without touching:
  - NL → IR parsing (`parse_task_query_nl`);
  - KG-lite resolution (`kg_lite`).

---

## 2. High-level concepts

At a high level, `TaskQuerySpec` captures:

- **Intent**: what kind of question is being asked
  (single task status, list, history, per-person summary, etc.).
- **Scope**: who / what / when the question is about
  (person, task, project, tags, status, time window).
- **Answer mode**: how the answer should be shaped
  (raw rows, counts by status, per-person summaries, etc.).
- **Additional filters**: a flexible list of typed filters for more advanced
  queries.
- **Debug / metadata**: extra fields used for tracing and diagnostics,
  which do not participate in the semantics directly.

`TaskQuerySpec` is consumed by the IR→plan layer (`build_task_query_plan` /
future `build_query_plan_v2`) and then by the SQL builder (`app/sql_builder.py`).

---

## 3. Field-level reference

The table below lists each major conceptual field in `TaskQuerySpec`, its
business meaning, the current physical mapping, and notes about the future
multi-table evolution.

> NOTE: “Current physical mapping” refers to the demo schema initialized by
> `scripts/init_tasks_sqlite.py`:
> - `tasks` table (denormalized status history);
> - `task_latest` view (latest status per `(person, task)`);
> - `persons` table (normalized person info).

### 3.1 Core intent / answer shape

| Field            | Type                      | Business meaning                                                                 | Current physical mapping                                                | Future multi-table notes                                                                                 |
|------------------|---------------------------|----------------------------------------------------------------------------------|-------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------|
| `intent`         | `TaskQueryIntent`        | Semantic intent type (single status, list, history, per-person summary, etc.).  | Drives which table/view is targeted (`tasks` vs `task_latest`) and plan projections/grouping. | Remains a pure semantic selector; only affects which query template / plan shape is chosen.             |
| `answer_mode`    | `TaskAnswerMode`         | Desired answer shape (raw rows, counts by status, per-person summary, etc.).    | Translated into plan-level `projections`, `group_by`, and default `sort`. | Remains independent of physical schema; multi-table logic will adapt the plan while keeping this stable. |
| `raw_query`      | `str`                    | Original natural-language question.                                              | Not mapped to DB; used for logging and text-based heuristics.          | Unchanged. May be used for additional routing or KG hints, but not for schema details.                  |
| `is_supported`   | `Optional[bool]`         | LLM hint whether the IR fast path should handle this query.                     | Not mapped to DB. Used for routing/guardrails.                         | Unchanged.                                                                                               |
| `intent_confidence` | `Optional[float]`     | LLM confidence (0–1) for `intent`.                                              | Not mapped to DB.                                                      | Unchanged.                                                                                               |
| `raw_intent_nl`  | `Optional[str]`          | NL summary of detected intent (LLM-provided).                                   | Not mapped to DB.                                                      | Unchanged.                                                                                               |

### 3.2 Entity / scope fields

| Field          | Type                    | Business meaning                                                                | Current physical mapping                                                                                       | Future multi-table notes                                                                                                               |
|----------------|-------------------------|---------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------|
| `person`       | `Optional[str]`        | Semantic “person” name (normalized by KG-lite).                                 | Filters on `task_latest.person` or `tasks.person` (denormalized string). A foreign key `tasks.person_id` points to `persons.id`. | IR remains “person” as a logical entity. Multi-table plans may join `tasks.person_id` to `persons.id` to access `team`, `role`, etc.  |
| `task`         | `Optional[str]`        | Task title / name (best guess).                                                 | Filters on `task_latest.task` / `tasks.task`.                                                                 | Unchanged at IR level. Physical schema may later introduce separate task-definition tables, but `task` remains the semantic label.    |
| `task_keywords`| `List[str]`            | Keywords extracted from the task description for fuzzy matching.                | Currently used for LIKE / scoring in higher-level logic; not directly mapped by the core SQL builder.         | Remains a semantic helper for ranking / recall. Multi-table plans may use it against a dedicated `task_text` or `task_search` table. |
| `project`      | `Optional[str]`        | Project identifier/name (E3D, billing, etc.).                                   | Filters on `tasks.project` / `task_latest.project` (text column).                                             | Target for normalization: `tasks.project_id → projects.id`. IR stays as `project` string; KG + lookup resolve to `projects` rows.     |
| `tags`         | `List[str]`            | Semantic tags/labels inferred from the query or task description.               | Implemented as LIKE filters on `tasks.tags` / `task_latest.tags` (comma-separated string).                    | Prime candidate for multi-table: `tasks` JOIN `task_tags` JOIN `tags`. IR remains a list of tag names/slugs, normalized by KG-lite.   |
| `priority`     | `Optional[int]`        | Priority (1 = highest).                                                         | Filters on `tasks.priority` / `task_latest.priority`; also used in default sort order.                        | IR stays numeric. Physical schema can later introduce priority lookup tables without changing the IR.                                  |
| `status`       | `List[TaskStatus]`     | Required task statuses (TODO, IN_PROGRESS, BLOCKED, DONE, ANY).                 | Filters on `tasks.status` / `task_latest.status` via an `IN` clause.                                          | IR remains a semantic enum. Multi-table plans still project and filter on a status column; joins do not change this.                  |

### 3.3 Time range fields

| Field          | Type           | Business meaning                                              | Current physical mapping                                                | Future multi-table notes                                                                                 |
|----------------|----------------|----------------------------------------------------------------|-------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------|
| `time_range`   | `Optional[TimeRange]` | Time window for the status timestamp (usually “when was it updated?”). | Mapped to `ts` column (`tasks.ts` or `task_latest.ts`) via `>=` / `<=`. | IR remains “status-time window”. Physical storages may move to event tables; the planner will track which timestamp column to use. |
| `due_range`    | `Optional[TimeRange]` | Time window for due date (deadline).                         | Mapped to `due_ts` column.                                              | IR remains “due-time window”. Multi-table plans can join a task-definition table if due dates are moved there.                          |
| `created_range`| `Optional[TimeRange]` | Time window for task creation time.                          | Mapped to `created_ts` column.                                          | IR remains “created-time window”. Physical layout may change; IR stays unchanged.                                                           |

### 3.4 Sorting, limits, and extra filters

| Field        | Type                    | Business meaning                                                     | Current physical mapping                                                                                          | Future multi-table notes                                                                                          |
|--------------|-------------------------|----------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|
| `order_by`   | `List[OrderBySpec]`    | Sorting instructions (field + direction) for the result set.        | Translated into plan `sort` entries; rendered by `sql_builder` as `ORDER BY` on columns like `ts`, `id`, etc.    | IR expresses logical ordering. Multi-table plans will decide which table/alias each order field belongs to.      |
| `limit`      | `Optional[int]`        | Max number of rows to return.                                        | Clamped and enforced as a positional `LIMIT ?` parameter in SQL.                                                 | Unchanged. The limit is applied after joins/aggregations as defined by the plan.                                 |
| `filters`    | `List[QueryFilter]`    | Generic filters (field + op + value(s)) for advanced use cases.      | Normalized via `QueryFilter.to_plan_filter()` and appended to plan `filters`, then rendered by `sql_builder`.    | Serves as an escape hatch for new dimensions (e.g., `team`, `department`, `project_stage`) without changing IR shape. |

### 3.5 Extra / metadata

| Field   | Type                | Business meaning                                                   | Current physical mapping | Future multi-table notes                                                                 |
|---------|---------------------|--------------------------------------------------------------------|--------------------------|------------------------------------------------------------------------------------------|
| `extra` | `Dict[str, Any]`   | Free-form metadata: KG resolution info, model scores, debug flags. | Not mapped to DB.        | Safe place to record multi-table routing decisions or KG provenance, without affecting semantics. |

Example `extra` fields (non-exhaustive):

- `extra.nl2sql_source`: whether the IR was produced by LLM vs. rule-based parser.
- `extra.kg_enabled`: whether KG-lite resolution was used.
- `extra.kg_person_source`, `extra.kg_project_source`, `extra.kg_category_source`: how entities were mapped.
- Future: `extra.query_plan_version` to record whether v1 or v2 multi-table planner was used.

---

## 4. Planned multi-table scenarios

The first multi-table scenarios are planned to be:

1. **Tasks ↔ persons ↔ department / team**

   - Logical question: “For a given department/team, how many overdue tasks are there?”
   - IR expression: extend `TaskQuerySpec` with either:
     - a dedicated field (`department` / `team`), or
     - a generic filter (`filters: [{"field": "team", "op": "eq", ...}]`).
   - Physical plan: join `tasks` with `persons` via `tasks.person_id = persons.id`
     and group by `persons.team`.

   The IR remains the same conceptually (“person”, “team”); only the IR→plan→SQL
   layer learns to express the join.

2. **Tasks ↔ projects**

   - Logical question: “For project X, what is the status of all tasks?”
   - IR expression: `project` field + `answer_mode` (e.g., `person_summary_by_project`).
   - Physical plan (future):
     - Introduce a `projects` dimension table and `tasks.project_id` FK.
     - Use KG-lite to normalize `spec.project` into a `projects` row.
     - Plan expresses joins to `projects` and uses `projects.code` / `projects.name`
       for grouping/selection.

3. **Tasks ↔ tags**

   - Logical question: “Show the status of all tasks tagged with A and B.”
   - IR expression: `tags: ["A", "B"]`.
   - Physical plan (future):
     - Introduce `tags` and `task_tags` tables.
     - Use KG-lite to normalize tag names.
     - Plan expresses joins `tasks -> task_tags -> tags` with appropriate filters.

In all cases, the IR (`TaskQuerySpec`) and KG-lite resolution API are kept
stable. Multi-table capabilities are introduced exclusively inside:

- The IR→plan layer (e.g., `build_query_plan_v2`), and
- The SQL builder (`sql_builder` or a v2 variant).

This document should be updated whenever we:

- Add new semantic fields to `TaskQuerySpec`, or
- Change how existing fields map to physical tables/joins.

