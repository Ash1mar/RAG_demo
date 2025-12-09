# Task Status Q&A (Step 2) – Design Notes

This document explains the “task status Q&A” part of the demo: how `/tasks/ask` and `/db/ask` work, what the `TaskQuerySpec` IR looks like, how KG-lite (the semantic dictionary) plugs in, and how to debug / tune the pipeline.

---

## 1. Scope & Endpoints

The task subsystem focuses on queries like “Has Zhang San finished task X?”, “List Zhang San’s TODO tasks this week”, “How many tasks are still not done?”, etc.

Endpoints:

- `GET /tasks/ask?q=...&topk=3&thresh=0.45`
  - Main task Q&A endpoint, returns a Chinese answer plus internal debugging fields.
  - Supports multiple resolver modes configured via `RESOLVER`.

- `GET /db/ask?q=...`
  - NL→JSON→SQL experimental endpoint over the `tasks` / `task_latest` tables.
  - Returns IR/SQL/rows only; no natural‑language answer.

---

## 2. Resolver Modes (`RESOLVER`)

The `RESOLVER` env var controls how `/tasks/ask` resolves person/task entities and builds answers:

- `rules`
  - Pure rules‑based baseline (string matching + heuristics).
  - Good for exact matches and transparent behavior.

- `embeddings`
  - Embedding‑only resolver using a “Focus Query” matrix over candidate persons/tasks.
  - Better robustness to typos and paraphrases than pure rules.

- `hybrid`
  - Combines embeddings + keyword heuristics for more stable scoring.
  - Uses FAISS + simple rule scores for gating and tie‑breaking.

- `hybrid_plus_rules`
  - `hybrid` plus extra rule assists in ranking and gating.
  - Slightly more complex logic, still non‑LLM.

- `hybrid_llm`
  - LLM NL→JSON + small model + FAISS + SQL compiler:
    1. `parse_task_query_nl` builds a `TaskQuerySpec` from NL (LLM first if `TASKS_NL2SQL_LLM=1`, otherwise rules).
    2. `TaskQueryEngine` uses embeddings + FAISS to align `spec.person` / `spec.task` with the candidate lists.
    3. `compile_tasks_sql` compiles the aligned `TaskQuerySpec` into safe, read‑only SQL.
    4. `SQLiteTasksStore` runs the query; `/tasks/ask` synthesizes a Chinese answer.

- `text2sql`
  - Text2SQL pipeline: LLM under strict AST validation generates SQL directly.
  - Often used together with the IR hint from `TaskQuerySpec`; see section 8.

---

## 3. Tasks Table Schema

The demo uses a simple SQLite schema (see `docs/START_AND_TEST.md` for initialization):

```sql
CREATE TABLE tasks (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  person      TEXT NOT NULL,
  task        TEXT NOT NULL,
  status      TEXT NOT NULL, -- DONE | TODO | IN_PROGRESS | BLOCKED
  ts          INTEGER NOT NULL, -- epoch millis; logical “status time”
  project     TEXT,
  tags        TEXT,           -- comma-separated strings
  priority    INTEGER,        -- 1 = highest priority
  due_ts      INTEGER,
  created_ts  INTEGER,
  updated_ts  INTEGER,
  status_note TEXT
);
```

Two logical views are used:

- `tasks`: full history (multiple rows per person+task); used for `task_history`/aggregations.
- `task_latest`: latest status per (person, task); used for most “current status / list” queries.

The SQL compiler (`app/services/sql_compiler.py`) only emits **read‑only** `SELECT` statements with a `LIMIT` and references to these tables.

---

## 4. NL→JSON IR: `TaskQuerySpec`

The semantic IR for task queries is the `TaskQuerySpec` Pydantic model (`app/services/nl2sql_engine.py`). It is produced by `parse_task_query_nl(q)`.

Key fields (not exhaustive):

- **High‑level intent**
  - `intent: TaskQueryIntent`
    - `task_status_single`: latest status for a single (person, task) pair.
    - `task_status_list`: multiple rows, usually a list of tasks/statuses.
    - `task_list_by_person`: list tasks for a given person.
    - `task_history`: status history for a person+task (timeline / completion time).
    - `person_summary`: aggregated counts by person/status.
    - `unknown`: fallback when intent cannot be determined.
  - `answer_mode: TaskAnswerMode`
    - `default`: standard row‑based answer.
    - `completion_time_latest`: answer with “when it was completed” (latest DONE row).
    - `task_count_by_status`: group counts by status.
    - `person_summary_by_project`: summary per project/person/status.
    - `overdue_count_by_person`: count of overdue tasks per person.

- **Core entities**
  - `person: Optional[str]`
  - `task: Optional[str]`
  - `task_keywords: List[str]`
  - `project: Optional[str]`
  - `tags: List[str]`
  - `priority: Optional[int]`
  - `status: List[TaskStatus]` (`DONE`, `TODO`, `IN_PROGRESS`, `BLOCKED`, `ANY`)

- **Time ranges**
  - `time_range: Optional[TimeRange]` – generally over `ts`.
  - `due_range: Optional[TimeRange]` – due time window (`due_ts`).
  - `created_range: Optional[TimeRange]` – created time window (`created_ts`).

- **Query shape & safety**
  - `order_by: List[OrderBySpec]`
  - `limit: Optional[int]` (bounded in code).
  - `filters: List[QueryFilter]`
  - `extra: Dict[str, Any]` – debug info, parse details, KG/LLM flags, etc.

### 4.1 NL→IR flow: LLM first, rules as fallback

`parse_task_query_nl(q: str) -> TaskQuerySpec`:

1. If `TASKS_NL2SQL_LLM=1` and LLM is configured:
   - Call `llm_client.get_llm_client().generate_task_query_spec(q)` to get a JSON IR.
   - Parse into `TaskQuerySpec`, then run `_post_process_intent` for heuristics (status detection, time hints, limit/order safety, KG-lite, etc.).
   - Set `spec.extra["nl2sql_source"] = "llm"`.
2. On error or if LLM is disabled:
   - Run `_rule_based_parse_task_query_nl_v2(q)` to build a spec using keyword rules.
   - Apply `_post_process_intent` in the same way.
   - Set `spec.extra["nl2sql_source"] = "rules"` and optionally record `nl2sql_llm_error` if LLM failed.

---

## 5. KG-lite: persons/projects/categories/tags dictionary

To avoid scattering business mappings (aliases, nicknames, category→tag expansions) across prompts and code, the task pipeline uses a light‑weight KG-lite layer.

- **Location**
  - Code: `app/services/kg_lite.py`
  - Data: `data/kg_data.json`

- **Schema**
  - `persons`: list of `{canonical, aliases}` entries, e.g.:
    - `"张三"` with aliases `["张工", "老张"]`
    - `"李四"` with aliases `["李工"]`
  - `projects`: `{canonical, aliases}` entries, e.g.:
    - `"芯片"` with aliases `["芯片项目", "芯片平台"]`
    - `"交付"` with aliases `["交付项目", "交付项目组"]`
    - `"E3D"` with aliases `["E3D项目", "E3D系统"]`
  - `categories`: `{name, aliases, tags}` entries, e.g.:
    - `"安监整改"` with aliases `["整改任务", "安监专项", "安全专项"]`, expanding to tags `["整改", "安全整改"]`.

- **Backend abstraction**
  - `KGBackend(Protocol)` defines:
    - `find_person(name) -> Optional[PersonEntry]`
    - `find_project(project, text) -> Optional[ProjectEntry]`
    - `find_category_tags(text) -> List[str]`
    - `snapshot() -> Dict[str, List[str]>`
  - `InMemoryKGBackend` loads JSON into an in‑memory `KGData` structure.
  - `KGResolver` wraps a backend and exposes high‑level helpers:
    - `resolve_person(name)`
    - `resolve_project(project, text)`
    - `resolve_category_tags(text)`
    - `debug_snapshot()`

- **Integration**
  - In `parse_task_query_nl` → `_post_process_intent`:
    - `kg_lite.resolve_person(spec.person)` normalizes `person`; if it changes, `spec.extra["kg_person_source"]` records the original string.
    - `kg_lite.resolve_project(spec.project, text)` normalizes `project` from explicit field or context text; source goes into `kg_project_source`.
    - `kg_lite.resolve_category_tags(text)` returns a list of tags inferred from the whole question; these are merged into `spec.tags`, and `kg_category_source` records the text used.
    - If any of the above changes occur, `spec.extra["kg_enabled"] = True`.
  - In Text2SQL (`resolver_mode="text2sql"` or `hybrid_llm` Text2SQL branch):
    - `task_query._make_text2sql_ir_hint(spec)` includes canonical `person` / `project` / `tags` in the IR hint used inside the Text2SQL prompt.
    - `_rewrite_text2sql_query(sql, hint)` uses these hints to rewrite literals, e.g.:
      - `person = '张工'` → `person = '张三'`
      - `project = '芯片平台'` → `project = '芯片'`
      - inject `tags LIKE '%整改%'` when IR/tags imply the “安监整改/安全专项” category.

- **Debugging KG‑lite**
  - `/tasks/ask` top‑level payload may contain:
    - `kg_enabled: true|false`
  - `payload["nl_ir"]["extra"]` may include:
    - `kg_person_source`, `kg_project_source`, `kg_category_source`
  - `scripts/batch_db_ask.py` prints these flags for each question so you can see when KG-lite actually changed the IR.

---

## 6. SQL Compiler (`compile_tasks_sql`)

Location: `app/services/sql_compiler.py`

Responsibilities:

- Accept a `TaskQuerySpec` and emit a safe `SELECT` SQL string + positional params (`CompiledSql`).
- Enforce intent‑specific constraints:
  - `task_status_single`: requires both `person` and `task`, always `LIMIT 1`, uses `task_latest`.
  - `task_status_list` / `task_list_by_person` / `task_history`: enforce reasonable limits (clamped), interpret status/time/tag filters, choose between `task_latest` and `tasks` depending on intent.
  - `person_summary`: requires at least one person scope (field or filter), emits `COUNT(*) AS task_count` with `GROUP BY person, status`.
- Prohibit writes / cross‑table access:
  - Only emits `SELECT` statements.
  - Only references `task_latest` / `tasks` tables via `build_task_query_plan` + `build_sql_from_ir`.

If a spec is obviously incomplete (e.g. missing person/task scope for a single‑status query) or intent is unsupported, `compile_tasks_sql` raises `TaskSqlCompileError`, which `/db/ask` turns into a 4xx response with a `detail.reason` message.

---

## 7. Responses

### 7.1 `/tasks/ask`

Typical top‑level fields (some are mode‑dependent):

- `answer`: Chinese text answer (or Text2SQL summarization).
- `resolver_mode`: `rules` / `embeddings` / `hybrid` / `hybrid_plus_rules` / `hybrid_llm` / `text2sql`.
- `intent`: high‑level intent label (string), derived from `TaskQueryIntent`.
- `sql`: the actual SQL used (for resolvers that hit the DB).
- `params`: parameters for the SQL.
- `rows`: raw DB rows returned by the query (or preview rows for Text2SQL).
- `candidates`: person/task candidate lists and scores (for non‑Text2SQL resolvers).
- `nl_ir`: serialized `TaskQuerySpec` (including `extra.kg_*` and `extra.nl2sql_*` hints).
- `kg_enabled`: whether KG-lite changed the IR for this query (propagated from `nl_ir.extra`).
- Text2SQL‑specific fields:
  - `text2sql`: list of queries with `sql` / `description` / `rows`.
  - `text2sql_model`, `text2sql_provider`.
  - `text2sql_raw_response` (when LLM output is invalid).
  - `error`, `reason`, `invalid_sql` (when Text2SQL fails).

### 7.2 `/db/ask`

Always returns a structured snapshot of the NL→IR→SQL pipeline:

- `query`: original NL query string.
- `ir`: serialized `TaskQuerySpec` (after heuristics + KG-lite).
- `sql`: compiled SQL string.
- `params`: positional parameters tuple.
- `rows`: raw rows from the tasks DB.

If the IR is incomplete or cannot be safely compiled, `/db/ask` returns 4xx with:

```json
{
  "detail": {
    "error": "cannot_compile_sql",
    "reason": "task_status_single requires both person and task"
  }
}
```

---

## 8. Text2SQL Details (resolver_mode="text2sql" / hybrid_llm branch)

Location: `app/services/task_query.py`

Flow for `_answer_via_text2sql`:

1. Build `base` payload, including `resolver_mode="text2sql"` and, if available, `nl_ir` from `TaskQuerySpec`.
2. Construct the Text2SQL prompt via `_build_text2sql_prompt(question, spec)`:
   - includes the `TEXT2SQL_SCHEMA` and IR hint returned by `_make_text2sql_ir_hint(spec)` (canonical person/project/tags, time/due/created ranges, filters, limit/order).
3. Call the configured LLM (`qwen3-coder:480b-cloud` by default) via `_call_text2sql_llm` to obtain a `Text2SQLResponseModel` (`queries: List[{sql,description}]`).
4. For each SQL (up to `TEXT2SQL_MAX_QUERIES`):
   - Apply `_rewrite_text2sql_query(sql, hint)`:
     - align time literals (`ts`, `created_ts`, `due_ts`) with IR ranges;
     - inject tag filters when IR has tags but SQL does not;
     - normalize `person` / `project` literals using KG‑aware hints;
     - clean up invalid `ORDER BY` tokens, strip semicolons;
     - enforce `priority = 1` where “高优P1” semantics apply.
   - Parse and validate via `_normalize_and_validate_text2sql_query` (sqlglot AST):
     - ensure only `SELECT`, only `task_latest`/`tasks` tables;
     - cap or insert `LIMIT` (≤100);
     - disallow suspicious comparisons and unsafe functions;
     - disallow positional / named placeholders (`?`, `:name`).
   - Execute the normalized SQL against `TasksStore.query` (no parameters; literals are in SQL).
5. Construct the final payload:
   - `text2sql`: details for each executed query;
   - `rows`: rows of the primary query;
   - `answer`: either summarized by a second LLM call (`_generate_text2sql_answer`) or a simple row summary (`_summarize_text2sql_rows`).

On any failure (LLM output wrong shape, invalid SQL, DB error), the `error` and `reason` fields are filled and no SQL is executed (for invalid SQL cases).

---

## 9. Self‑Testing & Tuning

- Use `tests/test_nl2sql_db_ask.py` (via `pytest`) to lock down:
  - IR parsing (`parse_task_query_nl`),
  - SQL compilation (`compile_tasks_sql`),
  - `/db/ask` behavior and error handling.
- Use `scripts/batch_db_ask.py` with `scripts/questions.txt` to:
  - compare behavior of different `RESOLVER` modes on the same question set;
  - observe how `nl_ir.extra.nl2sql_source` and `nl_ir.extra.kg_*` evolve;
  - debug Text2SQL answers vs. NL→JSON→SQL answers.
- Use `scripts/extract_kg_from_tasks.py` to preview distinct `person` / `project` / `tags` from your tasks DB and decide how to populate or extend `data/kg_data.json` as your domain grows.

