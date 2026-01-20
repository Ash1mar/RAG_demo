# Task Status Q&A (Step 2) - Design Notes

This document explains how the "task status Q&A" subsystem works: the `/tasks/ask` endpoint, the NL->JSON IR (`TaskQuerySpec`), the KG-lite semantic dictionary, the SQL compiler, and the optional Text2SQL branch. Use it together with `docs/START_AND_TEST.md` for end-to-end setup instructions.

---

## 1. Scope & Endpoints

The subsystem answers questions like "Has Zhang San finished task X?", "List Zhang San's TODO tasks this week", "How many tasks are still not done?", or "Give me a status summary for Zhang San and Li Si."

Endpoints:

- `GET /tasks/ask?q=...&topk=3&thresh=0.45`
  - Main task Q&A endpoint, returning a Chinese answer plus debugging fields (`sql`, `rows`, `nl_ir`, `candidates`, etc.).
  - Controlled by `RESOLVER` (rules / embeddings / hybrid / hybrid_plus_rules / hybrid_llm / text2sql).

- `GET /db/ask?q=...`
  - NL->JSON->SQL experiment endpoint over the `tasks`/`task_latest` tables.
  - Returns `query`, `ir`, `sql`, `params`, `rows` only (no natural-language answer).

---

## 1.1 Configuration Quick Note

- Optional centralized env file: `config/app.env` (loaded at startup).
- Override via `APP_CONFIG=path/to/your.env`.
- Tasks backend/dialect:
  - `TASKS_BACKEND=sqlite|mssql`
  - `TASKS_DIALECT=sqlite|mssql` (IR->SQL compiler)
  - `TASKS_TEXT2SQL_DIALECT=sqlite|mssql` (Text2SQL prompt/validation override)

---

## 2. Resolver Modes (`RESOLVER`)

- `rules`: keyword and string matching; baseline for exact names.
- `embeddings`: embedding-only "Focus Query" resolver (matrix scoring).
- `hybrid`: embeddings + keyword heuristics for better robustness.
- `hybrid_plus_rules`: `hybrid` plus additional rule assists for gating.
- `hybrid_llm`: LLM NL->JSON (`TaskQuerySpec`) + small model candidate alignment + unified SQL compiler.
- `text2sql`: LLM-generated SQL under AST validation (can also be triggered from `hybrid_llm` when complex analytics are needed).

`RESOLVER` is set via environment variable; change it and restart the API (or call `/tasks/reload`) to switch modes.

---

## 3. Tasks Table Schema

The tasks backend exposes two logical views over the same schema (`tasks` and `task_latest`), regardless of SQLite or SQL Server:

```sql
CREATE TABLE tasks (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  person      TEXT NOT NULL,
  task        TEXT NOT NULL,
  status      TEXT NOT NULL,        -- DONE / TODO / IN_PROGRESS / BLOCKED
  ts          INTEGER NOT NULL,     -- epoch millis ("status time")
  project     TEXT,
  tags        TEXT,                 -- comma-separated strings
  priority    INTEGER,              -- 1 = highest
  due_ts      INTEGER,
  created_ts  INTEGER,
  updated_ts  INTEGER,
  status_note TEXT
);
```

- `task_latest`: latest row per `(person, task)` (used for "current status / list" queries).
- `tasks`: full history (used for `task_history`, summaries, completion times).

Use compatibility views (`tasks`, `task_latest`) with the same logical columns when switching to SQL Server; set `TASKS_DIALECT=mssql` so the compiler emits T-SQL (`TOP`).

The SQL compiler only emits read-only `SELECT` statements referencing these tables and always enforces a row cap (SQLite uses `LIMIT`, SQL Server uses `TOP`).

---

## 4. NL->JSON IR: `TaskQuerySpec`

Located in `app/services/nl2sql_engine.py`, this Pydantic model captures the semantics of each task query.

Key fields:

- `intent: TaskQueryIntent`
  - `task_status_single`, `task_status_list`, `task_list_by_person`, `task_history`, `person_summary`, `unknown`.
- `answer_mode: TaskAnswerMode`
  - `default`, `completion_time_latest`, `task_count_by_status`, `person_summary_by_project`, `overdue_count_by_person`.
- Entities: `person`, `task`, `task_keywords`, `project`, `tags`, `priority`.
- Status list: `status: List[TaskStatus]` (can be empty -> "no restriction").
- Time windows: `time_range`, `due_range`, `created_range` (`TimeRange` models storing `start`/`end` tokens such as `now-7d`, `start_of_week`, ISO timestamps, etc.).
- Safety knobs: `order_by`, `limit`, `filters`, `extra` (for heuristics, KG flags, parse details).

### NL->IR flow

`parse_task_query_nl(q)` proceeds as follows:

1. If `TASKS_NL2SQL_LLM=1` and an LLM provider is configured, call `llm_client.generate_task_query_spec(q)` to get a JSON IR. Parse it into `TaskQuerySpec`. On error, fall back to rules.
2. Apply `_post_process_intent` (common heuristics):
   - detect status keywords if `status` is empty;
   - detect time/due/created ranges, priority hints, limit/order safety;
   - normalize fields via KG-lite;
   - record debug info in `spec.extra` (e.g. `nl2sql_source=llm`/`rules`, `kg_enabled`, etc.).
3. If LLM is disabled or fails, `_rule_based_parse_task_query_nl_v2` builds a spec using keyword rules and then runs `_post_process_intent` as above.

---

## 5. KG-lite: persons/projects/categories/status/priority dictionary

Rather than scattering alias mappings and category->tag expansions across Python code or prompts, we use a lightweight KG-lite layer.

- **Location**
  - Code: `app/services/kg_lite.py`
  - Data: `data/kg_data.json`

- **Schema**
  - `persons`: canonical names + aliases (e.g. `"张工"` / `"老张"` -> `"张三"`).
  - `projects`: canonical project/system names + aliases (e.g. `"芯片平台"` -> `"芯片"`).
  - `categories`: `{name, aliases, tags}` entries (e.g. `"安全专项"` / `"安监专项"` map to `"安监整改"` with tags `["整改","安全整改"]`).
  - `statuses`: canonical TaskStatus names (`DONE/TODO/IN_PROGRESS/BLOCKED`) and synonyms ("完成/搞定/已完成"等).
  - `priorities`: canonical integers (`1/2/3`) plus synonyms (`P1`, `高优`, `最高优先级` 等).

- **Backend abstraction**
  - `KGBackend(Protocol)` defines finder methods for person/project/category tags/status/priority plus `snapshot()`.
  - `InMemoryKGBackend` loads JSON into an in-memory `KGData` structure.
  - `KGResolver` exposes module-level helpers:
    - `resolve_person(name)`
    - `resolve_project(project, text)`
    - `resolve_category_tags(text)`
    - `resolve_status_value(value)`
    - `resolve_priority_value(value)`
    - `get_debug_snapshot()`

- **Integration points**
  - NL->JSON (`parse_task_query_nl` -> `_post_process_intent`):
    - normalize person/project/tags and record `kg_person_source` / `kg_project_source` / `kg_category_source` when they change;
    - normalize `spec.status` by mapping strings or enums to canonical `TaskStatus` values;
    - normalize `spec.priority` (e.g. `P1` / "高优" -> `1`);
    - set `spec.extra["kg_enabled"] = True` when any change occurs.
  - Text2SQL:
    - `task_query._make_text2sql_ir_hint(spec)` includes canonical person/project/tags/status/priority in the IR hint used in the prompt;
    - `_rewrite_text2sql_query(sql, hint)` uses those hints to rewrite SQL literals (`person = '张工'` -> `'张三'`, `priority = '高优'` -> `1`) and to inject missing `tags LIKE '%整改%'` filters.
    - Rewrite symbolic time literals (e.g. `now+7d`, `next_week`) and common T-SQL datetime math into epoch-ms constants.

- **Debugging KG-lite**
  - `/tasks/ask` includes `kg_enabled` at the top level when KG-lite intervenes.
  - `payload["nl_ir"]["extra"]` may include `kg_person_source`, `kg_project_source`, `kg_category_source` (printed by `scripts/batch_db_ask.py`).

---

## 6. SQL Compiler (`compile_tasks_sql`)

Located in `app/services/sql_compiler.py`. Responsibilities:

- Validate that required fields are present (e.g. single-status queries must include person+task scope).
- Convert `TaskQuerySpec` into a generic plan (`build_task_query_plan`) and then into SQL via `app/sql_builder.py`.
- Clamp `LIMIT`, enforce read-only `SELECT`, and prohibit cross-table access.
- Raise `TaskSqlCompileError` when IR is incomplete or intent unsupported.

---

## 7. Responses & Debug Fields

### `/tasks/ask`

Typical payload includes:

- `answer`: Chinese response (or Text2SQL summary).
- `resolver_mode`: `rules` / `embeddings` / `hybrid` / `hybrid_plus_rules` / `hybrid_llm` / `text2sql`.
- `intent`: string label derived from `TaskQueryIntent`.
- `sql`, `params`, `rows`: the SQL query executed (for DB-backed modes).
- `candidates`: person/task candidate scores (for non-Text2SQL resolvers).
- `nl_ir`: serialized `TaskQuerySpec`, including `extra.nl2sql_source` and `extra.kg_*` flags.
- `kg_enabled`: surfaced when KG-lite modified the IR.
- Text2SQL-specific fields: `text2sql`, `text2sql_model`, `text2sql_provider`, `text2sql_raw_response`, `error`, `reason`, etc.

### `/db/ask`

Always returns a structured snapshot of the NL->IR->SQL pipeline:

```json
{
  "query": "...",
  "ir": { ... TaskQuerySpec JSON ... },
  "sql": "SELECT ...",
  "params": ["张三", "E3D接口联调", 1],
  "rows": [ ... ]
}
```

If SQL compilation fails, `/db/ask` returns 4xx with `detail.error` and `detail.reason` describing the issue.

---

## 8. Text2SQL Branch (`resolver_mode="text2sql"` or `hybrid_llm`)

Located in `app/services/task_query.py` (`_answer_via_text2sql`):

1. Build the Text2SQL prompt with schema + IR hint, selecting SQLite or T-SQL rules via `TASKS_TEXT2SQL_DIALECT`/`TASKS_DIALECT`.
2. Call the configured LLM (e.g. `qwen3-coder:480b-cloud`) to obtain JSON `{ "queries": [{ "sql": "...", "description": "..." }, ...] }`.
3. For each SQL:
   - `_rewrite_text2sql_query` aligns time windows, tag filters, person/project literals, priority hints, etc.
   - `_normalize_and_validate_text2sql_query` (sqlglot) enforces read-only `SELECT`, allowed tables (`task_latest`/`tasks`), and a hard row cap (SQLite `LIMIT`, SQL Server `TOP`).
   - Execute via `TasksStore.query` (SQLite or SQL Server backend).
4. Optionally call `_generate_text2sql_answer` (LLM summarization) or `_summarize_text2sql_rows` for a lightweight summary.
5. Report errors (`text2sql_invalid_sql`, `text2sql_db_query_failed`, etc.) with detailed `reason` and raw LLM output when something fails.

---

## 9. Self-Test & Tuning

- **Pytest**: `pytest -q tests/test_nl2sql_db_ask.py` validates IR parsing, SQL compilation, and `/db/ask` responses end-to-end.
- **Batch testing**: `python scripts/batch_db_ask.py --file scripts/questions.txt`
  - Runs common questions (including alias-heavy and category-driven prompts) through `/tasks/ask`.
  - Prints `resolver_mode`, `intent`, `KG enabled`, `KG person/category/project source`, Text2SQL info if applicable.
- **KG updates**: `python scripts/extract_kg_from_tasks.py`
  - Outputs a KG-lite candidate JSON structure (canonical values only, aliases empty).
  - Workflow suggestion (documented in README/START guide):
    1. Run the script to generate `kg_data.generated.json`.
    2. Review/merge canonical names into `data/kg_data.json` (add aliases/tags manually or via additional tooling).
    3. For query alias gathering, future scripts can analyze query logs to propose new aliases for approval.

Use these tools to keep the NL->IR->SQL pipeline stable as you add more real data and complex queries.
