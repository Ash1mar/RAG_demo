# Start and Test Guide – Minimal RAG Demo

This guide walks you from a clean clone to a working demo:

- API up and running (FAISS store by default).
- Sample docs ingested and searchable.
- Task status Q&A (`/tasks/ask`) working in non‑LLM and LLM/SQL modes.
- NL→JSON→SQL pipeline (`/db/ask`) validated via pytest and curl.
- KG-lite / Text2SQL experiments available for inspection.

All commands assume project root as current directory.

---

## 0) Prerequisites

- Windows 10/11 (PowerShell) or macOS/Linux (Bash)
- Python 3.9+ (3.10/3.11 recommended)
- Docker Desktop with WSL 2 backend (for embedding / Milvus, optional)
- Git (optional)

For initial runs, you can avoid model downloads by using mock embeddings (`MOCK_EMB=1`).

---

## 1) Prepare Project and Python Env

PowerShell (Windows):

```powershell
python -m venv venv
./venv/Scripts/Activate
pip install -r requirements.txt

# Optional: use mock embeddings to avoid downloading models
$env:MOCK_EMB = '1'
```

Bash (macOS/Linux/WSL):

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
export MOCK_EMB=1
```

---

## 2) Initialize Demo Tasks DB

The task Q&A endpoints use a small SQLite DB (`data/tasks.db`). Initialize it once:

```bash
python scripts/init_tasks_sqlite.py
```

This creates and populates `data/tasks.db` with a handful of demo tasks for 张三 / 李四 and a few projects/tags.

### 2.1) (Optional) Import Enterprise Tasks (FACT_TASK_ASSIGN) into SQLite

If you have an enterprise single-table fact like `data/FACT_TASK_ASSIGN.csv`, import it into a separate SQLite DB and expose the canonical views (`tasks`, `task_latest`) so the existing pipeline can keep working without code changes.

PowerShell:

```powershell
# Input CSV (default: data/FACT_TASK_ASSIGN.csv)
$env:FACT_TASK_ASSIGN_CSV = 'data/FACT_TASK_ASSIGN.csv'

# Output SQLite DB (default: data/fact_tasks.db)
$env:TASKS_DB = 'data/fact_tasks.db'

python scripts/import_fact_task_assign_sqlite.py
```

Expected output:

- `Imported ...FACT_TASK_ASSIGN.csv into ...fact_tasks.db`
- `Created views: tasks, task_latest`

Notes:

- The importer creates a physical table `FACT_TASK_ASSIGN` and two compatibility views: `tasks` and `task_latest`.
- To switch between demo DB and enterprise DB, you only need to change `TASKS_DB` and restart the API.

### 2.2) Switch Tasks DB (Demo vs Enterprise)

The task pipeline reads the SQLite path from `TASKS_DB`.

- Demo: `TASKS_DB=data/tasks.db` (created by `scripts/init_tasks_sqlite.py`)
- Enterprise: `TASKS_DB=data/fact_tasks.db` (created by `scripts/import_fact_task_assign_sqlite.py`)

Always restart `uvicorn` after changing env vars.

### 2.3) (Optional) Portability via `TASKS_COL_*` field mapping

Most datasets should expose compatibility views named `tasks` / `task_latest` with the canonical columns (`person`, `task`, `status`, `ts`, `project`, `tags`, ...).

If your dataset uses different column names and you prefer not to create views, you can map logical fields to physical columns via env vars:

- `TASKS_COL_PERSON`, `TASKS_COL_TASK`, `TASKS_COL_STATUS`, `TASKS_COL_TS`, `TASKS_COL_PROJECT`, `TASKS_COL_TAGS`, ...
- Enterprise extensions: `TASKS_COL_OWNER`, `TASKS_COL_DIVISION_NAME`, `TASKS_COL_IS_READ`, `TASKS_COL_IS_DELEGATED`, ...

This is intended to reduce future migrations to "new DB + a few env vars" instead of code changes.

---

## 3) Quick Run – FAISS + Mock Embeddings (Recommended Start)

FAISS is the default vector store and works well on Windows/macOS/Linux.

PowerShell:

```powershell
$env:STORE = 'faiss'
$env:MOCK_EMB = '1'
uvicorn app.demo_app:app --host 0.0.0.0 --port 8000 --reload
```

Health check (new terminal):

```powershell
curl "http://127.0.0.1:8000/health"
```

Expected fields include:

- `status: "ok"`
- `embedder: "mock" or "sbert"`
- `vector_store: "faiss" or "milvus"`
- `tasks_store: "sqlite"`
- `tasks_ready: true`
- `resolver_mode`: current `RESOLVER` (e.g. `hybrid`)

---

## 4) Ingest Sample Text and Test RAG

With the API running:

PowerShell example:

```powershell
$b1 = @{
  doc_id = "report-2024"
  text   = "Milvus brings filters and shared indexing to this demo."
  source = "demo-notes"
  ts     = "2024-06-01T10:15:00Z"
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://127.0.0.1:8000/ingest" -Method POST -ContentType "application/json" -Body $b1
```

Then test search:

```powershell
curl "http://127.0.0.1:8000/search?q=Milvus&k=5"
curl "http://127.0.0.1:8000/search_hybrid?q=Milvus&k=5&alpha=0.5"
curl -X POST "http://127.0.0.1:8000/reset"
```

If you see documents in `results`, the RAG part is working.

---

## 5) Task Status – Non‑LLM Ask (Baseline)

With `data/tasks.db` initialized, you can query task status using only rules/embeddings (no LLM).

Set resolver mode and restart API if needed:

```powershell
$env:STORE = 'faiss'
$env:RESOLVER = 'hybrid'    # or 'rules' / 'embeddings' / 'hybrid_plus_rules'
$env:MOCK_EMB = '1'         # mock embeddings are fine for baseline
uvicorn app.demo_app:app --host 0.0.0.0 --port 8000 --reload
```

Example queries (PowerShell):

```powershell
curl "http://127.0.0.1:8000/tasks/ask?q=张三的提交9月周报完成了吗？"
curl "http://127.0.0.1:8000/tasks/ask?q=张三的E3D接口联调现在什么状态？"
curl "http://127.0.0.1:8000/tasks/ask?q=李四的整理工艺包V2是否已完成？"
curl 'http://127.0.0.1:8000/tasks/ask?q=老张九月周报搞定了没？'
```

You should see answers like “已完成 / TODO / IN_PROGRESS” along with the latest timestamps.

---

## 6) Enable Small Model – Hybrid / Hybrid_LLM

To compare with a real Chinese embedding model, start the Dockerized bge-small-zh embedder:

```powershell
docker compose build embedder
docker compose up -d embedder

$env:STORE    = 'faiss'
$env:RESOLVER = 'hybrid'        # or 'embeddings' / 'hybrid_llm'
$env:MOCK_EMB = '0'
$env:EMB_URL  = 'http://localhost:8080/embeddings'
$env:EMB_DIM  = '512'           # bge-small-zh

uvicorn app.demo_app:app --host 0.0.0.0 --port 8000 --reload
```

Re‑run the same `/tasks/ask` queries and observe:

- With `RESOLVER=rules`, exact matches work well but aliases/typos are brittle.
- With `RESOLVER=hybrid` or `embeddings`, alias/colloquial queries (e.g. “老张九月周报搞定了没”) should be more robust.

---

## 7) NL→JSON→SQL Experiments (`/db/ask`)

To inspect the full NL→IR→SQL pipeline for tasks, use `/db/ask`:

```powershell
curl "http://127.0.0.1:8000/db/ask?q=张三的E3D接口联调现在什么状态？"
```

You should see a JSON payload:

- `query`: original NL.
- `ir`: `TaskQuerySpec` (intent, person, task, status, time_range, tags, filters, extra).
- `sql`: compiled SQL (read‑only, with `LIMIT`).
- `params`: positional parameters.
- `rows`: raw rows from `tasks` or `task_latest`.

If the IR is incomplete or unsafe (e.g., missing person/task), `/db/ask` will return 4xx with a `detail.reason` explaining why SQL compilation failed.

---

## 8) Text2SQL – Config and Basic Test

Text2SQL is optional and requires an LLM provider configured via `LLM_*` env vars.

Example (Ollama, with a local `qwen3-coder:480b-cloud` model; adjust to your environment):

```powershell
$env:LLM_ENABLED = 'true'
$env:LLM_PROVIDER = 'ollama'
$env:LLM_MODEL = 'qwen2.5-coder:7b'             # for general NL→IR (if used)
$env:LLM_TEXT2SQL_MODEL = 'qwen3-coder:480b-cloud' # for Text2SQL
$env:RESOLVER = 'text2sql'

uvicorn app.demo_app:app --host 0.0.0.0 --port 8000 --reload
```

Then call:

```powershell
curl "http://127.0.0.1:8000/tasks/ask?q=张三的E3D接口联调现在什么状态？"
```

You should see:

- `resolver_mode: "text2sql"`
- A `text2sql` array with at least one `{sql, description, rows}` entry.
- `answer` summarizing the latest status from the returned rows (or a “no rows” message).

For stricter NL→JSON + Text2SQL combination, use:

```powershell
$env:RESOLVER = 'hybrid_llm'
$env:TASKS_NL2SQL_LLM = '1'
```

and repeat the query. The pipeline will be:

1. LLM (or rules) → `TaskQuerySpec` (`nl_ir`).
2. Hybrid small‑model resolver to align person/task.
3. SQL compiler to generate safe SQL.
4. (Optional) Text2SQL branch for complex analytics, under AST validation.

---

## 9) Pytest – NL→JSON→SQL End‑to‑End

There is a minimal pytest module to validate the NL→IR→SQL pipeline, independent of the UI:

```bash
pytest -q tests/test_nl2sql_db_ask.py
```

It covers three layers:

- IR layer: `parse_task_query_nl("张三的E3D接口联调现在什么状态？")` produces a `TaskQuerySpec` with correct `intent`/`person`/`task`/`limit`/`order_by`.
- SQL compiler layer: `compile_tasks_sql(spec)` emits a `SELECT ... FROM tasks ...` or `task_latest` query with the right `WHERE person = ? AND task = ?` clause and correct parameters.
- API layer: `/db/ask` includes `query`, `ir`, `sql`, `params`, `rows`, and returns 4xx for invalid queries.

---

## 10) KG-lite & Batch Experiments

### 10.1 Batch‑run `/tasks/ask` and inspect KG flags

`scripts/batch_db_ask.py` helps you run a list of questions through `/tasks/ask` and print key debug fields, including KG-lite usage.

Example (PowerShell):

```powershell
python scripts/batch_db_ask.py --file scripts/questions.txt
```

To run the same question set against different SQLite DBs, set `TASKS_DB` before starting the API:

```powershell
# Demo DB (initialized by scripts/init_tasks_sqlite.py)
$env:TASKS_DB = 'data/tasks.db'
uvicorn app.demo_app:app --host 0.0.0.0 --port 8000 --reload

# Enterprise DB (imported from FACT_TASK_ASSIGN.csv)
$env:TASKS_DB = 'data/fact_tasks.db'
uvicorn app.demo_app:app --host 0.0.0.0 --port 8000 --reload
```

Each question prints:

- `Answer`: the Chinese answer (or Text2SQL summary).
- `Resolver mode`: which branch handled the query (`rules`, `hybrid`, `hybrid_llm`, `text2sql`, …).
- `Intent`: high‑level intent label.
- `KG enabled`: whether KG-lite changed the IR for this query.
- `NL IR source`: `llm` or `rules`.
- If present under `nl_ir.extra`:
  - `KG person source`: original person string before normalization.
  - `KG category source`: original text from which category/tags were inferred.
  - `KG project source` (if recorded).

The sample `scripts/questions.txt` includes alias‑heavy and category‑style queries such as:

- “张工的E3D接口联调现在是什么状态？”
- “芯片平台里安全专项相关的任务都有哪些？”
- “交付项目组里张工最近一周还有哪些任务没完成？”

These are ideal for verifying KG-lite behavior:

- Person aliases: `张工` / `老张` → canonical `张三`.
- Project aliases: `"芯片平台"` / `"交付项目组"` normalized to `"芯片"` / `"交付"`.
- Category→tags: `"安全专项"` expands to tags like `"整改"` / `"安全整改"` in SQL filters.

### 10.2 Preview candidates for KG-lite from tasks DB

When moving beyond the demo DB, you should populate `data/kg_data.json` from actual task data rather than hand‑coding entries. The helper script `scripts/extract_kg_from_tasks.py` produces a KG-compatible candidate structure (canonical values only, alias lists left empty on purpose):

```bash
python scripts/extract_kg_from_tasks.py > kg_data.generated.json
```

The output JSON mirrors the KG schema:

- `persons`: `{canonical, aliases: []}` for each distinct `person` value.
- `projects`: `{canonical, aliases: []}` for distinct `project` values.
- `categories`: draft entries built from distinct tags (each tag becomes `{"name": tag, "aliases": [], "tags": [tag]}` to be merged manually later).
- `statuses`: distinct status values from the DB (upper‑cased).
- `priorities`: distinct numeric priority values.

Use this generated file as a staging area; you still need to review & merge aliases/tags manually.

### 10.3 Recommended KG update workflow

1. **When new canonical names appear** (e.g. new project or system):
   - Use the new name in the task system.
   - Periodically run `scripts/extract_kg_from_tasks.py` to produce `kg_data.generated.json`.
   - Review/merge canonical names and desired aliases into `data/kg_data.json` (commit changes along with any relevant docs).
2. **When you start collecting real query logs**:
   - Add a lightweight script that scans queries which did not map to canonical persons/projects.
   - Output “candidate aliases” for manual review.
   - Approved aliases can then be added to `kg_data.json` (or a future backend).

This keeps the KG-lite layer data-driven and ready for eventual migration to a graph backend without changing the NL→IR→SQL code.

---

## 11) Troubleshooting

- **No results from `/search` or `/search_hybrid`**
  - Ingest some docs via `/ingest` first.
  - Check `EMB_URL` and that the embedder container is running if `MOCK_EMB=0`.

- **No results from `/tasks/ask`**
  - Ensure `python scripts/init_tasks_sqlite.py` has been run at least once.
  - Check `TASKS_DB` points to the correct file.
  - Use `/db/ask` to inspect the generated IR and SQL.

- **Text2SQL errors**
  - Confirm `LLM_ENABLED=true` and provider/model env vars are set correctly.
  - Check `error` and `reason` in the JSON; often it is an AST validation issue (unsupported function, wrong table, missing LIMIT).
  - Use `batch_db_ask.py` to inspect `text2sql_raw_response` and adjust prompts.

- **KG-lite seems unused**
  - Look at `KG enabled` and `KG * source` fields in `batch_db_ask.py` output.
  - If they are always absent/false, verify `data/kg_data.json` entries and the exact wording of queries.

If you run into anything unclear, `docs/INSTRUCTIONS_TASKS.md` + the service code under `app/services/` are the best place to understand the full pipeline end‑to‑end.
