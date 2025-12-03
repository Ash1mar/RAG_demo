# Start and Test Guide – Minimal RAG Demo (FAISS + Milvus optional)

This guide shows how to run the API locally (best for VSCode hot‑reload), how to use a containerized Chinese embedding server (bge-small-zh), how to validate the NL→JSON→SQL pipeline for task status queries, and how to optionally plug in a local Ollama + LLM to generate `TaskQuerySpec` via structured outputs and drive the new `hybrid_llm` resolver mode.

---

## 0) Prerequisites

- Windows 10/11 with PowerShell (or macOS/Linux with Bash)
- Python 3.9+ (recommended 3.10/3.11)
- Docker Desktop with WSL 2 backend enabled
- Git (optional)

Tip: For the very first run, avoid model downloads by using mock embeddings. Set `MOCK_EMB=1` before launching the API.

---

## 1) Prepare Project and Python Env

PowerShell (project root):

```powershell
python -m venv venv
./venv/Scripts/Activate
pip install -r requirements.txt

# Optional: use mock embeddings to avoid model download
$env:MOCK_EMB='1'
```

Bash:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
export MOCK_EMB=1
```

---

## 2) Quick Run (FAISS mode, API local)

FAISS is the default store. Start API with hot‑reload:

```powershell
$env:STORE='faiss'
uvicorn app.demo_app:app --host 0.0.0.0 --port 8000 --reload
```

Health check (new terminal):

```powershell
curl "http://127.0.0.1:8000/health"
```

---

## 2.5) Start the Embedding Service in Docker (bge-small-zh)

Keep the API local; run only the embedder as a container.

```powershell
docker compose build embedder
docker compose up -d embedder
$env:EMB_URL='http://localhost:8080/embeddings'
uvicorn app.demo_app:app --host 0.0.0.0 --port 8000 --reload
```

Notes:
- First call downloads the model inside the container.
- Or GPU: build CUDA image and run with `--gpus all`:
  - `docker build -t rag-embedder:latest -f Dockerfile.embedder --build-arg TORCH_CUDA=cu121 .`
  - `docker run --gpus all -p 8080:8080 -e EMBED_DEVICE=cuda rag-embedder:latest`
- Or Hugging Face TEI for GPU:
  ```powershell
  docker run --rm --gpus all -p 8080:80 -e MODEL_ID=BAAI/bge-small-zh-v1.5 ghcr.io/huggingface/text-embeddings-inference:latest
  ```
- Or TEI for CPU:
  ```powershell
  docker run --rm -p 8080:80 -e MODEL_ID=BAAI/bge-small-zh-v1.5 ghcr.io/huggingface/text-embeddings-inference:cpu-1.5
  ```

---

## 3) (Optional) Start Milvus via profile

Milvus services are behind compose profile `milvus`. Only enable when needed.

```powershell
docker compose --profile milvus up -d
```

Stop:

```powershell
docker compose --profile milvus down
```

---

## 4) Ingest Sample Data

PowerShell:

```powershell
$b1 = @{doc_id="report-2024";text="Milvus brings filters and shared indexing to this demo.";source="demo-notes";ts="2024-06-01T10:15:00Z"} | ConvertTo-Json
Invoke-RestMethod -Uri "http://127.0.0.1:8000/ingest" -Method POST -ContentType "application/json" -Body $b1
```

---

## 5) Search / Hybrid / Keyword / Reset

```powershell
curl "http://127.0.0.1:8000/search?q=Milvus&k=5"
curl "http://127.0.0.1:8000/search_hybrid?q=Milvus&k=5&alpha=0.5"
curl -X POST "http://127.0.0.1:8000/reset"
```

---

## 6) Task Status – Non‑LLM Ask (Step 2 baseline)

Initialize sample tasks once (if not yet done):

```powershell
python scripts/init_tasks_sqlite.py
```

Then start the API (FAISS + mock or real embeddings as you prefer) and call:

```powershell
curl "http://127.0.0.1:8000/tasks/ask?q=张三的提交9月周报完成了吗？"
curl "http://127.0.0.1:8000/tasks/ask?q=张三的E3D接口联调现在什么状态？"
curl "http://127.0.0.1:8000/tasks/ask?q=李四的整理工艺包V2是否已完成？"
curl "http://127.0.0.1:8000/tasks/ask?q=老张九月报搞定了没？"
```

These use the existing non‑LLM resolver (`rules` / `embeddings` / `hybrid` / `hybrid_plus_rules`) depending on the `RESOLVER` env var.

Additional intent-focused examples (useful when exercising `hybrid_llm` or `/db/ask`):
- task_status_list: `张三最近的任务状态列表？`
- task_list_by_person: `张三都有哪些任务？`
- task_history: `张三的E3D接口联调历史状态记录`
- task_status_single: `张三的E3D接口联调现在什么状态？`
- task_count_by_status: `How many tasks are still not done for Zhang San?`  <!-- advanced, typically via LLM IR -->

---

## 7) Troubleshooting

- Use FAISS only on Windows: set `STORE=faiss` and ignore Milvus profile.
- No results: ingest first via `/ingest` and ensure embedder is up.
- Slow downloads or offline: set `MOCK_EMB=1`.
- Model dim mismatch after switching models: call `/reset` and `/tasks/reload`.

---

## 8) Compare: With vs Without Small Model

Goal: compare accuracy / robustness / latency between baseline (rules only) and using the Chinese small model (bge-small-zh), plus an optional third group (embeddings only).

Preparations:
- Keep `STORE=faiss` (Windows‑friendly).
- Initialize tasks: `python scripts/init_tasks_sqlite.py`.
- After switching modes, call `POST /tasks/reload`.

### A) Baseline – rules only

```powershell
$env:STORE='faiss'
$env:RESOLVER='rules'
Remove-Item Env:EMB_URL -ErrorAction Ignore
$env:MOCK_EMB='1'
Invoke-RestMethod -Uri "http://127.0.0.1:8000/tasks/reload" -Method POST | Out-Null

curl "http://127.0.0.1:8000/tasks/ask?q=张三的提交9月周报完成了吗？"
curl "http://127.0.0.1:8000/tasks/ask?q=张三的E3D接口联调现在什么状态？"
curl "http://127.0.0.1:8000/tasks/ask?q=李四的整理工艺包V2是否已完成？"
curl "http://127.0.0.1:8000/tasks/ask?q=老张九月报搞定了没？"
```

### B) With small model – hybrid (or embeddings)

```powershell
docker compose build embedder
docker compose up -d embedder
$env:STORE='faiss'
$env:RESOLVER='hybrid'       # vector-only via FAISS Focus Query (no rule fusion)
# or use 'embeddings' for matrix-based Focus Query
$env:MOCK_EMB='0'
$env:EMB_URL='http://localhost:8080/embeddings'
$env:EMB_DIM='512'           # bge-small-zh
Invoke-RestMethod -Uri "http://127.0.0.1:8000/tasks/reload" -Method POST | Out-Null

curl "http://127.0.0.1:8000/tasks/ask?q=张三的提交9月周报完成了吗？"
curl "http://127.0.0.1:8000/tasks/ask?q=张三的E3D接口联调现在什么状态？"
curl "http://127.0.0.1:8000/tasks/ask?q=李四的整理工艺包V2是否已完成？"
curl "http://127.0.0.1:8000/tasks/ask?q=老张九月报搞定了没？"
```

Optional boundary test: add `&thresh=0.5`.

Metrics to record:
- Correctness/robustness: whether answer matches expected; observe `candidates.*[].score` concentration.
- Fallback rate: how often it returns Top‑k candidates due to low confidence.
- Latency (rough):

```powershell
Measure-Command { curl "http://127.0.0.1:8000/tasks/ask?q=老张九月报搞定了没？" } | Select-Object TotalMilliseconds
```

Expected outcome:
- Hybrid/embeddings should handle colloquial / alias / typo better than rules.
- Rules performs close on exact matches; hybrid slightly slower due to remote embeddings call.

---

## 9) NL→JSON→SQL Tests (pytest)

To validate the NL→JSON→SQL pipeline end‑to‑end, a small pytest module is provided:

- File: `tests/test_nl2sql_db_ask.py`
- It covers three layers:
  - **IR layer**: calls `parse_task_query_nl("张三的E3D接口联调现在什么状态？")` and asserts that `intent`, `person`, `task`, `limit`, and `order_by` are parsed as expected.
  - **SQL compiler layer**: constructs a `TaskQuerySpec` by hand and calls `compile_tasks_sql(spec)`, checking that the generated SQL:
    - is a `SELECT ... FROM tasks ...` query
    - contains the expected `WHERE person = ? AND task = ?` clause
    - binds the correct parameters.
  - **API layer**: uses FastAPI’s `TestClient` to call `/db/ask`, asserting that the JSON payload includes `query`, `ir`, `sql`, `params`, and `rows`, and that invalid queries return a 4xx error.

Prerequisites:
- Make sure you have installed test dependencies (pytest, httpx, etc.) via `pip install -r requirements.txt` (or add them as needed).
- Ensure the tasks DB is initialized once:

```bash
python scripts/init_tasks_sqlite.py
```

Run tests (from project root):

```bash
pytest -q tests/test_nl2sql_db_ask.py
```

This validates that NL→JSON parsing, JSON→SQL compilation, and the `/db/ask` endpoint stay consistent. If you later swap out the internal implementation of `parse_task_query_nl` (for example, to call a real LLM), these tests should still pass as long as the IR semantics and SQL behavior remain compatible.

---

### 9.5) Manual NL Query Scenarios (covering new IR features)

Use these natural language prompts to quickly probe recently added IR fields (`filters`, multi-person scopes, summaries, time ranges). For each scenario you can:

```powershell
# Inspect IR/SQL directly
curl "http://127.0.0.1:8000/db/ask?q=<NL_QUERY>"

# Or run through the hybrid_llm resolver (needs RESOLVER=hybrid_llm, TASKS_NL2SQL_LLM=1)
curl "http://127.0.0.1:8000/tasks/ask?q=<NL_QUERY>"
```

| Scenario | NL query (copy/paste) | What to verify |
| --- | --- | --- |
| Single status baseline | `张三的E3D接口联调现在什么状态？` | Intent `task_status_single`, `person`/`task` filled, `limit=1`, SQL targets `task_latest`. |
| Completion time (latest DONE) | `张三的E3D接口联调是什么时候完成的？` | Parser sets `answer_mode=completion_time_latest`, `status=[DONE]`, `limit=1`, intent switches to `task_history`. `/tasks/ask` should respond `… was completed at …` instead of状态描述。 |
| Multi-person list (filters + time_range) | `张三和李四最近一周的任务列表还有哪些？` | `filters` contains `{"field":"person","op":"in","values":["张三","李四"]}`, `time_range.start=now-7d`, intent `task_status_list` or `task_list_by_person`. Hybrid resolver should echo `filter_persons`. |
| Project + tag filter | `把芯片项目里带#安全整改标签的任务都列出来` | `project="芯片"`, `tags=["安全整改"]`, SQL adds `project = ?` AND `tags LIKE ?`. |
| Priority + due_range | `列出李四本周截止的高优P1任务` | `priority=1`, `due_range` reflects current week boundaries, limit tightened, ORDER BY defaults to ts/priority. |
| Status counts by bucket (advanced) | `How many tasks are still not done for Zhang San in the last week?` | IR uses `answer_mode=task_count_by_status` with optional `time_range`; SQL projects `status, COUNT(*) AS task_count` and adds `GROUP BY status`. Typically triggered by LLM or explicit IR, not by keyword rules. |
| Person summary (group by) | `给我张三和李四的任务状态汇总` | Intent `person_summary`, `filters` includes multi-person IN; SQL should output `COUNT(*) AS task_count` with `GROUP BY person, status`. |
| Task history (full timeline) | `张三的E3D接口联调历史状态` | Intent `task_history`, SQL selects from `tasks` (not `task_latest`) with higher default limit (200). |

When verifying via `/db/ask`, focus on the `ir` bloc (`filters`, `time_range`, `order_by`) and generated SQL. When verifying via `/tasks/ask` in `hybrid_llm`, confirm:

- `candidates.persons` contains each resolved person with score 1.0 when filters are pre-aligned.
- `sql` / `params` reflect group-by or IN clauses as expected.
- `answer` surfaces aggregated preview for `person_summary` and multi-person lists (names included).

Feel free to extend the table with more domain-specific prompts (e.g., tags per department, time windows like “过去30天”), keeping the same inspection steps.

---

## 10) Environment Variables

- `STORE` = `faiss` (default) or `milvus`
- `DATA_DIR` = `data` (FAISS persistence dir)
- `RESOLVER` = `rules` | `embeddings` | `hybrid` | `hybrid_plus_rules` | `hybrid_llm`
- `MOCK_EMB` = `1` (use mock embeddings) or unset / `0`
- `EMB_URL` = `http://localhost:8080/embeddings` (use containerized embedder)
- `MODEL_NAME` / `EMB_DIM` (e.g., `BAAI/bge-small-zh-v1.5` + `512`)
- `LLM_ENABLED` = `true` / `false` (enable LLM client factory)
- `LLM_PROVIDER` = `dummy` / `ollama` / `openai`?`openai`=OpenAI-Compatible?? DashScope?
- `LLM_MODEL` = ??????? `qwen2.5-coder:7b`???? Ollama/DashScope tag?
- `LLM_TEXT2SQL_MODEL` = Text2SQL ??????? `qwen3-coder:480b-cloud`?
- `LLM_TEXT2SQL_PROVIDER` / `LLM_TEXT2SQL_OLLAMA_BASE_URL` / `LLM_TEXT2SQL_OPENAI_BASE_URL` / `LLM_TEXT2SQL_API_KEY` = ??? Text2SQL ???? provider/??/????????????????????????
- `LLM_OLLAMA_BASE_URL` = Ollama HTTP endpoint (default `http://localhost:11434`)
- `LLM_OPENAI_BASE_URL` = OpenAI-Compatible base URL (default `https://dashscope.aliyuncs.com/compatible-mode/v1`)
- `LLM_API_KEY` = API key for `openai`/`dashscope` provider
- `TASKS_NL2SQL_LLM` = `1` to enable LLM‑first NL→JSON parsing for `/db/ask` and `hybrid_llm` (falls back to rules on failure)

---

## 11) LLM‑driven NL→JSON + `hybrid_llm` Resolver

This section summarizes how to wire the NL→JSON→SQL pipeline with a local LLM (e.g., via Ollama) and the new `hybrid_llm` mode for `/tasks/ask`.

### 11.1) Enable the LLM client

1. Install and run Ollama locally, and pull a compatible model (for example `qwen2.5-coder:7b` / `deepseek-r1:8b` 等).
2. Configure LLM environment variables:

```powershell
$env:LLM_ENABLED='true'
$env:LLM_PROVIDER='ollama'
$env:LLM_MODEL='qwen2.5-coder:7b'  # or other model tag in Ollama
$env:LLM_TEXT2SQL_MODEL='qwen3-coder:480b-cloud'  # optional: Text2SQL uses bigger model
$env:LLM_OLLAMA_BASE_URL='http://localhost:11434'
```

若使用 OpenAI-Compatible 平台（例如通义千问 DashScope 的 `qwen2.5-coder:7b` 云端接口），可以改用：

```powershell
$env:LLM_ENABLED='true'
$env:LLM_PROVIDER='openai'
$env:LLM_MODEL='qwen2.5-coder:7b'
$env:LLM_TEXT2SQL_MODEL='qwen3-coder:480b-cloud'
$env:LLM_OPENAI_BASE_URL='https://dashscope.aliyuncs.com/compatible-mode/v1'
$env:LLM_API_KEY='your_dashscope_api_key'
$env:LLM_TEXT2SQL_API_KEY='your_dashscope_api_key'   # or另设独立 key
```

3. Optionally, enable LLM‑first NL→JSON parsing in the NL→SQL pipeline:

```powershell
$env:TASKS_NL2SQL_LLM='1'
```

Then start the API as usual:

```powershell
uvicorn app.demo_app:app --host 0.0.0.0 --port 8000 --reload
```

At this point, `parse_task_query_nl` will first try to call `LLMClient.generate_task_query_spec` to produce a `TaskQuerySpec` (JSON) and fall back to the rules‑based parser if the LLM call or validation fails. This behavior affects both `/db/ask` and the `hybrid_llm` resolver described next.

### 11.2) Use `hybrid_llm` for `/tasks/ask`

To route `/tasks/ask` through the “LLM NL→JSON + small model + FAISS + SQL compiler” pipeline:

```powershell
$env:RESOLVER='hybrid_llm'
$env:TASKS_NL2SQL_LLM='1'
Invoke-RestMethod -Uri "http://127.0.0.1:8000/tasks/reload" -Method POST | Out-Null
```

Then query as usual:

```powershell
curl "http://127.0.0.1:8000/tasks/ask?q=张三的E3D接口联调现在什么状态？"
```

Internally, the flow is:

1. LLM (or rules as fallback) produces a `TaskQuerySpec` via `parse_task_query_nl` (NL→JSON).
2. `TaskQueryEngine` uses the existing `EntityResolver` hybrid vector logic (bge+FAISS) to align `person` / `task` from the IR to the actual candidate lists.
3. The aligned `TaskQuerySpec` is compiled into a read‑only SQL query by `compile_tasks_sql`.
4. The SQL is executed via `SQLiteTasksStore.query`, and `/tasks/ask` returns a human‑readable Chinese answer plus `sql` / `params` / `rows` / `candidates` / `nl_ir` for debugging.

If you want a pure LLM NL→JSON→SQL debugging view (no vector alignment, no natural‑language answer), use `/db/ask` as described in section 9. The `hybrid_llm` mode is for “production‑style” `/tasks/ask` with LLM + small model + SQL compiler combined.

### 11.3) Phase 1 – “Let the LLM fill the IR completely” (no extra code)

To “eat the IR full” without touching parser code, teach the LLM prompt to populate *all existing* fields in `TaskQuerySpec`. Below is a lightweight checklist and JSON template you can copy into whatever prompt system you are using (Ollama, OpenAI, etc.).

**LLM output contract**

```json
{
  "intent": "task_status_single | task_status_list | task_list_by_person | task_history | person_summary",
  "answer_mode": "default | completion_time_latest | task_count_by_status | person_summary_by_project | overdue_count_by_person",
  "person": "张三",
  "task": "E3D接口联调",
  "task_keywords": ["接口", "联调"],
  "project": "芯片项目",
  "tags": ["安全整改"],
  "priority": 1,
  "status": ["DONE", "TODO"],
  "time_range": {"start": "now-7d", "end": "now"},
  "due_range": {"start": "start_of_week", "end": "end_of_week"},
  "created_range": null,
  "order_by": [
    {"field": "ts", "direction": "desc"},
    {"field": "priority", "direction": "asc"}
  ],
  "limit": 20,
  "filters": [
    {"field": "person", "op": "in", "values": ["张三", "李四"]},
    {"field": "status", "op": "in", "values": ["TODO"]},
    {"field": "project", "op": "eq", "value": "芯片"}
  ]
}
```

**Prompting tips (no parser code changes required)**

1. **Always fill `intent` + `answer_mode` explicitly.**  
   - Single-task status → `task_status_single` + `answer_mode=default`.  
   - “When was it finished?” → `intent=task_history`, `answer_mode=completion_time_latest`, `status=["DONE"]`, `limit=1`.  
   - “How many tasks still not done?” → set `answer_mode=task_count_by_status`, optionally `status=["TODO","IN_PROGRESS"]`, plus a `time_range` or `due_range` as needed.  
   - “Summaries per project/person” → set `answer_mode=person_summary_by_project`, optionally set `project` / `filters`.  
   - “How many overdue tasks per person” → set `answer_mode=overdue_count_by_person`, supply `due_range` or `time_range`, and keep `status` to non-DONE buckets.

2. **Use the typed filters instead of free-form strings.**  
   - Multi-person → push into `filters` with `{"field":"person","op":"in","values":[...]}`.  
   - Projects/tags/priority/time windows all have dedicated slots—prefer those over inventing new fields.

3. **Order + limit should stay bounded.**  
   - Default order: `ts desc`, `priority asc`.  
   - Default limit: 10 (or 50 for list-by-person); adapt only when user explicitly asks.

4. **Provide example IRs to the LLM for few-shot guidance.**  
   Here are two copy-paste ready examples:

   - *“张三的E3D接口联调是什么时候完成的？”*  
     ```json
     {
       "intent": "task_history",
       "answer_mode": "completion_time_latest",
       "person": "张三",
       "task": "E3D接口联调",
       "status": ["DONE"],
       "order_by": [{"field": "ts", "direction": "desc"}],
       "limit": 1,
       "filters": []
     }
     ```

   - *“How many tasks are still not done for Zhang San in the last week?”*  
     ```json
     {
       "intent": "task_status_list",
       "answer_mode": "task_count_by_status",
       "person": "张三",
       "status": ["TODO", "IN_PROGRESS"],
       "time_range": {"start": "now-7d", "end": "now"},
       "filters": [],
       "limit": 50,
       "order_by": []
     }
     ```
   - *“Give me project-wise status summary for 张三和李四”*  
     ```json
     {
       "intent": "task_status_list",
       "answer_mode": "person_summary_by_project",
       "filters": [
         {"field":"person","op":"in","values":["张三","李四"]}
       ],
       "project": "芯片项目",
       "status": [],
       "limit": 200,
       "order_by": []
     }
     ```
   - *“Who still has overdue tasks this week?”*  
     ```json
     {
       "intent": "task_status_list",
       "answer_mode": "overdue_count_by_person",
       "status": ["TODO","IN_PROGRESS","BLOCKED"],
       "due_range": {"start": "start_of_week", "end": "end_of_week"},
       "filters": [],
       "limit": 100,
       "order_by": []
     }
     ```

5. **Debug quickly via `/db/ask?q=<NL>`** to see the exact IR / SQL the backend consumed. Adjust your LLM prompt until the JSON mirrors what you expect.

With this approach, Phase 1 requires *zero* parser changes—the LLM simply fills the rich IR we already built, and the existing plan/compiler/formatter stack does the rest.

### 11.4) Phase 2 – Aggregation answer modes (LLM-only)

On top of Phase 1, you can now drive two new aggregation styles directly from the LLM output by setting `answer_mode` explicitly (no parser/rule changes needed):

1. **`person_summary_by_project`**  
   - **LLM IR**:  
     ```json
     {
       "intent": "task_status_list",
       "answer_mode": "person_summary_by_project",
       "project": "芯片项目",
       "filters": [{"field":"person","op":"in","values":["张三","李四"]}]
     }
     ```
   - **Plan / SQL**: generates `SELECT project, person, status, COUNT(*) AS task_count ... GROUP BY project, person, status`.  
   - **Answer formatter**: outputs natural language summaries per project, e.g. `Project 芯片项目: 张三(DONE=2, TODO=1); 李四(TODO=3)`.  
   - **Usage tip**: the LLM should push any scope (projects, people, time ranges) into the IR. Parser heuristics are untouched.

2. **`overdue_count_by_person`**  
   - **LLM IR**: set `answer_mode="overdue_count_by_person"`, supply a `due_range` (e.g., `"start_of_week"/"end_of_week"`) and keep `status` to non-DONE buckets (`["TODO","IN_PROGRESS","BLOCKED"]`).  
   - **Plan / SQL**: compiles into `SELECT person, COUNT(*) AS overdue_count ... GROUP BY person` with the provided filters/time windows.  
   - **Answer formatter**: returns `Overdue tasks per person ... Zhang三=3, 李四=1` plus scope hints derived from `time_range`/`due_range`.

Because these modes are only triggered by the LLM output, you can experiment safely without touching `_post_process_intent`; the normal `/tasks/ask` fallback paths remain unchanged.

### 11.5) Phase 3 – Freeze parser heuristics

To keep the rule-based parser stable, we “froze” the heuristics layer at a small, well-documented set (see `docs/INSTRUCTIONS_TASKS.md` for the table). The parser will continue to handle only:

- completion-time intent flips (`answer_mode=completion_time_latest`)
- multi-person filters
- generic time/due range hints
- priority keywords
- limit/order sanity checks

Everything else—including new aggregation modes like `person_summary_by_project`, `overdue_count_by_person`—must come from the LLM IR (`answer_mode`, `filters`, `time_range`, `due_range`, etc.). When you want new semantics, extend the IR/plan/formatter pipeline instead of adding more keyword rules.
---

### 12) Text2SQL 自测与调参（`/tasks/ask` + `scripts/batch_db_ask.py`）

在完成 IR / SQL 快路径后，你可以进一步启用 Text2SQL 模式，用 LLM 直接生成 SQL，再由后端 AST 校验 + 重写后执行。本节给出一个最小自测与调参流程。

#### 12.1) 启用 Text2SQL / hybrid_llm 模式

1. 配置 LLM（以 Ollama 为例）  
   ```bash
   export LLM_ENABLED=true
   export LLM_PROVIDER=ollama
   export LLM_MODEL=qwen2.5-coder:7b    # or other local tag
   export LLM_TEXT2SQL_MODEL=qwen3-coder:480b-cloud   # optional
   export LLM_OLLAMA_BASE_URL=http://localhost:11434
   ```
   若调用 DashScope / 其他 OpenAI-Compatible 云端模型：
   ```bash
   export LLM_ENABLED=true
   export LLM_PROVIDER=openai
   export LLM_MODEL=qwen2.5-coder:7b
   export LLM_TEXT2SQL_MODEL=qwen3-coder:480b-cloud
   export LLM_OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
   export LLM_API_KEY=your_dashscope_api_key
   export LLM_TEXT2SQL_API_KEY=your_dashscope_api_key   # 或另设独立 key
   ```
2. 选择解析模式  
   - 使用 `hybrid_llm` 作为解析器，并允许 NL→JSON→SQL 优先走 LLM：  
     ```bash
     export RESOLVER=hybrid_llm
     export TASKS_NL2SQL_LLM=1
     ```
3. 启动 API：  
   ```bash
   uvicorn app.demo_app:app --host 0.0.0.0 --port 8000 --reload
   ```

#### 12.2) 使用 `scripts/batch_db_ask.py` 批量自测 `/tasks/ask`

1. 准备一组 NL 问句（可参考 `scripts/questions.txt`），覆盖：  
   - 单人 / 多人；  
   - 标签 / 项目（tags / project）；  
   - 时间范围（“最近一周 / 本周截止 / 最近 7 天”）；  
   - 优先级（“高优 P1” 等）。  
2. 用脚本连续请求 `/tasks/ask`：  
   ```bash
   python scripts/batch_db_ask.py --file scripts/questions.txt --endpoint http://localhost:8000/tasks/ask
   ```
3. 每个问题会打印：  
   - `Answer`：自然语言回答；  
   - `Resolver mode` / `Intent`；  
   - 当 `resolver_mode="text2sql"` 时，还会额外打印：  
     - `Text2SQL error`（如 `text2sql_invalid_sql` / `text2sql_db_query_failed` / `text2sql_llm_failed`）；  
     - `Text2SQL reason`：AST / DB 的详细错误原因；  
     - `Text2SQL SQL` / `Text2SQL params`：最终执行的 SQL 与参数；  
     - `Text2SQL query #n`：每条 SQL 及其 `description` / `rows` 预览。

#### 12.3) 如何根据输出调参

- **SQL 正常执行但返回 0 行**  
  - 检查 `Text2SQL SQL` 条件是否与当前 demo 数据匹配（例如时间窗口是否覆盖 `tasks.db` 中的 ts 范围，是否有对应的 TODO / BLOCKED 记录等）。  
  - 如果 SQL 结构正确而数据条件过于苛刻，这是数据问题而非解析问题，可以通过调整 NL 问句或扩充 demo 数据来验证更多场景。

- **`text2sql_invalid_sql` / AST 解析失败**  
  - 核对 `Text2SQL reason` 和原始 `Text2SQL SQL`：  
    - 常见情况包括：括号不匹配、引用非白名单表、出现 `DATE_SUB` / `CURDATE`、带 `?` 占位符等。  
  - 根据原因调整 Text2SQL prompt：  
    - 明确禁止某些函数 / 方言；  
    - 强调“不要输出占位符，要输出具体时间表达式”；  
    - 引导 LLM 优先使用 IR hint 中的 person / project / tags / time_range / priority。

- **`text2sql_llm_failed` / `LLM output is not valid JSON`**  
  - 查看 `text2sql_raw_response`：通常是 LLM 在 JSON 外多写了说明文字，或 JSON 内部引号/逗号缺失。  
  - 对应处理：收紧系统 prompt，强调“只返回严格合法的 JSON，不要输出额外说明”，必要时更换或微调模型。

通过上述流程，你可以在不修改后端代码的前提下，对 Text2SQL 行为进行迭代式的 prompt / 模型调优，同时利用 AST 层确保 SQL 始终安全、结构可控。
