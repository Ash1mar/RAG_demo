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

## 10) Environment Variables

- `STORE` = `faiss` (default) or `milvus`
- `DATA_DIR` = `data` (FAISS persistence dir)
- `RESOLVER` = `rules` | `embeddings` | `hybrid` | `hybrid_plus_rules` | `hybrid_llm`
- `MOCK_EMB` = `1` (use mock embeddings) or unset / `0`
- `EMB_URL` = `http://localhost:8080/embeddings` (use containerized embedder)
- `MODEL_NAME` / `EMB_DIM` (e.g., `BAAI/bge-small-zh-v1.5` + `512`)
- `LLM_ENABLED` = `true` / `false` (enable LLM client factory)
- `LLM_PROVIDER` = `dummy` (default) or `ollama`
- `LLM_MODEL` = Ollama model tag (e.g., `deepseek-r1:7b`)
- `LLM_OLLAMA_BASE_URL` = Ollama HTTP endpoint (default `http://localhost:11434`)
- `TASKS_NL2SQL_LLM` = `1` to enable LLM‑first NL→JSON parsing for `/db/ask` and `hybrid_llm` (falls back to rules on failure)

---

## 11) LLM‑driven NL→JSON + `hybrid_llm` Resolver

This section summarizes how to wire the NL→JSON→SQL pipeline with a local LLM (e.g., via Ollama) and the new `hybrid_llm` mode for `/tasks/ask`.

### 11.1) Enable the LLM client

1. Install and run Ollama locally, and pull a compatible model (for example `deepseek-r1:7b`).
2. Configure LLM environment variables:

```powershell
$env:LLM_ENABLED='true'
$env:LLM_PROVIDER='ollama'
$env:LLM_MODEL='deepseek-r1:7b'          # or other model
$env:LLM_OLLAMA_BASE_URL='http://localhost:11434'
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

