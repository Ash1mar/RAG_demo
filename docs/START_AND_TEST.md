# Start and Test Guide – Minimal RAG Demo (FAISS + Milvus optional)

This guide shows how to run the API locally (best for VSCode hot‑reload), how to use a containerized Chinese embedding server (bge-small-zh), how to validate the NL→JSON→SQL pipeline for task status queries, and how to optionally plug in a local Ollama + LLM to generate `TaskQuerySpec` via structured outputs.

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

## 5) Search / Hybrid / Keyword / Answer / Reset

```powershell
curl "http://127.0.0.1:8000/search?q=Milvus&k=5"
curl "http://127.0.0.1:8000/search_hybrid?q=Milvus&k=5&alpha=0.5"
curl -X POST "http://127.0.0.1:8000/reset"
```

---

## 6) Task Status – Non‑LLM Ask (Step 2)

Initialize sample tasks once:

```powershell
python scripts/init_tasks_sqlite.py
```

Start API (if not already running):

```powershell
uvicorn app.demo_app:app --host 0.0.0.0 --port 8000 --reload
```

Ask in Chinese (rules + embeddings resolver, **no LLM**):

```powershell
curl "http://127.0.0.1:8000/tasks/ask?q=张三的提取月周报完成了吗？"
curl "http://127.0.0.1:8000/tasks/ask?q=张三的E3D接口联调现在什么状态？"
curl "http://127.0.0.1:8000/tasks/ask?q=李四的整理工艺包V2是否已完成？"
curl "http://127.0.0.1:8000/tasks/ask?q=老张九月报搞定了没？"
```

Payload includes: `answer`, `status`, `person`, `task`, `ts`, `sql`, `resolver_mode`, `alpha_vec`, `thresh`, and `candidates` with scores. In vector‑only hybrid mode, `alpha_vec` is present but not used for fusion.

For NL→JSON→SQL debugging, you can also call the experimental DB ask endpoint:

```powershell
curl "http://127.0.0.1:8000/db/ask?q=张三的E3D接口联调现在什么状态？"
```

This returns a JSON payload with: `query`, `ir` (TaskQuerySpec), `sql`, `params`, and `rows` (raw records from the `tasks` table). It does not generate a natural‑language answer and does not affect `/tasks/ask` behavior.

---

## 6.5) Optional: Enable Ollama‑based NL→JSON (TaskQuerySpec)

Goal: use a local Ollama server (for example with `deepseek-r1:7b`) to parse Chinese task queries into structured `TaskQuerySpec`, then keep the existing SQL compiler and DB layer unchanged.

### A) Prepare Ollama and model

- Install Ollama (see official docs), then make sure the daemon is running (default listens on `http://localhost:11434`).
- Pull the model you want to use (example):

```powershell
ollama pull deepseek-r1:7b
```

### B) Configure env vars for this project

PowerShell example (project root, before starting the API):

```powershell
$env:LLM_ENABLED='true'
$env:LLM_PROVIDER='ollama'
$env:LLM_MODEL='deepseek-r1:7b'          # or your own tag in Ollama
$env:LLM_OLLAMA_BASE_URL='http://localhost:11434'
$env:TASKS_NL2SQL_LLM='1'                # enable LLM path in parse_task_query_nl
```

Then start the API as usual:

```powershell
uvicorn app.demo_app:app --host 0.0.0.0 --port 8000 --reload
```

Now the `/db/ask` endpoint will first try to call `LLMClient.generate_task_query_spec` (Ollama) to produce a `TaskQuerySpec` via structured outputs (`format` JSON Schema). If the LLM call or validation fails, it automatically falls back to the existing rule‑based parser, so the endpoint remains safe to use.

Notes:
- The `/tasks/ask` non‑LLM endpoint continues to work as before; the LLM path is only used inside `parse_task_query_nl` for the NL→JSON→SQL experimental flow.
- You can switch back to rules‑only by either setting `LLM_PROVIDER='dummy'` or unsetting/clearing `TASKS_NL2SQL_LLM`.

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

curl "http://127.0.0.1:8000/tasks/ask?q=张三的提取月周报完成了吗？"
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

curl "http://127.0.0.1:8000/tasks/ask?q=张三的提取月周报完成了吗？"
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
- `RESOLVER` = `rules` | `embeddings` | `hybrid` | `hybrid_plus_rules`
- `MOCK_EMB` = `1` (use mock embeddings) or unset / `0`
- `EMB_URL` = `http://localhost:8080/embeddings` (use containerized embedder)
- `MODEL_NAME` / `EMB_DIM` (e.g., `BAAI/bge-small-zh-v1.5` + `512`)
- `LLM_ENABLED` = `true` / `false` (enable LLM client factory)
- `LLM_PROVIDER` = `dummy` (default) or `ollama`
- `LLM_MODEL` = Ollama model tag (e.g., `deepseek-r1:7b`)
- `LLM_OLLAMA_BASE_URL` = Ollama HTTP endpoint (default `http://localhost:11434`)
- `TASKS_NL2SQL_LLM` = `1` to enable LLM‑first NL→JSON parsing for `/db/ask` (falls back to rules on failure)
*** End Patch```)?;
