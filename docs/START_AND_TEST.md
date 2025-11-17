﻿# Start and Test Guide 鈥?Minimal RAG Demo (FAISS + Milvus optional)

This guide shows how to run the API locally (best for VSCode hot-reload) and how to use a containerized Chinese embedding server (bge-small-zh). Milvus is optional and can be enabled later via compose profile.

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

FAISS is the default store. Start API with hot-reload:

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
  
- Or  **运行新的 Hugging Face TEI 容器 for GPU：**

  ```powershell
  docker run --rm --gpus all -p 8080:80 -e MODEL_ID=BAAI/bge-small-zh-v1.5 ghcr.io/huggingface/text-embeddings-inference:latest
  ```

- Or  **运行新的 Hugging Face TEI 容器 for CPU：**

  ```powershell
  # 停掉当前容器（在跑 TEI 的窗口 Ctrl+C 或 docker stop <id>）
  docker run --rm -p 8080:80 -e MODEL_ID=BAAI/bge-small-zh-v1.5 ghcr.io/huggingface/text-embeddings-inference:cpu-1.5
  ```

  

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

## 6) Task Status — Non‑LLM Ask (Step 2)

Initialize sample tasks once:

```powershell
python scripts/init_tasks_sqlite.py
```

Ask in natural language:

```powershell
curl "http://127.0.0.1:8000/tasks/ask?q=寮犱笁鐨勬彁浜?鏈堝懆鎶ュ畬鎴愪簡鍚楋紵"
curl "http://127.0.0.1:8000/tasks/ask?q=寮犱笁鐨凟3D鎺ュ彛鑱旇皟鐜板湪浠€涔堢姸鎬侊紵"
curl "http://127.0.0.1:8000/tasks/ask?q=鏉庡洓鐨勬暣鐞嗗伐鑹哄寘V2鏄惁宸插畬鎴愶紵"
curl "http://127.0.0.1:8000/tasks/ask?q=鑰佸紶涔濇湀鎶ユ悶瀹氫簡娌★紵"
```

Payload includes: `answer`, `status`, `person`, `task`, `ts`, `sql`, `resolver_mode`, `alpha_vec`, `thresh`, and `candidates` with scores.
Note: in hybrid mode, `alpha_vec` is present but not used (hybrid is vector-only).

---

## 7) Troubleshooting

- Use FAISS only on Windows: set `STORE=faiss` and ignore Milvus profile.
- No results: ingest first via `/ingest` and ensure embedder is up.
- Slow downloads or offline: set `MOCK_EMB=1`.
- Model dim mismatch after switching models: call `/reset` and `/tasks/reload`.

---

## 9) Environment Variables

- `STORE` = `faiss` (default) or `milvus`
- `DATA_DIR` = `data` (FAISS persistence dir)
- `RESOLVER` = `rules` | `embeddings` | `hybrid`
- `MOCK_EMB` = `1` (use mock embeddings) or unset/`0`
- `EMB_URL` = `http://localhost:8080/embeddings` (use containerized embedder)
- `MODEL_NAME` / `EMB_DIM` (e.g., `BAAI/bge-small-zh-v1.5` + `512`)
- Milvus (optional later): `MILVUS_HOST`, `MILVUS_PORT`, `MILVUS_COLLECTION`

Examples (hybrid = vector-only via FAISS Focus Query):

```powershell
$env:STORE='faiss'; $env:RESOLVER='hybrid'; $env:EMB_URL='http://localhost:8080/embeddings'
uvicorn app.demo_app:app --reload
```


---

## 7) Compare: With vs Without Small Model

Goal: compare accuracy/robustness/latency between baseline (rules only) and using the Chinese small model (bge-small-zh), plus an optional third group (embeddings only).

Preparations:
- Keep STORE=faiss (Windows-friendly).
- Initialize tasks: python scripts/init_tasks_sqlite.py.
- After switching modes, call POST /tasks/reload.

A) Baseline — rules only

`powershell
='faiss'
='rules'
Remove-Item Env:EMB_URL -ErrorAction Ignore
Invoke-RestMethod -Uri "http://127.0.0.1:8000/tasks/reload" -Method POST | Out-Null
`

Test (4 queries):

`powershell
curl "http://127.0.0.1:8000/tasks/ask?q=张三的提交9月周报完成了吗？"
curl "http://127.0.0.1:8000/tasks/ask?q=张三的E3D接口联调现在什么状态？"
curl "http://127.0.0.1:8000/tasks/ask?q=李四的整理工艺包V2是否已完成？"
curl "http://127.0.0.1:8000/tasks/ask?q=老张九月报搞定了没？"
`

B) With small model — hybrid (or embeddings)

`powershell
docker compose build embedder
docker compose up -d embedder
='faiss'
='hybrid'       # vector-only via FAISS Focus Query (no rule fusion)
# or use 'embeddings' for matrix-based Focus Query
='0'
='http://localhost:8080/embeddings'
='512'
Invoke-RestMethod -Uri "http://127.0.0.1:8000/tasks/reload" -Method POST | Out-Null
`

Run the same 4 queries. Optional boundary test: add &thresh=0.5.

Metrics to record:
- Correctness/robustness: whether answer matches expected; observe candidates.*[].score concentration.
- Fallback rate: how often returns Top‑k candidates due to low confidence.
- Latency (rough):

`powershell
Measure-Command { curl "http://127.0.0.1:8000/tasks/ask?q=老张九月报搞定了没？" } | Select-Object TotalMilliseconds
`

Expected outcome:
- Hybrid/embeddings should handle colloquial/alias/typo better than rules.
- Rules performs close on exact matches; hybrid slightly slower due to remote embeddings call.



**三种测试方式的环境变量**

- 仅规则（baseline）
  - $env:STORE='faiss'
  - $env:RESOLVER='rules'
  - 建议清空远程嵌入：Remove-Item Env:EMB_URL -ErrorAction Ignore；$env:MOCK_EMB='1' 可用或不设
- 仅向量（小模型 embeddings）
  - docker compose up -d embedder（先起模型容器）
  - $env:STORE='faiss'
  - $env:RESOLVER='embeddings'
  - $env:MOCK_EMB='0'
  - $env:EMB_URL='http://localhost:8080/embeddings'
  - $env:EMB_DIM='512'（bge-small-zh）
- 融合（hybrid：规则 + 向量）
  - docker compose up -d embedder
  - $env:STORE='faiss'
  - $env:RESOLVER='hybrid'
  - $env:MOCK_EMB='0'
  - $env:EMB_URL='http://localhost:8080/embeddings'
  - $env:EMB_DIM='512'

提示

- 切换模式后执行：Invoke-RestMethod -Uri "http://127.0.0.1:8000/tasks/reload" -Method POST
- 未来在 macOS/Linux 想用 Milvus 时，将 STORE 改为 milvus（其余保持不变）。

---

## 8) Troubleshooting

---

Notes (New in Focus Query + Adaptive Thresholds)

- When running embeddings-only (`$env:RESOLVER='embeddings'`), the API applies Focus Query: it first extracts high-confidence rule candidates (>=0.8) and sends them alongside the full sentence to the embedder; the final score per candidate is the max similarity across these queries.
- Thresholds when `thresh` is omitted in `/tasks/ask`:
  - rules: 0.8
  - embeddings: 0.45
  - hybrid: 0.45
  You can still override by explicitly passing `&thresh=...`.
