# Start and Test Guide — Minimal RAG Demo (FAISS ↔ Milvus)

This guide walks you from a clean machine to a fully working demo, and verifies every endpoint: /health, /ingest, /search, /search_kw, /search_hybrid, /answer, /reset. Commands are PowerShell-friendly, with Bash equivalents noted.

---

## 0) Prerequisites

- Windows 10/11 with PowerShell (or macOS/Linux with Bash)
- Python 3.9+ (recommended 3.10/3.11)
- Docker Desktop with WSL 2 backend enabled
  - Allocate at least 4–6 GB RAM to Docker (Settings → Resources)
- Git (optional)

Tip: For the very first run, avoid model downloads by using mock embeddings. Set `MOCK_EMB=1` before launching the API.

---

## 1) Prepare Project and Python Env

In a PowerShell terminal at the project root (the folder containing `README_minimal_rag_demo.md`):

```powershell
python -m venv venv
./venv/Scripts/Activate
pip install -r requirements.txt

# Optional: use mock embeddings to avoid model download
$env:MOCK_EMB='1'
```

Bash equivalent:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
export MOCK_EMB=1
```

---

## 2) Quick Run (FAISS mode)

FAISS is the default store — fastest way to try the API.

```powershell
$env:STORE='faiss'
uvicorn app.demo_app:app --host 0.0.0.0 --port 8000 --reload
```

Health check (new terminal):

```powershell
curl "http://127.0.0.1:8000/health"
# Expect: { ..., "vector_store": "faiss", ... }
```

Stop with Ctrl+C in the server terminal when done.

---

## 3) Start Milvus (Docker Compose)

Run from the project root containing `docker-compose.yml`:

```powershell
docker compose up -d
docker compose ps
docker compose logs -f milvus-standalone   # optional: follow until ready
```

What starts:
- etcd: `quay.io/coreos/etcd:v3.5.12`
- MinIO: `minio/minio:RELEASE.2024-05-28T17-19-04Z`
- Milvus: `milvusdb/milvus:v2.4.3`

Troubleshooting:
- If containers are unhealthy, increase Docker memory to 4–6 GB+, then `docker compose down && docker compose up -d`.
- Ports used: 19530 (Milvus gRPC), 9091 (Milvus metrics), 9000/9001 (MinIO). Free them if conflicts arise.

Stop Milvus when finished testing:

```powershell
docker compose down
```

---

## 4) Run API with Milvus

In a new PowerShell terminal:

```powershell
$env:STORE='milvus'
$env:MILVUS_HOST='127.0.0.1'
$env:MILVUS_PORT='19530'
uvicorn app.demo_app:app --host 0.0.0.0 --port 8000 --reload
```

Verify:

```powershell
curl "http://127.0.0.1:8000/health"
# Expect: { ..., "vector_store": "milvus", ... }
```

Notes:
- Environment variables must be set in the same terminal session where you launch uvicorn.
- Change takes effect only on restart.

---

## 5) Ingest Sample Data

PowerShell (preferred on Windows):

```powershell
$b1 = @{doc_id="report-2024";text="Milvus brings filters and shared indexing to this demo.";source="demo-notes";ts="2024-06-01T10:15:00Z"} | ConvertTo-Json
Invoke-RestMethod -Uri "http://127.0.0.1:8000/ingest" -Method POST -ContentType "application/json" -Body $b1

$b2 = @{doc_id="notes-2023";text="FAISS local index; later we migrate to Milvus.";source="lab-notes";ts="2023-12-20T08:00:00Z"} | ConvertTo-Json
Invoke-RestMethod -Uri "http://127.0.0.1:8000/ingest" -Method POST -ContentType "application/json" -Body $b2
```

curl variant (be careful with quoting):

```powershell
curl -X POST "http://127.0.0.1:8000/ingest" -H "Content-Type: application/json" -d "{\"doc_id\":\"report-2024\",\"text\":\"Milvus brings filters and shared indexing to this demo.\",\"source\":\"demo-notes\",\"ts\":\"2024-06-01T10:15:00Z\"}"
```

Expected response contains `doc_id` and the number of chunks indexed.

---

## 6) Search, Hybrid, Keyword, Answer, Reset

Vector search (no filter):

```powershell
curl "http://127.0.0.1:8000/search?q=Milvus&k=5"
```

Vector search (filter by source + time window):

```powershell
curl "http://127.0.0.1:8000/search?q=filters&k=5&source=demo-notes&date_from=2024-05-01T00:00:00Z"
# Or with epoch ms: &date_to=1717200000000
```

Hybrid search (vector+keyword with alpha, filtered to a doc):

```powershell
curl "http://127.0.0.1:8000/search_hybrid?q=Milvus&k=5&alpha=0.5&doc_id=report-2024"
```

Keyword search (with date upper bound):

```powershell
curl "http://127.0.0.1:8000/search_kw?q=index&date_to=1717200000000"
```

Build a simple answer from top-k vector hits:

```powershell
curl "http://127.0.0.1:8000/answer?q=What%20changed%20in%20the%20demo%3F&k=6"
```

Reset indices (clears Milvus collection or FAISS files + keyword index):

```powershell
curl -X POST "http://127.0.0.1:8000/reset"
```

Expected fields per hit: `doc_id`, `text`, optional `source`, `ts`, and `score`.

---

## 7) (Optional) Migrate FAISS data → Milvus

Precondition: you have `data/index.faiss` and `data/meta.json` from prior FAISS ingests.

```powershell
python scripts/migrate_faiss_to_milvus.py `
  --data-dir data `
  --milvus-host 127.0.0.1 `
  --milvus-port 19530 `
  --collection rag_chunks `
  --drop-existing
```

Then ensure the API is running in Milvus mode (Section 4) and re-run searches to confirm identical behavior with server-side filtering.

---

## 8) Quick Smoke Checklist (10 minutes)

1. Start Milvus: `docker compose up -d`
2. Start API: set `STORE=milvus`, then `uvicorn app.demo_app:app --reload`
3. Health: `/health` shows `vector_store: "milvus"`
4. Ingest two docs (Section 5)
5. Run vector search and see hits (Section 6)
6. Add source + time filters and verify fewer results
7. Run hybrid search and keyword search
8. Call `/answer` and verify concise output
9. Call `/reset` and verify subsequent search returns empty until re-ingest

---

## 9) Troubleshooting

- Milvus unhealthy or slow to start
  - Increase Docker memory to 4–6 GB+, retry `docker compose up -d`
  - Check logs: `docker compose logs -f milvus-standalone`
- `/health` still shows `faiss`
  - Ensure `STORE=milvus` is set in the same terminal session before launching uvicorn
- No search results
  - Ingest first via `/ingest`; confirm Milvus is running and reachable at `MILVUS_HOST:PORT`
- 422 or 400 on `/ingest`
  - Prefer PowerShell `Invoke-RestMethod` with `ConvertTo-Json` to avoid quoting issues
- Slow or model download failures
  - Use mock embeddings: set `MOCK_EMB=1` for functional verification
- Port conflicts (19530/9091/9000/9001)
  - Stop conflicting services or adjust `docker-compose.yml`

---

## 10) Appendix — Environment Variables

Set in the same terminal before launching uvicorn; changes require restart.

- `STORE` = `faiss` (default) or `milvus`
- `MILVUS_HOST` = `localhost` (default) or server IP/hostname
- `MILVUS_PORT` = `19530` (default)
- `MILVUS_COLLECTION` = `rag_chunks` (default)
- `DATA_DIR` = `data` (FAISS persistence dir)
- `MOCK_EMB` = `1` (use mock embeddings) or unset/`0`
- `MODEL_NAME` / `EMB_DIM` (only if you switch to a real embedding model)

Examples:

```powershell
$env:STORE='milvus'; $env:MILVUS_HOST='127.0.0.1'; $env:MILVUS_PORT='19530'
uvicorn app.demo_app:app --reload
```

```bash
export STORE=milvus MILVUS_HOST=127.0.0.1 MILVUS_PORT=19530
uvicorn app.demo_app:app --reload
```

