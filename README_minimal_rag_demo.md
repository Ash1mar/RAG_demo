# Minimal RAG Demo – FAISS ↔ Milvus

This project started as a mock-embedding demo with a local FAISS index. It now ships with a **pluggable vector store interface** so you can stay on FAISS or switch to a managed Milvus backend for metadata filtering, shared access, and higher concurrency.

For a full start-and-test walkthrough (Windows PowerShell + Bash), see `docs/START_AND_TEST.md`.

---

## 1. Project Structure

```
RAG_demo/
├─ app/
│  ├─ demo_app.py                # FastAPI app with /health, /ingest, /search, /search_kw, /search_hybrid, /answer
│  ├─ services/
│  │  ├─ embeddings.py           # Embedder wrapper (mock or real SBERT)
│  │  ├─ chunking.py             # Simple text splitter
│  │  ├─ answer.py               # Answer builder (concatenate top chunks)
│  │  ├─ keyword.py              # Keyword/BM25 index with metadata passthrough
│  │  └─ hybrid.py               # Score fusion helpers
│  └─ vector_store/
│     ├─ base.py                 # VectorStore ABC (add_texts/search/reset + filters)
│     ├─ faiss_store.py          # Local FAISS implementation (persists to data/)
│     └─ milvus_store.py         # Milvus adapter (standalone deployment)
├─ data/                         # Default FAISS persistence (index.faiss + meta.json)
├─ docker-compose.yml            # Milvus + etcd + MinIO in single-node mode
├─ requirements.txt              # Python dependencies (FastAPI, FAISS, pymilvus...)
├─ run.sh / test_all.sh          # Helper scripts (unchanged)
└─ scripts/
   └─ migrate_faiss_to_milvus.py # Migration utility (FAISS files → Milvus collection)
```

---

## 2. Environment & Dependencies

### Python

```bash
python -m venv venv
source venv/bin/activate        # or .\venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Windows note:
- On Windows, `requirements.txt` skips installing Milvus client packages by default. FAISS mode works out of the box.
- If you later want Milvus on Windows, install Milvus client packages manually (e.g., `pymilvus==2.4.3` with compatible dependencies) and set `STORE=milvus`.

Key packages:

- `fastapi`, `uvicorn` – API server
- `sentence-transformers`, `faiss-cpu` – embeddings + FAISS store
- `pymilvus==2.4.3` – Milvus Python SDK

### Milvus (optional)

Spin up a local Milvus standalone using the provided Compose file:

```bash
docker compose up -d               # Uses docker-compose.yml
# wait for etcd/minio/milvus to become healthy
docker compose logs -f milvus      # optional: follow Milvus logs
```

This runs:

- `quay.io/coreos/etcd:v3.5.12`
- `minio/minio:RELEASE.2024-05-28T17-19-04Z`
- `milvusdb/milvus:v2.4.3`

Persistent volumes (`milvus-data`, `milvus-minio`, `milvus-etcd`) keep your vectors and metadata across restarts.

---

## 3. Running the API

### Configure the Vector Store

| Env var | Default | Description |
|---------|---------|-------------|
| `STORE` | `faiss` | Choose `faiss` or `milvus`. |
| `DATA_DIR` | `data` | FAISS persistence directory (index.faiss + meta.json). |
| `MILVUS_HOST` | `localhost` | Milvus hostname/IP. |
| `MILVUS_PORT` | `19530` | Milvus gRPC port. |
| `MILVUS_COLLECTION` | `rag_chunks` | Collection name used by the adapter. |

Example (Milvus):

```bash
export STORE=milvus
export MILVUS_HOST=127.0.0.1
export MILVUS_PORT=19530
uvicorn app.demo_app:app --host 0.0.0.0 --port 8000 --reload
```

Example (FAISS):

```bash
export STORE=faiss
uvicorn app.demo_app:app --host 0.0.0.0 --port 8000 --reload
```

Swagger UI: http://127.0.0.1:8000/docs

---

## 3.1. SQLite Tasks Store (Step 1)

Windows-friendly, read-only task data source using SQLite. This powers a minimal connectivity API to answer questions like “某人某任务是否完成/现在什么状态”.

Setup sample data:

```bash
python scripts/init_tasks_sqlite.py   # creates data/tasks.db with sample rows
```

Config (env vars):

- `TASKS_BACKEND` = `sqlite` (default; keep for now, future backends can plug in)
- `TASKS_DB` = path to SQLite file (default `data/tasks.db`)

Sample dataset inserted by the init script:

- 张三｜提交9月周报｜TODO → DONE（以最新为准）
- 张三｜E3D接口联调｜TODO
- 李四｜整理工艺包V2｜DONE

Quick checks:

```bash
curl "http://127.0.0.1:8000/tasks/status?person=张三&task=提交9月周报"    # DONE
curl "http://127.0.0.1:8000/tasks/status?person=张三&task=E3D接口联调"  # TODO
curl "http://127.0.0.1:8000/tasks/status?person=李四&task=整理工艺包V2"  # DONE
```

This API is intentionally minimal and will be swapped with real logic later; we keep abstraction slots to integrate with FAISS / Milvus / KG without changing upper-level call style.

---

## 3.2. 无模型问数（Step 2）

目标：不依赖大语言模型，仅用规则 + FAISS 相似度做实体解析（人名/任务名），从本地 SQLite 读取任务最新状态，并生成中文回答。保留后续切换 Milvus / KG 的接口位。

新端点：

- `GET /tasks/ask?q=...&topk=3&thresh=0.58`
  - 返回字段：`answer`、`status`、`person`、`task`、`ts`、`sql`、`candidates`（含打分，调试用）
  - 识别“状态查询”意图的关键词：完成/未完成/状态/进度/是否完成/搞定/结束
  - 实体解析：从 DB 导出候选（distinct person/task），用 Embedder + FAISS 近邻并与关键词重叠分数融合

示例（确保先初始化样例 DB）：

```bash
python scripts/init_tasks_sqlite.py
uvicorn app.demo_app:app --host 0.0.0.0 --port 8000 --reload

curl "http://127.0.0.1:8000/tasks/ask?q=张三的提交9月周报完成了吗？"
curl "http://127.0.0.1:8000/tasks/ask?q=张三的E3D接口联调现在什么状态？"
curl "http://127.0.0.1:8000/tasks/ask?q=李四的整理工艺包V2是否已完成？"
curl "http://127.0.0.1:8000/tasks/ask?q=老张九月报搞定了没？"
```

如果识别置信度低，会返回 Top-k 候选和分数，便于人工确认或迭代词表/别名（内置示例别名：老张→张三）。

可选：当样本或库有更新时，可重建实体索引：

```bash
curl -X POST "http://127.0.0.1:8000/tasks/reload"
```

---

## 4. API Summary

| Endpoint | Method | Notes |
|----------|--------|-------|
| `/health` | GET | Returns embedder info and current vector store backend. |
| `/ingest` | POST | Body: `{doc_id, text, source?, ts?}` – optional `source` tag and timestamp (ISO-8601 or epoch ms). |
| `/search` | GET | Params: `q`, `k`, plus optional `doc_id`, `source`, `date_from`, `date_to` filters. |
| `/search_kw` | GET | Keyword/BM25 search with same optional filters (applied after scoring). |
| `/search_hybrid` | GET | Combines vector + keyword (alpha in [0,1]) and respects the same filters. |
| `/answer` | GET | Builds a simple answer from top-k vector hits (no filters yet). |
| `/reset` | POST | Clears vector + keyword indices (drops Milvus collection when using the adapter). |
| `/tasks/status` | GET | Minimal connectivity: latest status for `person` + `task`. |
| `/tasks/ask` | GET | 无模型问数：解析 + 查询 + 中文应答（含 SQL 与候选打分）。 |

### Example Requests

```bash
# Ingest document with metadata
curl -X POST "http://127.0.0.1:8000/ingest" \
  -H "Content-Type: application/json" \
  -d '{
        "doc_id": "report-2024",
        "text": "Milvus brings filters and shared indexing to this demo.",
        "source": "demo-notes",
        "ts": "2024-06-01T10:15:00Z"
      }'

# Vector search filtered by source + time range
curl "http://127.0.0.1:8000/search?q=filters&k=5&source=demo-notes&date_from=2024-05-01T00:00:00Z"

# Hybrid search filtered to a doc_id
curl "http://127.0.0.1:8000/search_hybrid?q=Milvus&doc_id=report-2024&alpha=0.5"

# Keyword-only search filtered by timestamp ceiling (epoch ms)
curl "http://127.0.0.1:8000/search_kw?q=共享&date_to=1717200000000"
```

Filters cascade through the FAISS and Milvus adapters; results missing `ts` metadata are automatically excluded from date-bounded queries.

---

## 5. Migrating Existing FAISS Data to Milvus

If you have existing `data/index.faiss` + `data/meta.json`, run the helper script after Milvus is up:

```bash
python scripts/migrate_faiss_to_milvus.py \
  --data-dir data \
  --milvus-host 127.0.0.1 \
  --milvus-port 19530 \
  --collection rag_chunks \
  --drop-existing         # optional: recreate the collection
```

The script reconstructs vectors from the FAISS ID map, preserves metadata (`doc_id`, `text`, optional `source`, `ts`), and inserts grouped chunks per document into Milvus using the same cosine similarity configuration as the runtime adapter.

---

## 6. Benchmark & Operational Notes

### Minimal Benchmark Plan

1. **Cold ingest check** – Ingest 1k short chunks (e.g., duplicate the provided demo text) via `/ingest` while timing the request latency. Compare FAISS vs Milvus.
2. **Search latency** – With Milvus running, issue 100 concurrent `/search` requests using [`hey`](https://github.com/rakyll/hey) or `ab` for both filtered and unfiltered queries; record p95 latency.
3. **Hybrid scoring sanity** – Run `/search_hybrid` for at least 5 queries with filters to confirm deterministic ordering between FAISS and Milvus backends.

Log basic metrics (average ingest time, search p95, success rate). This is intentionally lightweight but enough to validate the migration.

### Operations Checklist

- **Persistence** – FAISS persists to `DATA_DIR`. Milvus persists inside Docker volumes (`milvus-data`, `milvus-minio`, `milvus-etcd`). Back up these volumes or configure MinIO to sync to external object storage for production.
- **Backups** – For Milvus, schedule periodic `backup` jobs (see Milvus docs) or snapshot MinIO buckets. For FAISS, continue copying `index.faiss` and `meta.json` after each ingest batch.
- **Monitoring** – Expose Milvus metrics via the built-in Prometheus endpoint (`:9091/metrics`). Watch etcd/minio container health, memory usage, and query latency. Consider wiring FastAPI logs to a collector (e.g., Grafana Loki) when running under load.
- **Scaling** – The current compose stack is single-node. For higher concurrency, migrate to Milvus distributed mode or Zilliz Cloud, and update `MILVUS_HOST/PORT` accordingly.

---

## 7. Next Steps

- Wire additional metadata fields (e.g., author, tags) into `/ingest` and store-level filters.
- Add automated benchmarks (Locust/Vegeta) to CI for regression detection.
- Provide `/ingest_bulk` and async ingestion pipelines.
- Extend `/answer` to accept the same filter set and forward to the vector store.

Happy hacking!
