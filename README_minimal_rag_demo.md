# Minimal RAG Demo (FAISS first, Milvus optional)

Minimal retrieval and “non‑LLM task status ask” demo with:
- Pluggable vector stores: FAISS (default) and Milvus (optional)
- Local API (FastAPI) for development; optional containerized deployment
- Non‑LLM entity resolution (rules / embeddings / hybrid) for task status Q&A
- Optional containerized embedding server (bge-small-zh) consumed by local API

For a complete hands‑on from init → start → test, see `docs/START_AND_TEST.md`.

---

## Project Structure

```
RAG_demo/
├─ app/
│  ├─ demo_app.py                 # FastAPI app (API endpoints)
│  ├─ services/
│  │  ├─ embeddings.py            # Embedder wrapper (mock/remote/local)
│  │  ├─ chunking.py              # Simple text splitter
│  │  ├─ answer.py                # Build answer from top-k hits
│  │  ├─ keyword.py               # Keyword/BM25 index
│  │  ├─ hybrid.py                # Score fusion helper
│  │  └─ task_query.py            # Non‑LLM task ask engine (Step 2)
│  ├─ vector_store/
│  │  ├─ base.py                  # VectorStore ABC
│  │  ├─ faiss_store.py           # FAISS implementation
│  │  └─ milvus_store.py          # Milvus adapter (optional)
│  └─ tasks_store/
│     ├─ base.py                  # TasksStore ABC
│     └─ sqlite_store.py          # SQLite implementation (read‑only)
├─ embedder_server/
│  └─ server.py                   # Containerized embedding service (bge-small-zh)
├─ data/                          # FAISS persistence / SQLite file
├─ scripts/
│  └─ migrate_faiss_to_milvus.py  # Optional migration helper
├─ docs/
│  ├─ START_AND_TEST.md           # End‑to‑end init/start/test guide
│  └─ INSTRUCTIONS_TASKS.md       # Step 2 task Q&A details
├─ docker-compose.yml             # embedder (+ milvus via profile)
├─ Dockerfile.embedder            # Embedding server image (CPU/GPU)
├─ requirements.txt               # Python deps
└─ .env.example                   # Env templates
```

---

## Key Features

- Vector search (FAISS default), keyword search, hybrid fusion
- Non‑LLM task ask (`/tasks/ask`) with:
  - Intent detection by keywords (完成/未完成/状态/进度/是否完成/搞定/结束)
  - Entity resolution: rules / embeddings / hybrid (configurable)
  - SQLite read‑only task store; returns answer + SQL + candidates with scores
- Embeddings options:
  - Mock (no downloads) for quick start
  - Local SBERT (e.g., MiniLM or bge-small-zh)
  - Remote embedding server via `EMB_URL` (containerized bge-small-zh)
- Milvus optional, gated by compose profile `milvus`

---

## Configuration (Env Vars)

- Vector store: `STORE=faiss|milvus` (default `faiss`)
- Resolver: `RESOLVER=rules|embeddings|hybrid` (default `hybrid`)
- Embeddings:
  - `MOCK_EMB=1|0` (mock on/off)
  - `MODEL_NAME` (e.g., `BAAI/bge-small-zh-v1.5`, `sentence-transformers/all-MiniLM-L6-v2`)
  - `EMB_DIM` (bge-small-zh=512; MiniLM=384)
  - `EMB_URL` (use external embedder, e.g., `http://localhost:8080/embeddings`)
  - `EMB_TIMEOUT` (HTTP timeout seconds, default 12)
- Milvus (optional later): `MILVUS_HOST`, `MILVUS_PORT`, `MILVUS_COLLECTION`
- Tasks store: `TASKS_DB` (default `data/tasks.db`)

See `.env.example` for a reference layout.

---

## Development Modes

- Local API + containerized embedder (recommended on Windows):
  1) `docker compose build embedder && docker compose up -d embedder`
  2) Set `EMB_URL=http://localhost:8080/embeddings` and run `uvicorn app.demo_app:app --reload`
- FAISS‑only on Windows: keep `STORE=faiss` and ignore Milvus profile
- Milvus later: `docker compose --profile milvus up -d` and set `STORE=milvus`

---

## Where to Start

- End‑to‑end steps: `docs/START_AND_TEST.md`
- Task Q&A details (Step 2): `docs/INSTRUCTIONS_TASKS.md`

---

## What’s New

- Embeddings‑only Focus Query: when `RESOLVER=embeddings`, the resolver first extracts rule‑high candidates (>=0.8) and uses them as focused queries alongside the full sentence. Scores take the max over these queries, improving alignment for short entity names.
- Mode‑adaptive thresholds (used when `thresh` is omitted in `/tasks/ask`):
  - rules: 0.8
  - embeddings: 0.45
  - hybrid: 0.58
  You can still provide `thresh` explicitly to override.
