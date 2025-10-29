# Minimal RAG Demo (FastAPI + FAISS, Mock Embedding)

A minimal Retrieval-Augmented Generation (RAG) demo — using mock embeddings and FAISS vector search.  
You can ingest text, search similar chunks, and reset memory — all within FastAPI.

---

## 1) Overview

**Current stage:** FAISS + Mock Embedding + **Persistence**
Implements ingestion → chunking → mock embedding → FAISS vector indexing → **on-disk persistence** → search → reset.

---

## 2) Project Structure

```
rag-minimal/
├─ app/
│  ├─ __init__.py
│  ├─ demo_app.py               # FastAPI main entry (MVP)
│  ├─ services/
│  │  ├─ embeddings.py          # Embedder class (mock/real switch)
│  │  └─ chunking.py            # Text chunking logic
│  │  ├─ answer.py              # Build minimal answer from hits      
│  │  ├─ keyword.py             # BM25 keyword index & search         
│  │  └─ hybrid.py              # Score normalization & fusion        
│  └─ vector_store/
│     ├─ base.py                # VectorStore interface
│     └─ faiss_store.py         # FAISS implementation (default)
├─ data/                        # Stores persisted index.faiss + meta.json (auto-created)
├─ requirements.txt
├─ run.sh
├─ README.md
└─ .gitignore
```

---

## 3) Setup & Run

### Environment

- Python >= 3.10
- FastAPI, FAISS, sentence-transformers (installed via requirements.txt)

### Steps

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate      # macOS/Linux
# or .\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run server
uvicorn app.demo_app:app --host 0.0.0.0 --port 8000 --reload
```

Visit: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 4) API Endpoints

| Endpoint | Method | Description |
|-----------|---------|-------------|
| `/health` | GET | Health check, embedding mode |
| `/ingest` | POST | Ingest and index text |
| `/search` | GET | Search similar chunks |
| `/reset` | POST | Clear in-memory data |
| `/search_kw` | GET | Keyword/BM25 retrieval (q, k) |
| `/search_hybrid` | GET | Hybrid (vector+keyword) with alpha in [0,1] |
| `/answer` | GET | Minimal answer by concatenating top-k hits (q, k, max_chars, include_scores) |

### Example Usage

```bash
# Ingest text
curl -X POST "http://127.0.0.1:8000/ingest"   -H "Content-Type: application/json"   -d '{"doc_id":"demo1","text":"RAG 利用外部知识增强生成。\n\n本系统支持中文与英文检索。"}'

# Search
curl "http://127.0.0.1:8000/search?q=外部知识&k=3"

# Reset
curl -X POST "http://127.0.0.1:8000/reset"
```

---

## 5) How to Test

1. Start server (`uvicorn ...`)

2. Open Swagger UI → [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

3. Try `/ingest`, then `/search`, and finally `/reset`.

4. Observe Top-K results and scores.

   #### Minimal answer (no LLM)
   ```bash
   GET /answer?q=RAG 的主要思想是什么&k=5&max_chars=300
   ```

5. A ready-to-run demo script is provided:
   - `test_all.sh` (Git Bash/WSL/macOS/Linux)

     

---



## Persistence

Files:
- `data/index.faiss` — FAISS index (vectors + IDs)
- `data/meta.json`   — metadata for chunks ({_id, doc_id, text}, next_id)

Behavior:
- On startup, existing index/meta are auto-loaded.
- After each ingest, both files are atomically rewritten.
- `POST /reset` clears memory and deletes both files.



## 6) Current Limitations

- No real embedding model (mock vectors only)

- FAISS persistence is minimal (local files; no concurrent writers yet).

- No keyword or hybrid retrieval

- No frontend UI

  

## 6.5) Persistence

This version adds **minimal persistence** for FAISS and metadata.

| File               | Description                                         |
| ------------------ | --------------------------------------------------- |
| `data/index.faiss` | Saved FAISS index (vectors + IDs)                   |
| `data/meta.json`   | Metadata for each chunk (doc_id, text, id, next_id) |

### Behavior
- On startup, the system auto-loads existing `index.faiss` and `meta.json`.
- Every ingestion automatically saves both files (atomic write).
- `POST /reset` clears in-memory data and deletes these files.
- After restart, previous ingested data remain searchable.

### Why This Matters
- Enables **persistent retrieval** across restarts.
- Allows reproducible demos, offline regression, and stable behavior.
- Serves as a foundation for switching to Milvus/Weaviate later.



## Keyword & Hybrid Usage

- Keyword search:
  GET /search_kw?q=<keywords>&k=5

- Hybrid search:
  GET /search_hybrid?q=<query>&k=5&alpha=0.6
  (alpha 越大越偏向向量检索；越小越偏向关键词检索)

---

## 7) Next Steps

| Step | Feature | Goal |
|------|----------|------|
| 0 | ✅ Persistence | Save/load FAISS index and metadata |
| 1 | `/ingest_bulk` | Batch ingestion Save/load FAISS + metadata |
| 2 | `/ingest_bulk` | Batch ingestion |
| 3 | `/search_kw` | Keyword retrieval (BM25) |
| 4 | `/search_hybrid` | Hybrid search |
| 5 | `/answer` | Minimal QA (concatenate top chunks) |
| 6 | `/report` | HTML report rendering |
| 7 | `/` | Static UI |
| 8 | Vector DB | Milvus/Weaviate adapter |

---

© 2025 Minimal RAG Demo Template — Mock Embedding Edition
