# Minimal RAG Demo (FastAPI + FAISS, Mock Embedding)

A minimal Retrieval-Augmented Generation (RAG) demo — using mock embeddings and FAISS vector search.  
You can ingest text, search similar chunks, and reset memory — all within FastAPI.

---

## 1) Overview

**Current stage:** FAISS + Mock Embedding (no real model, in-memory retrieval).  
Implements ingestion → chunking → mock embedding → FAISS search → reset.

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
│  └─ vector_store/
│     ├─ base.py                # VectorStore interface
│     └─ faiss_store.py         # FAISS implementation (default)
├─ data/                        # index.faiss, meta.json (if persistence added)
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

---

## 6) Current Limitations

- No real embedding model (mock vectors only)
- FAISS in-memory index (no persistence yet)
- No keyword or hybrid retrieval
- No frontend UI

---

## 7) Next Steps

| Step | Feature | Goal |
|------|----------|------|
| 1 | Persistence | Save/load FAISS + metadata |
| 2 | `/ingest_bulk` | Batch ingestion |
| 3 | `/search_kw` | Keyword retrieval (BM25) |
| 4 | `/search_hybrid` | Hybrid search |
| 5 | `/answer` | Minimal QA (concatenate top chunks) |
| 6 | `/report` | HTML report rendering |
| 7 | `/` | Static UI |
| 8 | Vector DB | Milvus/Weaviate adapter |

---

© 2025 Minimal RAG Demo Template — Mock Embedding Edition
