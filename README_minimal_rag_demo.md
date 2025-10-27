# Minimal RAG Demo (FastAPI + FAISS, Mock Embedding)

A minimal retrieval backend — no model required.
 You can ingest text, perform vector-based “semantic” search, and reset memory through FastAPI.

------

## 1) Overview

**What works now**

- Ingest → Chunk → Mock-Embed → FAISS Index
- Search (Top-K similar chunks)
- Reset index
- Runs 100 % offline (no model download)

------

## 2) Main Components

| Layer        | File                              | Purpose                                                    |
| ------------ | --------------------------------- | ---------------------------------------------------------- |
| API          | `app/demo_app.py`                 | Defines `/health`, `/ingest`, `/search`, `/reset`          |
| Embedding    | `app/services/embeddings.py`      | `Embedder` class; default uses Mock mode (`use_mock=True`) |
| Chunking     | `app/services/chunking.py`        | Splits text into paragraph-sized chunks                    |
| Vector Store | `app/vector_store/faiss_store.py` | In-memory FAISS index (add/search/reset)                   |
| Interface    | `app/vector_store/base.py`        | Defines `VectorStore` API for future replacements          |

------

## 3) Run the Demo

```
python -m venv venv
source venv/bin/activate  # macOS/Linux
# or .\venv\Scripts\activate # Windows
pip install -r requirements.txt
uvicorn app.demo_app:app --reload
```

Open → http://127.0.0.1:8000/docs

------

## 4) Endpoints

| Path      | Method | Description                              |
| --------- | ------ | ---------------------------------------- |
| `/health` | GET    | Shows status and embedding mode (`mock`) |
| `/ingest` | POST   | Add text to index                        |
| `/search` | GET    | Retrieve top-K similar chunks            |
| `/reset`  | POST   | Clear index                              |

**Example**

```
curl -X POST "http://127.0.0.1:8000/ingest" \
 -H "Content-Type: application/json" \
 -d '{"doc_id":"demo1","text":"RAG 利用外部知识增强生成。\n\n本系统支持中文与英文检索。"}'
curl "http://127.0.0.1:8000/search?q=外部知识&k=3"
```

------

## 5) Testing / Demo Flow

1. Run server
2. Visit `/docs` (Swagger UI)
3. Try `/ingest` → `/search` → `/reset`
4. Observe scores and texts returned

------

## 6) Current Limitations

| Item                 | Status | Note                             |
| -------------------- | ------ | -------------------------------- |
| Real embedding model | 🚫      | Mock hash-based vector only      |
| Persistence          | 🚫      | FAISS in-memory, lost on restart |
| Keyword /BM25        | 🔜      | Next step                        |
| Hybrid search        | 🔜      | Vector + Keyword                 |
| Answer generation    | 🔜      | Concatenate top chunks           |
| UI page              | 🔜      | Simple HTML front end            |

------

## 7) Next Steps (Roadmap)

| Step | Feature          | Goal                               |
| ---- | ---------------- | ---------------------------------- |
| 1    | `/ingest_bulk`   | Batch ingest                       |
| 2    | `/search_kw`     | Keyword retrieval                  |
| 3    | `/search_hybrid` | Hybrid search                      |
| 4    | `/answer`        | Minimal QA (no LLM)                |
| 5    | `/report`        | Render HTML report                 |
| 6    | `/`              | Static UI                          |
| 7    | Swap VectorStore | Milvus/Weaviate                    |
| 8    | Add real model   | Sentence-BERT or OpenAI Embeddings |

------

**Why Mock Embedding?**

- Deterministic and offline; proves the retrieval logic without GPU or weights.

**Why FAISS?**

- Zero infra dependency; fast CPU index sufficient for early prototype.