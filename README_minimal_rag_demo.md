# Minimal RAG Demo (CNPE VSCode Workshop Edition)

This project is a minimal, but realistic, RAG + task‑status demo for the CNPE VSCode workshop:

- FastAPI backend with pluggable vector store (FAISS by default, Milvus optional).
- Chinese embeddings (mock / local `SentenceTransformer` / remote bge‑small‑zh container).
- SQLite‑backed task status Q&A API (`/tasks/ask`) with multiple resolver modes (rules, embeddings, hybrid, hybrid_plus_rules, hybrid_llm, text2sql).
- An experimental NL→JSON→SQL pipeline (`/db/ask`) targeting a single `tasks` table, with a structured IR (`TaskQuerySpec`) and safe SQL compiler.
- A lightweight KG‑lite layer for domain semantics (persons/projects/categories/tags) that powers both NL→IR and Text2SQL hints.

For a full “from zero to working” guide, see `docs/START_AND_TEST.md`.

---

## Project Structure

Top‑level layout (simplified):

```text
RAG_demo/
├─ app/
│  ├─ demo_app.py               # FastAPI app (HTTP routes)
│  ├─ __init__.py
│  ├─ services/
│  │  ├─ answer.py              # Simple Chinese answer composer
│  │  ├─ chunking.py            # Simple text chunking
│  │  ├─ embeddings.py          # Embedder: mock / local SentenceTransformer / remote HTTP
│  │  ├─ hybrid.py              # Vector + keyword hybrid ranking
│  │  ├─ keyword.py             # Keyword / BM25-style index
│  │  ├─ task_query.py          # Task status parsing, ranking, and resolvers (Step 2 core)
│  │  ├─ nl2sql_engine.py       # Tasks NL→JSON semantic IR (TaskQuerySpec)
│  │  ├─ sql_compiler.py        # TaskQuerySpec → read‑only SQL compiler (tasks / task_latest)
│  │  ├─ kg_lite.py             # KG-lite: persons/projects/categories/tags dictionary (from data/kg_data.json)
│  │  └─ llm_client.py          # LLMClient abstraction and providers (dummy / Ollama / OpenAI‑compatible)
│  ├─ vector_store/
│  │  ├─ faiss_store.py         # FAISS-based vector store (Windows‑friendly default)
│  │  └─ milvus_store.py        # Milvus-based vector store (enabled via compose profile)
│  └─ tasks_store/
│     ├─ base.py                # TasksStore abstraction
│     └─ sqlite_store.py        # SQLiteTasksStore (read‑only tasks DB + helper queries)
│
├─ data/
│  ├─ faiss/                    # FAISS index + metadata
│  ├─ tasks.db                  # Demo tasks SQLite DB (created by scripts/init_tasks_sqlite.py)
│  └─ kg_data.json              # KG-lite data: persons/projects/categories/tags
│
├─ embedder_server/
│  ├─ server.py                 # bge-small-zh embedding HTTP service (for Docker)
│  └─ requirements.txt          # embedder container dependencies
│
├─ scripts/
│  ├─ init_tasks_sqlite.py      # Initialize demo tasks data into data/tasks.db
│  ├─ batch_db_ask.py           # Batch-call /tasks/ask, print resolver_mode / NL IR / KG flags / Text2SQL info
│  ├─ extract_kg_from_tasks.py  # Preview distinct persons/projects/tags from tasks DB for KG-lite seeding
│  ├─ questions.txt             # Sample Chinese task questions for batch_db_ask
│  └─ demo_queries.http         # VSCode REST Client examples
│
├─ tests/
│  ├─ conftest.py               # Add project root to sys.path for app.* imports
│  └─ test_nl2sql_db_ask.py     # NL→JSON→SQL end‑to‑end tests (FastAPI TestClient)
│
├─ docs/
│  ├─ START_AND_TEST.md         # From-zero start / test guide
│  └─ INSTRUCTIONS_TASKS.md     # Step 2: task Q&A + NL→SQL design notes
│
├─ docker-compose.yml           # embedder + Milvus profiles
├─ Dockerfile                   # API service image
├─ Dockerfile.embedder          # Embedding service image (CPU/GPU)
├─ requirements.txt             # Python dependencies (API side)
├─ .env.example                 # Example environment variables
└─ README_minimal_rag_demo.md   # This document
```

---

## Key Features

- **Minimal RAG stack**
  - `/ingest`: ingest raw text by `doc_id` → chunk → embed → write into vector store + keyword index.
  - `/search`: pure vector search (FAISS or Milvus), supports filtering by `doc_id`/`source`/time.
  - `/search_kw`: keyword/BM25-style search.
  - `/search_hybrid`: hybrid vector + keyword merging.
  - `/reset`: clear vector store and keyword index (does not touch `tasks.db`).

- **Task status Q&A (`/tasks/ask`)**
  - Uses `TaskQueryEngine` (`app/services/task_query.py`) to handle Chinese task questions:
    - intent detection (status / progress / completion / list / summary);
    - entity resolution (person / task / project / tags), via multiple resolver modes:
      - `rules` / `embeddings` / `hybrid` / `hybrid_plus_rules` (non‑LLM baselines);
      - `hybrid_llm`: LLM NL→JSON + small model + FAISS + SQL compiler;
      - `text2sql`: direct LLM‑generated SQL (with AST safety checks).
  - Backed by `SQLiteTasksStore`, returning:
    - `answer` (short Chinese answer), `status`, `person`, `task`, `ts`, `id`;
    - `resolver_mode`, `sql`, `params`, `rows` (for debugging);
    - `candidates.*` (person/task candidate scores for non‑LLM resolvers);
    - `nl_ir` (the NL→JSON semantic IR, including `TaskQuerySpec.extra`).

- **NL→JSON→SQL experiment (`/db/ask`)**
  - A pure NL→IR→SQL pipeline for the `tasks`/`task_latest` tables, without natural‑language answer generation:
    - `query`: original NL;
    - `ir`: `TaskQuerySpec` JSON produced by `parse_task_query_nl` (LLM‑first, rules as fallback);
    - `sql`: read‑only SQL compiled from IR by `compile_tasks_sql`;
    - `params`: positional parameters tuple;
    - `rows`: raw records from SQLite after executing the query.

- **Text2SQL with AST validation**
  - When `resolver_mode="text2sql"` or `hybrid_llm` (Text2SQL branch enabled), `/tasks/ask` can:
    1. Use `parse_task_query_nl` to build a `TaskQuerySpec` IR hint.
    2. Build a Text2SQL prompt combining schema + IR hint, and call an LLM (e.g. `qwen3-coder:480b-cloud`) to get JSON `{"queries":[{"sql": "...", "description": "..."}]}`.
    3. For each SQL: rewrite + validate via `sqlglot` AST (`app/services/task_query.py`) and then run it against `SQLiteTasksStore.query`.
  - Safety checks include:
    - allow only `SELECT` statements referencing `task_latest`/`tasks` tables;
    - enforce a hard `LIMIT` (≤100 rows);
    - reject dangerous keywords/functions and positional placeholders;
    - normalize time windows, tag filters, and priority filters using the IR hint.

- **KG-lite semantic dictionary for tasks**
  - Implemented in `app/services/kg_lite.py` with data in `data/kg_data.json`.
  - Captures domain semantics as **data**, not scattered rules:
    - persons: canonical names + aliases, e.g. `"张工" / "老张"` → `"张三"`;
    - projects/systems: `"芯片项目" / "芯片平台"` → `"芯片"`; `"交付项目组"` → `"交付"`; `"E3D系统"` → `"E3D"`;
    - categories: `"安全专项"` / `"安监专项"` belong to category `"安监整改"`, which expands to tags `["整改","安全整改"]`.
  - NL→JSON: `_post_process_intent` calls `kg_lite.resolve_person/resolve_project/resolve_category_tags` to normalize `TaskQuerySpec.person/project/tags`, and records flags like `kg_enabled`, `kg_person_source`, `kg_project_source`, `kg_category_source` in `spec.extra`.
  - Text2SQL: `task_query._make_text2sql_ir_hint` includes canonical person/project/tags in the IR hint; `_rewrite_text2sql_query` uses them to align LLM SQL output (e.g. rewriting `person = '张工'` to `person = '张三'`, injecting missing `tags LIKE '%整改%'` filters).
  - Backends:
    - Default: `InMemoryKGBackend` loading from JSON.
    - Extensible via `KGBackend` Protocol; future backends (e.g. Neo4j) can be plugged in without changing IR/SQL/Text2SQL logic.

---

## Configuration (Environment Variables)

Common env vars (see also `docs/START_AND_TEST.md` and `.env.example`):

- **Vector store / storage**
  - `STORE=faiss|milvus` (default `faiss`)
  - `DATA_DIR`: vector data directory for FAISS (default `data`)
  - `MILVUS_HOST` / `MILVUS_PORT` / `MILVUS_COLLECTION`: used when `STORE=milvus`

- **Task resolver modes**
  - `RESOLVER=rules|embeddings|hybrid|hybrid_plus_rules|hybrid_llm|text2sql` (default `hybrid`)

- **Embeddings**
  - `MOCK_EMB=1|0`: enable mock embeddings (1) or real model/HTTP (0)
  - `MODEL_NAME`: local `SentenceTransformer` model name (e.g. `sentence-transformers/all-MiniLM-L6-v2` or `BAAI/bge-small-zh-v1.5`)
  - `EMB_DIM`: embedding dimension (MiniLM=384, bge-small-zh=512)
  - `EMB_URL`: remote embedding HTTP endpoint (e.g. `http://localhost:8080/embeddings`)
  - `EMB_TIMEOUT`: HTTP timeout in seconds (default 12)

- **Tasks store / DB**
  - `TASKS_BACKEND=sqlite` (current implementation)
  - `TASKS_DB`: SQLite file path (default `data/tasks.db`)

- **LLM / NL2SQL / Text2SQL**
  - `LLM_ENABLED=true|false`
  - `LLM_PROVIDER=dummy|ollama|openai` (`openai` here means “OpenAI‑compatible”, e.g. DashScope)
  - `LLM_MODEL`: default chat/model name, e.g. `qwen2.5-coder:7b`
  - `LLM_TEXT2SQL_MODEL`: optional dedicated Text2SQL model (default `qwen3-coder:480b-cloud`)
  - `LLM_TEXT2SQL_PROVIDER` / `LLM_TEXT2SQL_OLLAMA_BASE_URL` / `LLM_TEXT2SQL_OPENAI_BASE_URL` / `LLM_TEXT2SQL_API_KEY`: overrides for Text2SQL provider/endpoint; fall back to `LLM_PROVIDER` settings when unset.
  - `LLM_OLLAMA_BASE_URL`: base URL for Ollama HTTP API (default `http://localhost:11434`)
  - `LLM_OPENAI_BASE_URL`: base URL for OpenAI‑compatible APIs (default DashScope URL in this demo)
  - `LLM_API_KEY`: required when `LLM_PROVIDER=openai`/`dashscope`
  - `TASKS_NL2SQL_LLM=1`: prefer LLM for NL→JSON IR (`TaskQuerySpec`); fall back to rule‑based parser on failure.

---

## Development / Run Modes

Typical local setups:

- **Local API + mock embeddings (fastest to start)**
  1. Create and activate venv, install deps
     - `python -m venv venv`
     - `./venv/Scripts/Activate` (PowerShell) or `source venv/bin/activate` (Bash)
     - `pip install -r requirements.txt`
  2. Set env vars
     - `STORE=faiss`
     - `MOCK_EMB=1`
  3. Start API
     - `uvicorn app.demo_app:app --host 0.0.0.0 --port 8000 --reload`

- **Local API + Dockerized bge-small-zh embedder**
  1. Build and start embedder container
     - `docker compose build embedder`
     - `docker compose up -d embedder`
  2. Set env vars
     - `STORE=faiss`
     - `RESOLVER=hybrid` (or `embeddings` / `hybrid_llm`)
     - `MOCK_EMB=0`
     - `EMB_URL=http://localhost:8080/embeddings`
     - `EMB_DIM=512`
  3. Start API as above

- **Milvus mode (optional, recommended on Linux/WSL)**
  - `docker compose --profile milvus up -d`
  - Set `STORE=milvus` and related Milvus env vars.

---

## Where to Read Next

- From zero to a working demo (init DB, start API, run tests):
  - `docs/START_AND_TEST.md`
- Deep dive into task Q&A + NL→SQL design:
  - `docs/INSTRUCTIONS_TASKS.md`
  - `app/services/task_query.py`
  - `app/services/nl2sql_engine.py`
  - `app/services/sql_compiler.py`
  - `app/services/llm_client.py`
  - `app/services/kg_lite.py`

---

## Text2SQL & SQL AST Overview

The tasks subsystem contains an optional Text2SQL pipeline that lets an LLM propose SQL while the backend enforces safety and applies structured hints:

- **Entry points & modes**
  - `/tasks/ask` when `resolver_mode="text2sql"` or `resolver_mode="hybrid_llm"`:
    1. Build a `TaskQuerySpec` via `parse_task_query_nl` (LLM‑first, rule fallback).
    2. Construct a Text2SQL prompt with schema + IR hint (including KG‑normalized person/project/tags).
    3. Ask the LLM to return JSON `{"queries":[{"sql":"...","description":"..."}]}`.
    4. For each query, apply `_rewrite_text2sql_query` + AST validation, then execute with `SQLiteTasksStore.query`.

- **Safety and rewriting (`app/services/task_query.py`)**
  - Use `sqlglot` to parse SQL into AST:
    - allow only top‑level `SELECT`;
    - allow only `task_latest` / `tasks` as tables;
    - inject or cap `LIMIT` (max 100 rows) and normalize `ORDER BY` placement;
    - reject dangerous keywords/functions and placeholders (`?`, named params, `DATE_SUB`, etc.).
  - Beyond AST:
    - normalize time windows (`now-7d`, `start_of_week`, etc.) into concrete epoch millis;
    - inject `tags LIKE '%...%'` filters for tag‑driven queries;
    - normalize priority conditions (e.g. “高优P1” → `priority = 1`);
    - use KG‑aware IR hints to rewrite `person` / `project` literals and align them with canonical values.

- **Debug fields**
  - When Text2SQL fails, `/tasks/ask` returns:
    - `error`: e.g. `text2sql_invalid_sql`, `text2sql_db_query_failed`, `text2sql_llm_failed`;
    - `reason`: detailed cause (AST or DB error);
    - `text2sql_raw_response`: raw LLM output (for prompt tuning);
    - `text2sql`: per‑query execution info (`sql`, `description`, `rows`);
    - `text2sql_model`, `text2sql_provider`: which LLM/endpoint was used.
  - `scripts/batch_db_ask.py` can be used to run a set of NL questions against `/tasks/ask` and inspect `resolver_mode`, `intent`, `nl_ir.extra`, and Text2SQL details.

