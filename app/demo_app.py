from __future__ import annotations

from app.config_loader import load_app_config

# Load env-file config as early as possible so import-time settings (e.g. app.config.llm_settings)
# see the intended values.
load_app_config()

from datetime import datetime, timezone
import os
from os import getenv
import sys
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from app.services.answer import build_answer
from app.services.chunking import simple_chunk
from app.services.embeddings import Embedder
from app.services.hybrid import merge_scores
from app.services.keyword import KeywordIndex
from app.vector_store.faiss_store import FaissVectorStore
from app.services.task_query import TaskQueryEngine
from app.services.nl2sql_engine import parse_task_query_nl, build_task_query_plan
from app.services.sql_compiler import compile_tasks_sql, TaskSqlCompileError
from app.tasks_store.base import TasksStore
from app.tasks_store.sqlite_store import SQLiteTasksStore, SQLiteTasksConfig


app = FastAPI(title="Minimal RAG Demo", version="0.1.0")

# ---- Global singletons (demo 级；生产迁移) ----
def _running_under_pytest() -> bool:
    if os.getenv("PYTEST_CURRENT_TEST"):
        return True
    if "pytest" in sys.modules:
        return True
    return any("pytest" in arg.lower() for arg in sys.argv)


EMBEDDER = Embedder(
    model_name=getenv("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"),
    use_mock=(getenv("MOCK_EMB", "0") == "1") or _running_under_pytest(),
    dim=int(getenv("EMB_DIM", "384")),
)

store_type = getenv("STORE", "faiss").lower()
if store_type == "milvus":
    try:
        # Lazy import so pymilvus is only required when using Milvus backend
        from app.vector_store.milvus_store import MilvusVectorStore  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Milvus backend requested but unavailable. Install pymilvus (Linux/WSL recommended) or set STORE=faiss."
        ) from exc
    VSTORE = MilvusVectorStore(
        dim=EMBEDDER.dim,
        host=getenv("MILVUS_HOST", "localhost"),
        port=int(getenv("MILVUS_PORT", "19530")),
        collection_name=getenv("MILVUS_COLLECTION", "rag_chunks"),
    )
else:
    VSTORE = FaissVectorStore(dim=EMBEDDER.dim, data_dir=getenv("DATA_DIR", "data"))

KW_INDEX = KeywordIndex()

# ---- Tasks store (read-only) ----
tasks_backend = getenv("TASKS_BACKEND", "sqlite").lower()
TASKS: TasksStore
if tasks_backend == "sqlite":
    TASKS = SQLiteTasksStore(SQLiteTasksConfig(db_path=getenv("TASKS_DB", "data/tasks.db")))
elif tasks_backend == "mssql":
    try:
        from app.tasks_store.mssql_store import MSSQLTasksStore, MSSQLTasksConfig
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "MSSQL backend requested but pyodbc is unavailable. Install pyodbc and the SQL Server ODBC driver."
        ) from exc
    TASKS = MSSQLTasksStore(
        MSSQLTasksConfig(
            server=getenv("TASKS_MSSQL_SERVER", "127.0.0.1,1433"),
            database=getenv("TASKS_MSSQL_DATABASE", "fact_tasks"),
            user=getenv("TASKS_MSSQL_USER", "sa"),
            password=getenv("TASKS_MSSQL_PASSWORD", ""),
            driver=getenv("TASKS_MSSQL_DRIVER", "ODBC Driver 18 for SQL Server"),
            encrypt=(getenv("TASKS_MSSQL_ENCRYPT", "yes").lower() != "no"),
            trust_server_certificate=(
                getenv("TASKS_MSSQL_TRUST_CERT", "yes").lower() != "no"
            ),
            timeout_sec=float(getenv("TASKS_MSSQL_TIMEOUT_SEC", "5")),
        )
    )
else:
    # Placeholder for future backends (e.g., KG). Keep API stable.
    TASKS = SQLiteTasksStore(SQLiteTasksConfig(db_path=getenv("TASKS_DB", "data/tasks.db")))

# Non-LLM task status query engine
resolver_mode = getenv("RESOLVER", "hybrid").lower()  # rules | embeddings | hybrid
TQ_ENGINE = TaskQueryEngine(tasks_store=TASKS, embedder=EMBEDDER, resolver_mode=resolver_mode)


# ---- Schemas ----
class IngestReq(BaseModel):
    doc_id: str
    text: str
    source: Optional[str] = None
    ts: Optional[str] = Field(
        None,
        description="Optional ISO-8601 timestamp or epoch milliseconds for the document",
    )


class SearchResp(BaseModel):
    results: List[Dict[str, Any]]


# ---- Endpoints ----
@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "embedder": "mock" if EMBEDDER.use_mock else "sbert",
        "dim": EMBEDDER.dim,
        "vector_store": store_type,
        "tasks_store": tasks_backend,
        "tasks_ready": TASKS.ready(),
        "resolver_mode": resolver_mode,
    }


@app.post("/ingest")
def ingest(req: IngestReq) -> Dict[str, Any]:
    if not req.text.strip():
        raise HTTPException(400, "empty text")
    chunks = simple_chunk(req.text)
    embs = EMBEDDER.encode(chunks)
    metadata = {
        "source": req.source,
        "ts": _parse_timestamp_to_millis(req.ts) if req.ts else None,
    }
    VSTORE.add_texts(req.doc_id, chunks, embs, metadata=metadata)
    KW_INDEX.add(req.doc_id, chunks, metadata=metadata)
    return {"doc_id": req.doc_id, "chunks": len(chunks), "indexed": len(chunks)}


@app.get("/search", response_model=SearchResp)
def search(
    q: str,
    k: int = 5,
    doc_id: Optional[str] = Query(None, description="Filter by document ID"),
    source: Optional[str] = Query(None, description="Filter by source label"),
    date_from: Optional[str] = Query(None, description="Start timestamp (ISO-8601 or epoch ms)"),
    date_to: Optional[str] = Query(None, description="End timestamp (ISO-8601 or epoch ms)"),
) -> Dict[str, Any]:
    if not q.strip():
        raise HTTPException(400, "empty query")
    q_emb = EMBEDDER.encode([q])[0]
    filters = _build_filters(doc_id=doc_id, source=source, date_from=date_from, date_to=date_to)
    hits = VSTORE.search(q_emb, top_k=k, filters=filters)
    return {"results": hits}


@app.post("/reset")
def reset() -> Dict[str, str]:
    VSTORE.reset()
    KW_INDEX.reset()
    return {"status": "reset"}


# ---- Minimal API for tasks status (connectivity test) ----
@app.get("/tasks/status")
def task_status(person: str, task: str) -> Dict[str, Any]:
    """Return latest status for a given person+task.

    Example queries (for local sample DB):
    - 张三, 提交9月周报? -> DONE
    - 张三, E3D接口联调  -> TODO
    - 李四, 整理工艺包V2 -> DONE
    """
    person = person.strip()
    task = task.strip()
    if not person or not task:
        raise HTTPException(400, "person and task are required")
    rec = TASKS.get_latest_status(person, task)
    if not rec:
        return {"found": False, "person": person, "task": task}
    return {
        "found": True,
        "person": rec["person"],
        "task": rec["task"],
        "status": rec["status"],
        "ts": rec["ts"],
        "id": rec["id"],
    }


@app.get("/tasks/ask")
def tasks_ask(q: str, topk: int = 3, thresh: Optional[float] = None) -> Dict[str, Any]:
    """

 
    """
    if not q.strip():
        raise HTTPException(400, "empty query")

    payload = TQ_ENGINE.answer(q, topk=topk, thresh=thresh)
    try:
        # For engines that do not populate IR fields themselves, backfill from a fresh parse.
        if "nl_ir" not in payload or "ir" not in payload:
            spec = parse_task_query_nl(q)
            if "nl_ir" not in payload:
                payload["nl_ir"] = spec.dict()
            if "ir" not in payload:
                payload["ir"] = build_task_query_plan(spec)
            extra = spec.extra or {}
            if extra.get("kg_enabled"):
                payload["kg_enabled"] = True
        else:
            # When nl_ir is already provided by the engine, try to read KG flag from it.
            try:
                nl_ir = payload.get("nl_ir") or {}
                extra = nl_ir.get("extra") or {}
                if extra.get("kg_enabled"):
                    payload["kg_enabled"] = True
            except Exception:
                pass
    except Exception:
        if "nl_ir" not in payload:
            payload["nl_ir"] = {"raw_query": q, "error": "failed_to_serialize_ir"}
        if "ir" not in payload:
            payload["ir"] = {"raw_query": q, "error": "failed_to_build_query_plan"}
    return payload
@app.get("/db/ask")
def db_ask(q: str = Query(..., description="Natural language task query for direct NL→JSON→SQL experiment")) -> Dict[str, Any]:
    """NL→JSON→SQL 闭环实验端点：直接以 tasks 表为目标的只读查询?
    流程?    1. 调用 parse_task_query_nl(q) 得到 TaskQuerySpec 语义 IR?    2. 调用 compile_tasks_sql(spec) 生成只读 SQL 和参数；
    3. 使用 SQLiteTasksStore.query(sql, params) 执行查询?    4. 返回结构�?JSON，包含原�?query、IR、SQL ?rows，用于调?NL→SQL 链路?
    注意：本端点不会生成自然语言回答，也不会替代 /tasks/ask 的逻辑?    """
    text = (q or "").strip()
    if not text:
        raise HTTPException(400, "empty query")

    # 1) NL -> IR
    spec = parse_task_query_nl(text)

    # 2) IR -> SQL（只读、tasks 单表）    
    try:
        compiled = compile_tasks_sql(spec)
    except TaskSqlCompileError as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "cannot_compile_sql",
                "reason": str(exc),
            },
        )

    # 3) 执行 SQL（只读查询）
    try:
        rows = TASKS.query(compiled.sql, compiled.params)
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(
            status_code=500,
            detail={
                "error": "db_query_failed",
                "reason": str(exc),
            },
        )

    # 4) 
    return {
        "query": text,
        "ir": spec.dict(),
        "sql": compiled.sql,
        "params": compiled.params,
        "rows": rows,
        "kg_enabled": bool((spec.extra or {}).get("kg_enabled")),
    }


@app.post("/tasks/reload")
def tasks_reload() -> Dict[str, Any]:
    """�ؽ�ʵ����������������ݿ����¼��غ�ѡ��"""
    return {"reloaded": True, **TQ_ENGINE.reload()}


@app.get("/answer")
def answer(
    q: str = Query(..., description="User question / query"),
    k: int = Query(6, ge=1, le=50, description="top-k chunks to consider"),
    max_chars: int = Query(600, ge=100, le=4000, description="max characters for composed answer"),
    include_scores: bool = Query(True, description="include scores in citations"),
) -> Dict[str, Any]:
    if not q.strip():
        raise HTTPException(400, "empty query")

    q_emb = EMBEDDER.encode([q])[0]
    hits = VSTORE.search(q_emb, top_k=k)
    payload = build_answer(hits, max_chars=max_chars, include_scores=include_scores)
    return payload


@app.get("/search_kw")
def search_kw(
    q: str = Query(..., description="Keyword query"),
    k: int = Query(5, ge=1, le=50, description="top-k"),
    doc_id: Optional[str] = Query(None, description="Filter by document ID"),
    source: Optional[str] = Query(None, description="Filter by source label"),
    date_from: Optional[str] = Query(None, description="Start timestamp (ISO-8601 or epoch ms)"),
    date_to: Optional[str] = Query(None, description="End timestamp (ISO-8601 or epoch ms)"),
) -> Dict[str, Any]:
    if not q.strip():
        raise HTTPException(400, "empty query")
    filters = _build_filters(doc_id=doc_id, source=source, date_from=date_from, date_to=date_to)
    hits = _filter_results(KW_INDEX.search(q, top_k=k), filters)
    return {"results": hits}


@app.get("/search_hybrid")
def search_hybrid(
    q: str = Query(..., description="Hybrid query"),
    k: int = Query(5, ge=1, le=50, description="top-k"),
    alpha: float = Query(0.6, ge=0.0, le=1.0, description="weight for vector score (0~1)"),
    doc_id: Optional[str] = Query(None, description="Filter by document ID"),
    source: Optional[str] = Query(None, description="Filter by source label"),
    date_from: Optional[str] = Query(None, description="Start timestamp (ISO-8601 or epoch ms)"),
    date_to: Optional[str] = Query(None, description="End timestamp (ISO-8601 or epoch ms)"),
) -> Dict[str, Any]:
    if not q.strip():
        raise HTTPException(400, "empty query")

    q_emb = EMBEDDER.encode([q])[0]
    filters = _build_filters(doc_id=doc_id, source=source, date_from=date_from, date_to=date_to)
    vec_hits = VSTORE.search(q_emb, top_k=k * 2, filters=filters)
    kw_hits = _filter_results(KW_INDEX.search(q, top_k=k * 2), filters)

    fused = merge_scores(vec_hits, kw_hits, k=k, alpha=alpha)
    fused = _filter_results(fused, filters)
    return {"results": fused}


def _parse_timestamp_to_millis(value: str) -> int:
    value = value.strip()
    if not value:
        raise HTTPException(400, "timestamp cannot be empty")
    if value.isdigit():
        return int(value)
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise HTTPException(400, f"invalid timestamp: {value}") from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _build_filters(
    *,
    doc_id: Optional[str],
    source: Optional[str],
    date_from: Optional[str],
    date_to: Optional[str],
) -> Dict[str, Any]:
    filters: Dict[str, Any] = {}
    if doc_id:
        filters["doc_id"] = doc_id
    if source:
        filters["source"] = source
    if date_from:
        filters["date_from"] = _parse_timestamp_to_millis(date_from)
    if date_to:
        filters["date_to"] = _parse_timestamp_to_millis(date_to)
    return filters


def _filter_results(results: List[Dict[str, Any]], filters: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not filters:
        return results
    filtered: List[Dict[str, Any]] = []
    for item in results:
        if _match_filters(item, filters):
            filtered.append(item)
    return filtered


def _match_filters(meta: Dict[str, Any], filters: Dict[str, Any]) -> bool:
    doc_id = filters.get("doc_id")
    if doc_id and meta.get("doc_id") != doc_id:
        return False
    source = filters.get("source")
    if source and meta.get("source") != source:
        return False

    date_from = filters.get("date_from")
    date_to = filters.get("date_to")
    if date_from is None and date_to is None:
        return True
    ts_val = meta.get("ts")
    if ts_val is None:
        return False
    if date_from is not None and int(ts_val) < int(date_from):
        return False
    if date_to is not None and int(ts_val) > int(date_to):
        return False
    return True
