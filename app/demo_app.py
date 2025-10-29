from __future__ import annotations

from datetime import datetime, timezone
from os import getenv
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from app.services.answer import build_answer
from app.services.chunking import simple_chunk
from app.services.embeddings import Embedder
from app.services.hybrid import merge_scores
from app.services.keyword import KeywordIndex
from app.vector_store.faiss_store import FaissVectorStore
from app.vector_store.milvus_store import MilvusVectorStore


app = FastAPI(title="Minimal RAG Demo", version="0.1.0")

# ---- Global singletons (demo 级；生产迁移到 DI/Factory) ----
EMBEDDER = Embedder(
    model_name=getenv("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"),
    use_mock=getenv("MOCK_EMB", "0") == "1",
    dim=int(getenv("EMB_DIM", "384")),
)

store_type = getenv("STORE", "faiss").lower()
if store_type == "milvus":
    VSTORE = MilvusVectorStore(
        dim=EMBEDDER.dim,
        host=getenv("MILVUS_HOST", "localhost"),
        port=int(getenv("MILVUS_PORT", "19530")),
        collection_name=getenv("MILVUS_COLLECTION", "rag_chunks"),
    )
else:
    VSTORE = FaissVectorStore(dim=EMBEDDER.dim, data_dir=getenv("DATA_DIR", "data"))

KW_INDEX = KeywordIndex()


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
