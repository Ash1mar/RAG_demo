# FastAPI 入口：/health /ingest /search /reset
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List, Dict, Any
import os

from app.services.embeddings import Embedder
from app.services.chunking import simple_chunk
from app.vector_store.faiss_store import FaissVectorStore
from app.services.answer import build_answer
from app.services.keyword import KeywordIndex
from app.services.hybrid import merge_scores


app = FastAPI(title="Minimal RAG Demo", version="0.1.0")

# ---- Global singletons (demo 级；生产迁移到 DI/Factory) ----
EMBEDDER = Embedder(model_name=os.getenv("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"),
                    use_mock=os.getenv("MOCK_EMB", "0") == "1",
                    dim=int(os.getenv("EMB_DIM", "384")))
VSTORE = FaissVectorStore(dim=EMBEDDER.dim)

KW_INDEX = KeywordIndex()


# 新增从环境变量读取 DATA_DIR（可选），默认 data/
VSTORE = FaissVectorStore(dim=EMBEDDER.dim, data_dir=os.getenv("DATA_DIR", "data"))

# ---- Schemas ----
class IngestReq(BaseModel):
    doc_id: str
    text: str

class SearchResp(BaseModel):
    results: List[Dict[str, Any]]

# ---- Endpoints ----
@app.get("/health")
def health():
    return {"status": "ok", "embedder": "mock" if EMBEDDER.use_mock else "sbert", "dim": EMBEDDER.dim}

@app.post("/ingest")
def ingest(req: IngestReq):
    if not req.text.strip():
        raise HTTPException(400, "empty text")
    chunks = simple_chunk(req.text)
    embs = EMBEDDER.encode(chunks)
    VSTORE.add_texts(req.doc_id, chunks, embs)
    KW_INDEX.add(req.doc_id, chunks)
    return {"doc_id": req.doc_id, "chunks": len(chunks), "indexed": len(chunks)}

@app.get("/search", response_model=SearchResp)
def search(q: str, k: int = 5):
    if not q.strip():
        raise HTTPException(400, "empty query")
    q_emb = EMBEDDER.encode([q])[0]
    hits = VSTORE.search(q_emb, top_k=k)
    return {"results": hits}

@app.post("/reset")
def reset():
    VSTORE.reset()
    KW_INDEX.reset()
    return {"status": "reset"}

@app.get("/answer")
def answer(
    q: str = Query(..., description="User question / query"),
    k: int = Query(6, ge=1, le=50, description="top-k chunks to consider"),
    max_chars: int = Query(600, ge=100, le=4000, description="max characters for composed answer"),
    include_scores: bool = Query(True, description="include scores in citations"),
):
    """
    Minimal Answer（无LLM版本）：
    - 使用向量检索的 top-k 片段
    - 以高分到低分拼接成一段可读文本
    - 返回 citations（含 doc_id、text、可选 score）
    """
    if not q.strip():
        from fastapi import HTTPException
        raise HTTPException(400, "empty query")

    q_emb = EMBEDDER.encode([q])[0]
    hits = VSTORE.search(q_emb, top_k=k)
    payload = build_answer(hits, max_chars=max_chars, include_scores=include_scores)
    return payload

@app.get("/search_kw")
def search_kw(
    q: str = Query(..., description="Keyword query"),
    k: int = Query(5, ge=1, le=50, description="top-k"),
):
    if not q.strip():
        from fastapi import HTTPException
        raise HTTPException(400, "empty query")
    hits = KW_INDEX.search(q, top_k=k)
    return {"results": hits}

@app.get("/search_hybrid")
def search_hybrid(
    q: str = Query(..., description="Hybrid query"),
    k: int = Query(5, ge=1, le=50, description="top-k"),
    alpha: float = Query(0.6, ge=0.0, le=1.0, description="weight for vector score (0~1)"),
):
    if not q.strip():
        from fastapi import HTTPException
        raise HTTPException(400, "empty query")

    # 向量检索 top-k（可以适当提升为 2k，增加覆盖）
    q_emb = EMBEDDER.encode([q])[0]
    vec_hits = VSTORE.search(q_emb, top_k=k * 2)

    # 关键词检索 top-k（同样扩大覆盖）
    kw_hits = KW_INDEX.search(q, top_k=k * 2)

    fused = merge_scores(vec_hits, kw_hits, k=k, alpha=alpha)
    return {"results": fused}

