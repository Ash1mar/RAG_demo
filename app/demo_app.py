# FastAPI 入口：/health /ingest /search /reset
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import os

from app.services.embeddings import Embedder
from app.services.chunking import simple_chunk
from app.vector_store.faiss_store import FaissVectorStore

app = FastAPI(title="Minimal RAG Demo", version="0.1.0")

# ---- Global singletons (demo 级；生产迁移到 DI/Factory) ----
EMBEDDER = Embedder(model_name=os.getenv("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"),
                    use_mock=os.getenv("MOCK_EMB", "0") == "1",
                    dim=int(os.getenv("EMB_DIM", "384")))
VSTORE = FaissVectorStore(dim=EMBEDDER.dim)

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
    return {"status": "reset"}
