from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Embedding Server", version="0.1.0")


class EmbeddingRequest(BaseModel):
    inputs: List[str]
    normalize: bool = True


MODEL_NAME = os.getenv("MODEL_NAME", "BAAI/bge-small-zh-v1.5")
EMBED_DEVICE = os.getenv("EMBED_DEVICE", "auto")  # 'auto' | 'cpu' | 'cuda'
EMBED_BATCH = int(os.getenv("EMBED_BATCH", "32"))

_model = None
_dim: Optional[int] = None
_device = "cpu"


def _load_model():
    global _model, _dim, _device
    if _model is not None:
        return
    from sentence_transformers import SentenceTransformer
    import torch

    if EMBED_DEVICE == "auto":
        _device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        _device = EMBED_DEVICE
    _model = SentenceTransformer(MODEL_NAME, device=_device)
    # probe dim
    tmp = _model.encode(["probe"], normalize_embeddings=True)
    _dim = int(np.asarray(tmp).shape[1])


@app.get("/health")
def health():
    try:
        _load_model()
        return {
            "status": "ok",
            "model": MODEL_NAME,
            "device": _device,
            "dim": _dim,
            "batch": EMBED_BATCH,
        }
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


@app.post("/embeddings")
def embeddings(req: EmbeddingRequest):
    if not req.inputs:
        raise HTTPException(400, "inputs cannot be empty")
    _load_model()
    try:
        embs = _model.encode(
            req.inputs,
            normalize_embeddings=bool(req.normalize),
            batch_size=EMBED_BATCH,
            convert_to_numpy=True,
        )
        # ensure float32
        arr = np.asarray(embs, dtype=np.float32)
        return {"embeddings": arr.tolist()}
    except Exception as exc:
        raise HTTPException(500, f"embedding failed: {exc}")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("EMBED_PORT", "8080"))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)

