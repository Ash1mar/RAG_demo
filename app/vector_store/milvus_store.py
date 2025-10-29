from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)

from app.vector_store.base import VectorStore


class MilvusVectorStore(VectorStore):
    """Vector store backed by Milvus."""

    def __init__(
        self,
        dim: int,
        host: str = "localhost",
        port: int | str = 19530,
        collection_name: str = "rag_chunks",
        index_type: str = "IVF_FLAT",
        metric_type: str = "IP",
        nlist: int = 1024,
        nprobe: int = 16,
    ) -> None:
        self.dim = dim
        self.collection_name = collection_name
        self._index_type = index_type
        self._metric_type = metric_type
        self._nlist = max(1, nlist)
        self._nprobe = max(1, nprobe)

        connections.connect(alias="default", host=host, port=str(port))
        self.collection = self._get_or_create_collection()

    # ------------------------------------------------------------------
    # VectorStore API
    # ------------------------------------------------------------------
    def add_texts(
        self,
        doc_id: str,
        chunks: List[str],
        embeddings: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if embeddings.ndim != 2 or embeddings.shape[1] != self.dim:
            raise ValueError(f"embeddings shape {embeddings.shape} != (N, {self.dim})")

        metadata = metadata or {}
        source_val = metadata.get("source") or ""
        ts_val = metadata.get("ts")
        ts_val = int(ts_val) if ts_val is not None else -1

        # Normalize for cosine similarity via inner product
        embs = embeddings.astype(np.float32)
        norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-10
        embs = embs / norms

        entities = [
            [doc_id for _ in chunks],
            [source_val for _ in chunks],
            [ts_val for _ in chunks],
            chunks,
            embs.tolist(),
        ]

        self.collection.insert(entities)
        self.collection.flush()

    def search(
        self,
        query_emb: np.ndarray,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if self.collection.num_entities == 0:
            return []

        filters = filters or {}

        vector = query_emb.astype(np.float32).reshape(1, -1)
        norm = np.linalg.norm(vector, axis=1, keepdims=True) + 1e-10
        vector = (vector / norm).tolist()

        expr = self._build_expr(filters)
        search_params = {
            "metric_type": self._metric_type,
            "params": {"nprobe": self._nprobe},
        }

        results = self.collection.search(
            data=vector,
            anns_field="embedding",
            param=search_params,
            limit=max(1, top_k),
            expr=expr,
            output_fields=["doc_id", "text", "source", "ts"],
        )

        hits: List[Dict[str, Any]] = []
        for hit in results[0]:
            meta = {
                "doc_id": hit.entity.get("doc_id"),
                "text": hit.entity.get("text"),
            }
            source_val = hit.entity.get("source")
            if source_val:
                meta["source"] = source_val
            ts_val = hit.entity.get("ts")
            if ts_val is not None and int(ts_val) >= 0:
                meta["ts"] = int(ts_val)
            meta["score"] = float(hit.score)
            meta["_id"] = hit.id
            hits.append(meta)
        return hits

    def reset(self) -> None:
        if utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)
        self.collection = self._create_collection()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _get_or_create_collection(self) -> Collection:
        if utility.has_collection(self.collection_name):
            collection = Collection(self.collection_name)
            collection.load()
            return collection
        return self._create_collection()

    def _create_collection(self) -> Collection:
        fields = [
            FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="ts", dtype=DataType.INT64),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2048),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
        ]
        schema = CollectionSchema(fields=fields, description="Minimal RAG demo chunks")
        collection = Collection(name=self.collection_name, schema=schema)

        index_params = {
            "index_type": self._index_type,
            "metric_type": self._metric_type,
            "params": {"nlist": self._nlist},
        }
        collection.create_index(field_name="embedding", index_params=index_params)
        collection.load()
        return collection

    @staticmethod
    def _build_expr(filters: Dict[str, Any]) -> Optional[str]:
        clauses: List[str] = []
        doc_id = filters.get("doc_id")
        if doc_id:
            clauses.append(f'doc_id == "{doc_id}"')
        source = filters.get("source")
        if source:
            clauses.append(f'source == "{source}"')
        date_from = filters.get("date_from")
        if date_from is not None:
            clauses.append(f"ts >= {int(date_from)}")
        date_to = filters.get("date_to")
        if date_to is not None:
            clauses.append(f"ts <= {int(date_to)}")
        if not clauses:
            return None
        return " and ".join(clauses)
