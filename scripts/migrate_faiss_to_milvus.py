#!/usr/bin/env python3
"""Migrate persisted FAISS index/meta into Milvus."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import faiss  # type: ignore
import numpy as np

# Allow importing the demo app modules when executed from repo root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.vector_store.milvus_store import MilvusVectorStore  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Migrate FAISS index/meta into Milvus")
    parser.add_argument("--data-dir", default="data", help="Directory containing index.faiss and meta.json")
    parser.add_argument("--index-file", default="index.faiss", help="FAISS index filename")
    parser.add_argument("--meta-file", default="meta.json", help="Metadata JSON filename")
    parser.add_argument("--milvus-host", default="127.0.0.1", help="Milvus host")
    parser.add_argument("--milvus-port", default=19530, type=int, help="Milvus port")
    parser.add_argument(
        "--collection",
        default="rag_chunks",
        help="Milvus collection name to insert into",
    )
    parser.add_argument(
        "--drop-existing",
        action="store_true",
        help="Drop and recreate the Milvus collection before inserting",
    )
    return parser.parse_args()


def load_faiss_index(index_path: Path) -> faiss.Index:
    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {index_path}")
    return faiss.read_index(str(index_path))


def load_metadata(meta_path: Path) -> List[Dict[str, object]]:
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")
    with meta_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return payload.get("meta", [])


def group_by_doc(meta: List[Dict[str, object]]) -> Dict[str, List[Dict[str, object]]]:
    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for rec in meta:
        doc = str(rec.get("doc_id"))
        grouped[doc].append(rec)
    return grouped


def reconstruct_vectors(index: faiss.Index, ids: List[int]) -> np.ndarray:
    vectors = []
    for chunk_id in ids:
        vec = index.reconstruct(int(chunk_id))
        vectors.append(np.array(vec, dtype=np.float32))
    return np.vstack(vectors)


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    index_path = data_dir / args.index_file
    meta_path = data_dir / args.meta_file

    index = load_faiss_index(index_path)
    meta = load_metadata(meta_path)
    if not meta:
        print("No metadata found; nothing to migrate.")
        return

    dimension = index.d
    store = MilvusVectorStore(
        dim=dimension,
        host=args.milvus_host,
        port=args.milvus_port,
        collection_name=args.collection,
    )
    if args.drop_existing:
        store.reset()

    grouped = group_by_doc(meta)

    total_chunks = 0
    for doc_id, records in grouped.items():
        chunk_ids = [int(rec["_id"]) for rec in records if "_id" in rec]
        texts = [str(rec.get("text", "")) for rec in records]
        embs = reconstruct_vectors(index, chunk_ids)
        # Normalize again for safety
        norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-10
        embs = embs / norms

        metadata: Dict[str, Optional[object]] = {}
        # assume consistent metadata per document
        for key in ("source", "ts"):
            values = [rec.get(key) for rec in records if rec.get(key) is not None]
            if values:
                metadata[key] = values[0]

        store.add_texts(doc_id, texts, embs, metadata=metadata)
        total_chunks += len(records)

    print(f"Migrated {len(grouped)} docs / {total_chunks} chunks into Milvus collection '{args.collection}'.")


if __name__ == "__main__":
    main()
