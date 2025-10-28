# app/vector_store/faiss_store.py
from __future__ import annotations
from typing import List, Dict, Any
from pathlib import Path
import json
import os
import tempfile

import numpy as np
import faiss

from app.vector_store.base import VectorStore


class FaissVectorStore(VectorStore):
    """
    FAISS 向量库（可持久化版）
    - 向量相似度：余弦（做 L2 归一化 + 内积）
    - 使用 IndexIDMap 确保向量有稳定的 int64 ID（便于持久化与删除）
    - 将索引和元数据分别落盘：
        index:  data/index.faiss
        meta :  data/meta.json   形如 { "next_id": int, "meta": [{_id, doc_id, text}, ...] }
    - API 与原版一致：add_texts / search / reset
    """

    def __init__(
        self,
        dim: int,
        data_dir: str = "data",
        index_file: str = "index.faiss",
        meta_file: str = "meta.json",
    ):
        self.dim = dim
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self._idx_path = self.data_dir / index_file
        self._meta_path = self.data_dir / meta_file

        # 用内积配合归一化实现余弦
        self.index: faiss.Index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
        # 元数据列表 + id->meta 快速映射
        self.meta: List[Dict[str, Any]] = []
        self._meta_by_id: Dict[int, Dict[str, Any]] = {}

        # 递增 id（确保重启后不冲突）
        self._next_id: int = 0

        self._load_if_exists()

    # -------------------------
    # Public API
    # -------------------------
    def add_texts(self, doc_id: str, chunks: List[str], embeddings: np.ndarray) -> None:
        if embeddings.ndim != 2 or embeddings.shape[1] != self.dim:
            raise ValueError(f"embeddings shape {embeddings.shape} != (N, {self.dim})")

        # 归一化，配合内积得到余弦相似度
        embs = embeddings.astype(np.float32)
        faiss.normalize_L2(embs)

        n = embs.shape[0]
        ids = np.arange(self._next_id, self._next_id + n, dtype=np.int64)
        self._next_id += n

        # 写入向量与 ID
        self.index.add_with_ids(embs, ids)

        # 写入元数据
        for local_i, c in enumerate(chunks):
            mid = int(ids[local_i])
            rec = {"_id": mid, "doc_id": doc_id, "text": c}
            self.meta.append(rec)
            self._meta_by_id[mid] = rec

        # 每次写入后持久化（小体量 demo 简化做法）
        self._persist()

    def search(self, query_emb: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        if self.index.ntotal == 0:
            return []

        q = query_emb.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(q)

        k = max(1, min(top_k, self.index.ntotal))
        D, I = self.index.search(q, k)  # I 是返回的向量 ID（非行号）
        scores = D[0]
        ids = I[0]

        results: List[Dict[str, Any]] = []
        for s, mid in zip(scores, ids):
            meta = self._meta_by_id.get(int(mid))
            if meta is None:
                # 理论上不应发生；为兼容性做兜底
                # 可选策略：跳过或返回最小信息
                continue
            results.append({"score": float(s), **meta})
        return results

    def reset(self) -> None:
        self.index.reset()
        self.meta.clear()
        self._meta_by_id.clear()
        self._next_id = 0

        # 清理磁盘文件
        try:
            self._idx_path.unlink(missing_ok=True)
        except Exception:
            pass
        try:
            self._meta_path.unlink(missing_ok=True)
        except Exception:
            pass

    # -------------------------
    # Persistence
    # -------------------------
    def _persist(self) -> None:
        """将 index 与 meta 落盘；使用临时文件 + 替换，降低破损风险。"""
        # 1) index
        tmp_idx = None
        try:
            with tempfile.NamedTemporaryFile(delete=False) as tf:
                tmp_idx = Path(tf.name)
            faiss.write_index(self.index, str(tmp_idx))
            tmp_idx.replace(self._idx_path)
        finally:
            if tmp_idx and tmp_idx.exists():
                # 正常情况下上面 replace 已经移动了；这里做兜底清理
                try:
                    tmp_idx.unlink()
                except Exception:
                    pass

        # 2) meta
        payload = {"next_id": self._next_id, "meta": self.meta}
        tmp_meta = None
        try:
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as tf:
                tmp_meta = Path(tf.name)
                json.dump(payload, tf, ensure_ascii=False)
            tmp_meta.replace(self._meta_path)
        finally:
            if tmp_meta and tmp_meta.exists():
                try:
                    tmp_meta.unlink()
                except Exception:
                    pass

    def _load_if_exists(self) -> None:
        """若磁盘已有 index/meta，则读入并恢复内存结构。"""
        if not (self._idx_path.exists() and self._meta_path.exists()):
            return

        # 1) index
        self.index = faiss.read_index(str(self._idx_path))

        # 2) meta + next_id
        try:
            with open(self._meta_path, "r", encoding="utf-8") as f:
                obj = json.load(f)
        except Exception:
            # meta 破损时回退为干净状态（避免卡住）
            self.meta = []
            self._meta_by_id = {}
            self._next_id = int(self.index.ntotal)  # 退化估计
            return

        self._next_id = int(obj.get("next_id", 0))
        self.meta = obj.get("meta", []) or []

        # 建立 id->meta 快速映射
        self._meta_by_id = {int(rec["_id"]): rec for rec in self.meta if "_id" in rec}

        # 若 next_id 缺失或不可信，以 max(_id)+1 为准兜底
        if self.meta:
            max_id = max(int(rec["_id"]) for rec in self.meta if "_id" in rec)
            if self._next_id <= max_id:
                self._next_id = max_id + 1
