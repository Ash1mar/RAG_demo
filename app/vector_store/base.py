from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np


class VectorStore(ABC):
    @abstractmethod
    def add_texts(
        self,
        doc_id: str,
        chunks: List[str],
        embeddings: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        ...

    @abstractmethod
    def search(
        self,
        query_emb: np.ndarray,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        ...

    @abstractmethod
    def reset(self) -> None:
        ...
