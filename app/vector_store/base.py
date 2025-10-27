from typing import List, Dict, Any
import numpy as np
from abc import ABC, abstractmethod

class VectorStore(ABC):
    @abstractmethod
    def add_texts(self, doc_id: str, chunks: List[str], embeddings: np.ndarray) -> None: ...
    @abstractmethod
    def search(self, query_emb: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]: ...
    @abstractmethod
    def reset(self) -> None: ...
