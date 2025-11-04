from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class TasksStore(ABC):
    """Abstract task store interface.

    Keeps a minimal, stable API so upper layers don't depend on
    the specific backend (SQLite, KG, etc.).
    """

    @abstractmethod
    def ready(self) -> bool:
        """Returns True if the store is reachable and usable."""
        ...

    @abstractmethod
    def get_latest_status(self, person: str, task: str) -> Optional[Dict[str, Any]]:
        """Return the latest record for a given person+task.

        Expected record fields:
        - person: str
        - task: str
        - status: str  (e.g., "DONE" | "TODO")
        - ts: int      (epoch millis)
        - id: int      (monotonic row id)
        """
        ...

    @abstractmethod
    def list_persons(self) -> List[str]:
        """Return distinct person names available in the backend."""
        ...

    @abstractmethod
    def list_tasks(self) -> List[str]:
        """Return distinct task names available in the backend."""
        ...

    def search(self, *, person: Optional[str] = None, task: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Optional: list recent matching records (backend may override for efficiency)."""
        # Default naive implementation using get_latest_status only when both provided.
        if person and task:
            rec = self.get_latest_status(person, task)
            return [rec] if rec else []
        return []
