from __future__ import annotations

from os import getenv
from typing import Dict

from app.tasks_domain.base import TaskDomain
from app.tasks_domain.tasks import TasksDomain

_REGISTRY: Dict[str, TaskDomain] = {
    "tasks": TasksDomain(),
}


def get_tasks_domain() -> TaskDomain:
    name = (getenv("TASKS_DOMAIN", "tasks") or "tasks").strip().lower()
    return _REGISTRY.get(name, _REGISTRY["tasks"])
