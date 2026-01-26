from __future__ import annotations

import importlib
from os import getenv
from typing import Dict, Optional, Type

from app.tasks_domain.base import TaskDomain
from app.tasks_domain.tasks import TasksDomain

_REGISTRY: Dict[str, TaskDomain] = {
    "tasks": TasksDomain(),
}


def _load_domain(name: str) -> Optional[TaskDomain]:
    module_name = name
    if "." not in module_name:
        module_name = f"app.tasks_domain.{module_name}"
    try:
        mod = importlib.import_module(module_name)
    except ModuleNotFoundError:
        return None

    domain = getattr(mod, "DOMAIN", None)
    if isinstance(domain, TaskDomain):
        return domain

    get_domain = getattr(mod, "get_domain", None)
    if callable(get_domain):
        domain = get_domain()
        if isinstance(domain, TaskDomain):
            return domain

    candidates: list[Type[TaskDomain]] = []
    for _, value in vars(mod).items():
        if isinstance(value, type) and issubclass(value, TaskDomain) and value is not TaskDomain:
            candidates.append(value)

    for cls in candidates:
        instance = cls()
        if (getattr(instance, "name", "") or "").strip().lower() == name:
            return instance

    if candidates:
        return candidates[0]()

    return None


def get_tasks_domain() -> TaskDomain:
    name = (getenv("TASKS_DOMAIN", "tasks") or "tasks").strip().lower()
    if name in _REGISTRY:
        return _REGISTRY[name]
    loaded = _load_domain(name)
    if loaded is not None:
        _REGISTRY[name] = loaded
        return loaded
    return _REGISTRY["tasks"]
