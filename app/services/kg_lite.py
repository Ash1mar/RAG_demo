from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol


@dataclass
class PersonEntry:
    canonical: str
    aliases: List[str]


@dataclass
class ProjectEntry:
    canonical: str
    aliases: List[str]


@dataclass
class CategoryEntry:
    name: str
    aliases: List[str]
    tags: List[str]


@dataclass
class StatusEntry:
    canonical: str
    aliases: List[str]


@dataclass
class PriorityEntry:
    canonical: int
    aliases: List[str]


@dataclass
class KGData:
    persons: List[PersonEntry]
    projects: List[ProjectEntry]
    categories: List[CategoryEntry]
    statuses: List[StatusEntry]
    priorities: List[PriorityEntry]


class KGBackend(Protocol):
    """Abstract backend interface for KG-lite lookups."""

    def find_person(self, name: str) -> Optional[PersonEntry]:
        ...

    def find_project(self, project: Optional[str], text: Optional[str]) -> Optional[ProjectEntry]:
        ...

    def find_category_tags(self, text: str) -> List[str]:
        ...

    def find_status(self, value: str) -> Optional[StatusEntry]:
        ...

    def find_priority(self, value: Any) -> Optional[PriorityEntry]:
        ...

    def snapshot(self) -> Dict[str, List[str]]:
        ...


DEFAULT_KG_RAW: Dict[str, Any] = {
    "persons": [
        {"canonical": "张三", "aliases": ["张工", "老张"]},
        {"canonical": "李四", "aliases": ["李工"]},
    ],
    "projects": [
        {"canonical": "芯片", "aliases": ["芯片项目", "芯片平台"]},
        {"canonical": "交付", "aliases": ["交付项目", "交付项目组"]},
        {"canonical": "E3D", "aliases": ["E3D项目", "E3D系统"]},
    ],
    "categories": [
        {
            "name": "安监整改",
            "aliases": ["整改任务", "安监专项", "安全专项"],
            "tags": ["整改", "安全整改"],
        }
    ],
    "statuses": [
        {"canonical": "DONE", "aliases": ["完成", "搞定", "done", "已完成", "完结"]},
        {"canonical": "TODO", "aliases": ["未完成", "待办", "todo", "还没做"]},
        {"canonical": "IN_PROGRESS", "aliases": ["进行中", "在做", "in progress", "跟进中"]},
        {"canonical": "BLOCKED", "aliases": ["阻塞", "卡住", "blocked"]},
    ],
    "priorities": [
        {"canonical": 1, "aliases": ["P1", "p1", "高优", "最高优先级"]},
        {"canonical": 2, "aliases": ["P2", "p2", "中优"]},
        {"canonical": 3, "aliases": ["P3", "p3", "低优"]},
    ],
}


def _load_raw_kg(path: Path) -> Dict[str, Any]:
    if path.exists():
        try:
            text = path.read_text(encoding="utf-8")
            data = json.loads(text)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return DEFAULT_KG_RAW


def _build_entries(raw_list: List[Dict[str, Any]], *, key: str) -> List[str]:
    entries: List[str] = []
    for item in raw_list:
        value = str(item.get(key) or "").strip()
        if value:
            entries.append(value)
    return entries


def _build_kg_data(raw: Dict[str, Any]) -> KGData:
    persons_raw = raw.get("persons") or []
    projects_raw = raw.get("projects") or []
    categories_raw = raw.get("categories") or []
    statuses_raw = raw.get("statuses") or []
    priorities_raw = raw.get("priorities") or []

    persons = [
        PersonEntry(
            canonical=str(item.get("canonical") or "").strip(),
            aliases=[alias.strip() for alias in item.get("aliases") or [] if alias],
        )
        for item in persons_raw
        if item.get("canonical")
    ]
    projects = [
        ProjectEntry(
            canonical=str(item.get("canonical") or "").strip(),
            aliases=[alias.strip() for alias in item.get("aliases") or [] if alias],
        )
        for item in projects_raw
        if item.get("canonical")
    ]
    categories = [
        CategoryEntry(
            name=str(item.get("name") or "").strip(),
            aliases=[alias.strip() for alias in item.get("aliases") or [] if alias],
            tags=[tag.strip() for tag in item.get("tags") or [] if tag],
        )
        for item in categories_raw
        if item.get("name")
    ]
    statuses = [
        StatusEntry(
            canonical=str(item.get("canonical") or "").strip().upper(),
            aliases=[alias.strip() for alias in item.get("aliases") or [] if alias],
        )
        for item in statuses_raw
        if item.get("canonical")
    ]
    priorities: List[PriorityEntry] = []
    for item in priorities_raw:
        canonical = item.get("canonical")
        if canonical is None:
            continue
        try:
            canonical_int = int(canonical)
        except (TypeError, ValueError):
            continue
        aliases = [str(alias).strip() for alias in item.get("aliases") or [] if str(alias).strip()]
        priorities.append(PriorityEntry(canonical=canonical_int, aliases=aliases))

    return KGData(
        persons=persons,
        projects=projects,
        categories=categories,
        statuses=statuses,
        priorities=priorities,
    )


class InMemoryKGBackend:
    def __init__(self, data: KGData):
        self._data = data

    def find_person(self, name: str) -> Optional[PersonEntry]:
        token = (name or "").strip()
        if not token:
            return None
        for entry in self._data.persons:
            if token == entry.canonical or token in entry.aliases:
                return entry
        return None

    def find_project(self, project: Optional[str], text: Optional[str]) -> Optional[ProjectEntry]:
        token = (project or "").strip()
        context = (text or "").strip()
        if not token and not context:
            return None
        for entry in self._data.projects:
            if token and (token == entry.canonical or token in entry.aliases):
                return entry
            if context and (
                entry.canonical in context or any(alias in context for alias in entry.aliases)
            ):
                return entry
        return None

    def find_category_tags(self, text: str) -> List[str]:
        token = (text or "").strip()
        if not token:
            return []
        tags: List[str] = []
        for entry in self._data.categories:
            if (
                entry.name in token
                or token in entry.name
                or any(alias in token or token in alias for alias in entry.aliases)
            ):
                for tag in entry.tags:
                    if tag and tag not in tags:
                        tags.append(tag)
        return tags

    def find_status(self, value: str) -> Optional[StatusEntry]:
        token = (value or "").strip()
        if not token:
            return None
        upper = token.upper()
        for entry in self._data.statuses:
            if upper == entry.canonical.upper():
                return entry
            if any(upper == alias.upper() for alias in entry.aliases):
                return entry
        return None

    def find_priority(self, value: Any) -> Optional[PriorityEntry]:
        if value in (None, ""):
            return None
        token = str(value).strip()
        if not token:
            return None
        try:
            numeric = int(token)
        except ValueError:
            numeric = None
        for entry in self._data.priorities:
            if numeric is not None and entry.canonical == numeric:
                return entry
            if any(token.lower() == str(alias).lower() for alias in entry.aliases):
                return entry
        return None

    def snapshot(self) -> Dict[str, List[str]]:
        return {
            "persons": [p.canonical for p in self._data.persons],
            "projects": [p.canonical for p in self._data.projects],
            "categories": [c.name for c in self._data.categories],
            "statuses": [s.canonical for s in self._data.statuses],
            "priorities": [str(p.canonical) for p in self._data.priorities],
        }


class KGResolver:
    def __init__(self, backend: KGBackend):
        self._backend = backend

    def resolve_person(self, name: Optional[str]) -> Optional[str]:
        if not name:
            return None
        entry = self._backend.find_person(name)
        if entry is None:
            token = name.strip()
            return token or None
        return entry.canonical

    def resolve_project(self, project: Optional[str], text: Optional[str] = None) -> Optional[str]:
        entry = self._backend.find_project(project, text)
        if entry is None:
            token = (project or "").strip()
            return token or None
        return entry.canonical

    def resolve_category_tags(self, text: Optional[str]) -> List[str]:
        if not text:
            return []
        return self._backend.find_category_tags(text)

    def resolve_status(self, value: Optional[str]) -> Optional[str]:
        if not value:
            return None
        entry = self._backend.find_status(value)
        if entry is None:
            token = value.strip()
            return token or None
        return entry.canonical

    def resolve_priority(self, value: Any) -> Optional[int]:
        entry = self._backend.find_priority(value)
        if entry is None:
            try:
                return int(str(value).strip())
            except (TypeError, ValueError):
                return None
        return entry.canonical

    def debug_snapshot(self) -> Dict[str, List[str]]:
        return self._backend.snapshot()


KG_DATA_PATH = Path(os.getenv("KG_DATA_PATH", str(Path("data") / "kg_data.json")))
KG_BACKEND = InMemoryKGBackend(_build_kg_data(_load_raw_kg(KG_DATA_PATH)))
KG_RESOLVER = KGResolver(KG_BACKEND)


def resolve_person(name: Optional[str]) -> Optional[str]:
    return KG_RESOLVER.resolve_person(name)


def resolve_project(project: Optional[str], text: Optional[str] = None) -> Optional[str]:
    return KG_RESOLVER.resolve_project(project, text)


def resolve_category_tags(category_or_text: Optional[str]) -> List[str]:
    return KG_RESOLVER.resolve_category_tags(category_or_text)


def resolve_status_value(value: Optional[str]) -> Optional[str]:
    return KG_RESOLVER.resolve_status(value)


def resolve_priority_value(value: Any) -> Optional[int]:
    return KG_RESOLVER.resolve_priority(value)


def get_debug_snapshot() -> Dict[str, List[str]]:
    return KG_RESOLVER.debug_snapshot()
