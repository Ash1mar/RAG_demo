from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol


@dataclass
class PersonEntry:
    canonical: str
    aliases: List[str]


@dataclass
class CategoryEntry:
    name: str
    aliases: List[str]
    tags: List[str]


@dataclass
class ProjectEntry:
    canonical: str
    aliases: List[str]


@dataclass
class KGData:
    persons: List[PersonEntry]
    projects: List[ProjectEntry]
    categories: List[CategoryEntry]


class KGBackend(Protocol):
    """Abstract backend for KG-lite lookups.

    This allows future backends (e.g., Neo4j, other graph DBs) to implement
    the same interface without changing resolver logic.
    """

    def find_person(self, name: str) -> Optional[PersonEntry]:
        ...

    def find_project(self, project: Optional[str], text: Optional[str]) -> Optional[ProjectEntry]:
        ...

    def find_category_tags(self, text: str) -> List[str]:
        ...

    def snapshot(self) -> Dict[str, List[str]]:
        ...


# Default in-code KG data; kept minimal and schema-oriented.
_DEFAULT_KG_RAW: Dict[str, Any] = {
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
}


def _load_raw_kg(path: Path) -> Dict[str, Any]:
    """Load KG data from JSON file if present; otherwise fall back to defaults."""
    if path.exists():
        try:
            text = path.read_text(encoding="utf-8")
            data = json.loads(text)
            if isinstance(data, dict):
                return data
        except Exception:
            # Fallback to in-code defaults on any error.
            pass
    return _DEFAULT_KG_RAW


def _build_kg_data(raw: Dict[str, Any]) -> KGData:
    persons_raw = raw.get("persons") or []
    projects_raw = raw.get("projects") or []
    categories_raw = raw.get("categories") or []

    persons = [
        PersonEntry(
            canonical=str(item.get("canonical") or "").strip(),
            aliases=list(item.get("aliases") or []),
        )
        for item in persons_raw
        if item.get("canonical")
    ]
    projects = [
        ProjectEntry(
            canonical=str(item.get("canonical") or "").strip(),
            aliases=list(item.get("aliases") or []),
        )
        for item in projects_raw
        if item.get("canonical")
    ]
    categories = [
        CategoryEntry(
            name=str(item.get("name") or "").strip(),
            aliases=list(item.get("aliases") or []),
            tags=list(item.get("tags") or []),
        )
        for item in categories_raw
        if item.get("name")
    ]
    return KGData(persons=persons, projects=projects, categories=categories)


class InMemoryKGBackend:
    """Simple in-memory backend loading KG data from JSON or defaults."""

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

    def snapshot(self) -> Dict[str, List[str]]:
        return {
            "persons": [p.canonical for p in self._data.persons],
            "projects": [p.canonical for p in self._data.projects],
            "categories": [c.name for c in self._data.categories],
        }


class KGResolver:
    """Facade used by the rest of the codebase for KG lookups."""

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

    def debug_snapshot(self) -> Dict[str, List[str]]:
        return self._backend.snapshot()


_KG_DATA_PATH = Path("data") / "kg_data.json"
_KG_BACKEND = InMemoryKGBackend(_build_kg_data(_load_raw_kg(_KG_DATA_PATH)))
_KG_RESOLVER = KGResolver(_KG_BACKEND)


def resolve_person(name: Optional[str]) -> Optional[str]:
    """Module-level helper for resolving person names via KG-lite."""
    return _KG_RESOLVER.resolve_person(name)


def resolve_category_tags(category_or_text: Optional[str]) -> List[str]:
    """Module-level helper for resolving category / text into tags via KG-lite."""
    return _KG_RESOLVER.resolve_category_tags(category_or_text)


def resolve_project(project: Optional[str], text: Optional[str] = None) -> Optional[str]:
    """Module-level helper for resolving project names via KG-lite."""
    return _KG_RESOLVER.resolve_project(project, text)


def get_debug_snapshot() -> Dict[str, List[str]]:
    """Return a tiny snapshot of KG-lite for debug / inspection."""
    return _KG_RESOLVER.debug_snapshot()

