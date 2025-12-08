from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


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


# Minimal in-memory KG-lite store.
_PERSONS: List[PersonEntry] = [
    PersonEntry(canonical="张三", aliases=["张工", "老张"]),
    PersonEntry(canonical="李四", aliases=["李工"]),
]

_CATEGORIES: List[CategoryEntry] = [
    CategoryEntry(
        name="安监整改",
        aliases=["整改任务", "安监专项", "安全专项"],
        tags=["整改", "安全整改"],
    ),
]

_PROJECTS: List[ProjectEntry] = [
    ProjectEntry(canonical="芯片", aliases=["芯片项目", "芯片平台"]),
    ProjectEntry(canonical="交付", aliases=["交付项目", "交付项目组"]),
    ProjectEntry(canonical="E3D", aliases=["E3D项目", "E3D系统"]),
]


def resolve_person(name: Optional[str]) -> Optional[str]:
    """Resolve a person name or alias to its canonical form.

    Returns the canonical name if found; otherwise returns the original name.
    """
    if not name:
        return None
    token = name.strip()
    if not token:
        return None

    for entry in _PERSONS:
        if token == entry.canonical or token in entry.aliases:
            return entry.canonical
    return token


def resolve_category_tags(category: Optional[str]) -> List[str]:
    """Resolve a logical category name or text fragment into a list of tags."""
    if not category:
        return []
    token = category.strip()
    if not token:
        return []

    for entry in _CATEGORIES:
        text = token
        if (
            entry.name in text
            or text in entry.name
            or any(alias in text or text in alias for alias in entry.aliases)
        ):
            return list(entry.tags)
    return []


def resolve_project(project: Optional[str], text: Optional[str] = None) -> Optional[str]:
    """Resolve project name or alias (optionally from full text) to canonical form."""
    token = (project or "").strip()
    context = (text or "").strip()

    if not token and not context:
        return None

    for entry in _PROJECTS:
        if token and (token == entry.canonical or token in entry.aliases):
            return entry.canonical
        if context and (
            entry.canonical in context
            or any(alias in context for alias in entry.aliases)
        ):
            return entry.canonical
    return token or None


def get_debug_snapshot() -> Dict[str, List[str]]:
    """Return a tiny snapshot of KG-lite for debug / inspection."""
    persons = [p.canonical for p in _PERSONS]
    categories = [c.name for c in _CATEGORIES]
    projects = [p.canonical for p in _PROJECTS]
    return {
        "persons": persons,
        "categories": categories,
        "projects": projects,
    }
