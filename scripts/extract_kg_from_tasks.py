"""
Skeleton script for extracting KG-lite seed data from the tasks table.

Planned usage (future):
  - SELECT DISTINCT person/project/tags/status FROM tasks or task_latest;
  - Normalize / clean values;
  - Write a JSON/YAML file compatible with data/kg_data.json for KG-lite.

For now this script only prints distinct values from the demo SQLite DB to
illustrate the intended ETL flow.
"""
from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Set


def _connect_tasks_db() -> sqlite3.Connection:
    db_path = os.getenv("TASKS_DB", "data/tasks.db")
    return sqlite3.connect(db_path)


def _distinct_values(conn: sqlite3.Connection, column: str, table: str = "tasks") -> List[str]:
    cur = conn.cursor()
    try:
        cur.execute(f"SELECT DISTINCT {column} FROM {table}")
    except sqlite3.Error:
        return []
    rows = cur.fetchall()
    return [str(row[0]) for row in rows if row and row[0] is not None]


def preview_kg_candidates() -> Dict[str, Any]:
    """Return a KG-lite candidate structure (canonical values only)."""
    conn = _connect_tasks_db()
    try:
        persons = sorted({value for value in _distinct_values(conn, "person") if value})
        projects = sorted({value for value in _distinct_values(conn, "project") if value})
        statuses = sorted({value.upper() for value in _distinct_values(conn, "status") if value})
        priority_values = sorted(
            {value for value in _distinct_values(conn, "priority") if value not in (None, "")}
        )
        raw_tags = _distinct_values(conn, "tags")
    finally:
        conn.close()

    tags: Set[str] = set()
    for value in raw_tags:
        if not value:
            continue
        for part in str(value).split(","):
            token = part.strip()
            if token:
                tags.add(token)

    categories = [
        {
            "name": tag,
            "aliases": [],
            "tags": [tag],
        }
        for tag in sorted(tags)
    ]

    def _wrap_entries(values: List[str]) -> List[Dict[str, Any]]:
        return [{"canonical": value, "aliases": []} for value in values]

    def _wrap_priorities(values: List[str]) -> List[Dict[str, Any]]:
        entries: List[Dict[str, Any]] = []
        for value in values:
            try:
                canonical = int(value)
            except ValueError:
                continue
            entries.append({"canonical": canonical, "aliases": []})
        return entries

    candidates = {
        "persons": _wrap_entries(persons),
        "projects": _wrap_entries(projects),
        "categories": categories,
        "statuses": [{"canonical": value, "aliases": []} for value in statuses],
        "priorities": _wrap_priorities(priority_values),
    }
    return candidates


def main() -> None:
    preview = preview_kg_candidates()
    print("# KG-lite candidate data extracted from tasks DB")
    print(json.dumps(preview, ensure_ascii=False, indent=2))
    print(
        "\n# NOTE:\n"
        "# - This script only emits canonical values with empty alias lists.\n"
        "# - Review and merge the output into data/kg_data.json manually (or via a dedicated script).\n"
        "# - Categories are generated from existing tags as one-to-one drafts; adjust/merge as needed."
    )


if __name__ == "__main__":
    main()
