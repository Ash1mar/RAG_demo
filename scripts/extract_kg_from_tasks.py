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
from typing import Dict, List, Set


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


def preview_kg_candidates() -> Dict[str, List[str]]:
    """Return a preview of candidate KG values from the demo DB."""
    conn = _connect_tasks_db()
    try:
        persons = _distinct_values(conn, "person")
        projects = _distinct_values(conn, "project")
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

    return {
        "persons": sorted(set(persons)),
        "projects": sorted(set(p for p in projects if p)),
        "tags": sorted(tags),
    }


def main() -> None:
    preview = preview_kg_candidates()
    print("# KG-lite candidate values extracted from tasks DB")
    print(json.dumps(preview, ensure_ascii=False, indent=2))
    print(
        "\n# TODO: map these canonical values and aliases into data/kg_data.json\n"
        "# so that KG-lite can use real data instead of demo entries."
    )


if __name__ == "__main__":
    main()

