from __future__ import annotations

import argparse
import json
import random
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
RESET_SQL = ROOT / "experiments" / "sql" / "reset.sql"

FIRST_NAMES = [
    "Alex",
    "Jamie",
    "Taylor",
    "Morgan",
    "Casey",
    "Jordan",
    "Sam",
    "Avery",
    "Riley",
    "Quinn",
    "Logan",
    "Eden",
    "Skyler",
    "Robin",
    "Harper",
]

LAST_NAMES = [
    "Chen",
    "Shen",
    "Zhao",
    "Li",
    "Garcia",
    "Patel",
    "Khan",
    "Ito",
    "Suzuki",
    "Taylor",
    "Walker",
    "Singh",
    "Kim",
]

TEAMS = ["Platform", "Intelligence", "Delivery", "Product", "Ops", "Quality", "Infra"]
ROLES = [
    "Backend Engineer",
    "Data Scientist",
    "Product Manager",
    "QA Lead",
    "Solutions Architect",
    "Program Manager",
    "Research Engineer",
]

PROJECT_CODES = ["E3D", "ORBIT", "NOVA", "LYNX", "APEX", "ATLAS", "PULSE", "LUMEN", "CRUX"]
PROJECT_THEMES = ["Integration", "Automation", "Insight", "Compliance", "Acceleration"]

ITEM_TYPES = ["TASK", "BUG", "REQUEST"]
TAGS = [
    "integration",
    "api",
    "reporting",
    "ops",
    "handoff",
    "retro",
    "incident",
    "migration",
    "planning",
]

TAG_ALIASES = {
    "integration": ["system integration", "interop"],
    "api": ["interface", "endpoint"],
    "reporting": ["dashboard", "metrics report"],
    "ops": ["operations", "runbook"],
    "handoff": ["transition", "ownership change"],
    "retro": ["retrospective", "post-mortem"],
    "incident": ["outage", "sev incident"],
    "migration": ["migrate", "porting"],
    "planning": ["roadmap", "sprint planning"],
}

STATUS_PIPELINES = {
    "DONE": ["TODO", "IN_PROGRESS", "DONE"],
    "IN_PROGRESS": ["TODO", "IN_PROGRESS"],
    "BLOCKED": ["TODO", "IN_PROGRESS", "BLOCKED"],
    "TODO": ["TODO"],
    "CANCELED": ["TODO", "CANCELED"],
}

FINAL_STATUS_WEIGHTS = {
    "DONE": 0.46,
    "IN_PROGRESS": 0.25,
    "BLOCKED": 0.12,
    "TODO": 0.1,
    "CANCELED": 0.07,
}


def parse_date(date_str: str) -> datetime:
    try:
        dt = datetime.fromisoformat(date_str)
    except ValueError as exc:
        raise ValueError(f"Invalid date string '{date_str}'. Use YYYY-MM-DD.") from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt


def epoch_ms(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def random_datetime(rng: random.Random, start: datetime, end: datetime) -> datetime:
    total_seconds = (end - start).total_seconds()
    if total_seconds <= 0:
        return start
    offset = rng.random() * total_seconds
    return start + timedelta(seconds=offset)


def slugify(value: str) -> str:
    return value.lower().replace(" ", "_")


@dataclass
class SeedOptions:
    db_path: Path
    summary_path: Path
    items: int
    people: Optional[int]
    projects: Optional[int]
    random_seed: int
    start_date: datetime
    end_date: datetime


class SeedGenerator:
    def __init__(self, options: SeedOptions):
        self.opts = options
        self.rng = random.Random(options.random_seed)
        self.conn: Optional[sqlite3.Connection] = None

    # --------------------------
    # Entry point
    # --------------------------

    def run(self) -> Dict[str, object]:
        db_dir = self.opts.db_path.parent
        db_dir.mkdir(parents=True, exist_ok=True)
        summary_dir = self.opts.summary_path.parent
        summary_dir.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(str(self.opts.db_path))
        conn.execute("PRAGMA foreign_keys = ON")
        self.conn = conn
        try:
            self._reset_schema()
            people = self._insert_people()
            projects = self._insert_projects(people)
            items = self._insert_items(people, projects)
            status_summary = self._insert_statuses(items, people)
            alias_count = self._insert_aliases(people, projects)
            conn.commit()
        finally:
            conn.close()
            self.conn = None

        summary = self._build_summary(people, projects, items, status_summary, alias_count)
        self.opts.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    # --------------------------
    # Schema helpers
    # --------------------------

    def _reset_schema(self) -> None:
        sql = RESET_SQL.read_text(encoding="utf-8")
        assert self.conn is not None
        self.conn.executescript(sql)

    # --------------------------
    # Generators
    # --------------------------

    def _insert_people(self) -> List[Dict[str, object]]:
        target = self.opts.people or max(6, self.opts.items // 10)
        rows: List[Dict[str, object]] = []
        used_handles: set[str] = set()

        for idx in range(target):
            first = self.rng.choice(FIRST_NAMES)
            last = self.rng.choice(LAST_NAMES)
            handle = slugify(f"{first}.{last}")
            handle_candidate = handle
            suffix = 1
            while handle_candidate in used_handles:
                suffix += 1
                handle_candidate = f"{handle}{suffix}"
            handle = handle_candidate
            used_handles.add(handle)

            display_name = f"{first} {last}"
            team = self.rng.choice(TEAMS)
            role = self.rng.choice(ROLES)
            created_ts = epoch_ms(
                random_datetime(
                    self.rng,
                    self.opts.start_date - timedelta(days=30),
                    self.opts.start_date,
                )
            )
            payload = (handle, display_name, team, role, created_ts)
            assert self.conn is not None
            cur = self.conn.execute(
                """
                INSERT INTO people(handle, display_name, team, role, created_ts)
                VALUES(?, ?, ?, ?, ?)
                """,
                payload,
            )
            rows.append(
                {
                    "person_id": cur.lastrowid,
                    "handle": handle,
                    "display_name": display_name,
                    "team": team,
                    "role": role,
                }
            )
        return rows

    def _insert_projects(self, people: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
        target = self.opts.projects or max(3, self.opts.items // 25)
        rows: List[Dict[str, object]] = []
        used_codes: set[str] = set()
        for idx in range(target):
            code = self.rng.choice(PROJECT_CODES)
            while code in used_codes:
                code = f"{code}{idx+1}"
            used_codes.add(code)

            theme = self.rng.choice(PROJECT_THEMES)
            name = f"{theme} Initiative {code}"
            owner = self.rng.choice(people)
            created_ts = epoch_ms(
                random_datetime(
                    self.rng,
                    self.opts.start_date - timedelta(days=45),
                    self.opts.end_date,
                )
            )
            status = self.rng.choice(["ACTIVE", "PAUSED", "CLOSED"])
            payload = (code, name, owner["person_id"], status, created_ts)
            assert self.conn is not None
            cur = self.conn.execute(
                """
                INSERT INTO projects(code, name, owner_person_id, status, created_ts)
                VALUES(?, ?, ?, ?, ?)
                """,
                payload,
            )
            rows.append(
                {
                    "project_id": cur.lastrowid,
                    "code": code,
                    "name": name,
                    "status": status,
                }
            )
        return rows

    def _insert_items(
        self,
        people: Sequence[Dict[str, object]],
        projects: Sequence[Dict[str, object]],
    ) -> List[Dict[str, object]]:
        rows: List[Dict[str, object]] = []
        assert self.conn is not None
        for idx in range(self.opts.items):
            owner = self.rng.choice(people)
            project = self.rng.choice(projects) if projects and self.rng.random() > 0.2 else None

            created_dt = random_datetime(self.rng, self.opts.start_date, self.opts.end_date)
            due_dt = created_dt + timedelta(days=self.rng.randint(3, 18))
            due_dt = min(due_dt, self.opts.end_date + timedelta(days=5))

            title = self._generate_title()
            tags = self._generate_tags()
            description = f"Auto-generated item focusing on {tags.split(',')[0]} scenarios."
            item_type = self.rng.choice(ITEM_TYPES)
            priority = self.rng.randint(1, 4)

            payload = (
                title,
                item_type,
                owner["person_id"],
                project["project_id"] if project else None,
                priority,
                tags,
                epoch_ms(created_dt),
                epoch_ms(due_dt),
                description,
                0,
            )

            cur = self.conn.execute(
                """
                INSERT INTO items(
                  title, item_type, owner_person_id, project_id,
                  priority, tags, created_ts, due_ts, description, is_deleted
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                payload,
            )
            rows.append(
                {
                    "item_id": cur.lastrowid,
                    "created_dt": created_dt,
                    "owner": owner,
                    "project": project,
                }
            )
        return rows

    def _insert_statuses(
        self,
        items: Sequence[Dict[str, object]],
        people: Sequence[Dict[str, object]],
    ) -> Dict[str, int]:
        status_counter: Dict[str, int] = {key: 0 for key in STATUS_PIPELINES}
        assert self.conn is not None
        for item in items:
            final_status = self._pick_final_status()
            status_counter[final_status] += 1
            pipeline = STATUS_PIPELINES[final_status]
            current_time = item["created_dt"]
            for stage in pipeline:
                delta_days = self.rng.uniform(0.25, 4.0)
                current_time += timedelta(days=delta_days)
                note = self._status_note(stage)
                changed_by = item["owner"] if self.rng.random() < 0.7 else self.rng.choice(people)
                payload = (
                    item["item_id"],
                    stage,
                    note,
                    epoch_ms(current_time),
                    changed_by["person_id"],
                )
                self.conn.execute(
                    """
                    INSERT INTO item_status_history(item_id, status, note, changed_ts, changed_by_person_id)
                    VALUES(?, ?, ?, ?, ?)
                    """,
                    payload,
                )
        return status_counter

    def _insert_aliases(
        self,
        people: Sequence[Dict[str, object]],
        projects: Sequence[Dict[str, object]],
    ) -> int:
        assert self.conn is not None
        count = 0
        # People aliases
        for person in people:
            first, last = person["display_name"].split(" ")
            candidates = {
                person["display_name"],
                first,
                last,
                f"{first} {last[0]}",
                person["handle"].replace(".", " "),
            }
            for alias in sorted(candidates):
                if alias.lower() == person["handle"]:
                    continue
                cur = self.conn.execute(
                    """
                    INSERT OR IGNORE INTO entity_aliases(entity_type, entity_ref, alias)
                    VALUES(?, ?, ?)
                    """,
                    ("PERSON", person["handle"], alias),
                )
                count += cur.rowcount

        # Project aliases
        for project in projects:
            aliases = {
                project["code"],
                project["code"].lower(),
                project["name"],
                f"Project {project['code']}",
            }
            for alias in sorted(aliases):
                cur = self.conn.execute(
                    """
                    INSERT OR IGNORE INTO entity_aliases(entity_type, entity_ref, alias)
                    VALUES(?, ?, ?)
                    """,
                    ("PROJECT", project["code"], alias),
                )
                count += cur.rowcount

        # Tag aliases
        for tag, alias_list in TAG_ALIASES.items():
            for alias in alias_list:
                cur = self.conn.execute(
                    """
                    INSERT OR IGNORE INTO entity_aliases(entity_type, entity_ref, alias)
                    VALUES(?, ?, ?)
                    """,
                    ("TAG", tag, alias),
                )
                count += cur.rowcount
        return count

    # --------------------------
    # Helpers
    # --------------------------

    def _generate_title(self) -> str:
        verbs = [
            "Refine",
            "Audit",
            "Validate",
            "Align",
            "Prototype",
            "Review",
            "Harden",
            "Ship",
            "Automate",
            "Trace",
        ]
        nouns = [
            "schema",
            "pipeline",
            "dashboard",
            "handoff plan",
            "API adapter",
            "report",
            "playbook",
            "migration",
            "checkpoint",
        ]
        qualifiers = ["phase", "sprint", "beta", "release", "support"]
        return f"{self.rng.choice(verbs)} {self.rng.choice(nouns)} {self.rng.choice(qualifiers)}"

    def _generate_tags(self) -> str:
        tag_count = self.rng.randint(1, min(3, len(TAGS)))
        return ",".join(sorted(self.rng.sample(TAGS, tag_count)))

    def _status_note(self, status: str) -> Optional[str]:
        mapping = {
            "TODO": "Planned in backlog",
            "IN_PROGRESS": "Work picked up by owner",
            "BLOCKED": "Waiting for dependency",
            "DONE": "Validated and closed",
            "CANCELED": "Dropped after triage",
        }
        return mapping.get(status)

    def _pick_final_status(self) -> str:
        statuses = list(FINAL_STATUS_WEIGHTS.keys())
        weights = [FINAL_STATUS_WEIGHTS[s] for s in statuses]
        total = sum(weights)
        threshold = self.rng.random() * total
        acc = 0.0
        for status, weight in zip(statuses, weights):
            acc += weight
            if threshold <= acc:
                return status
        return statuses[-1]

    def _build_summary(
        self,
        people: Sequence[Dict[str, object]],
        projects: Sequence[Dict[str, object]],
        items: Sequence[Dict[str, object]],
        status_counter: Dict[str, int],
        alias_count: int,
    ) -> Dict[str, object]:
        created_dates = sorted(item["created_dt"] for item in items)
        month_buckets: Dict[str, int] = {}
        for dt in created_dates:
            key = dt.strftime("%Y-%m")
            month_buckets[key] = month_buckets.get(key, 0) + 1

        return {
            "items": len(items),
            "people": len(people),
            "projects": len(projects),
            "aliases": alias_count,
            "status_distribution": status_counter,
            "date_distribution": {
                "range_start": created_dates[0].isoformat() if created_dates else None,
                "range_end": created_dates[-1].isoformat() if created_dates else None,
                "by_month": month_buckets,
            },
            "seed": self.opts.random_seed,
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deterministic experiment data generator")
    parser.add_argument("--db", required=True, help="SQLite DB path to populate.")
    parser.add_argument("--summary", required=True, help="Summary JSON output path.")
    parser.add_argument("--items", type=int, default=200, help="Number of items/tasks to generate.")
    parser.add_argument("--people", type=int, help="Number of people to generate.")
    parser.add_argument("--projects", type=int, help="Number of projects to generate.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--start-date", dest="start_date", default="2024-09-01", help="Earliest creation date (YYYY-MM-DD).")
    parser.add_argument("--end-date", dest="end_date", default="2024-11-30", help="Latest creation date (YYYY-MM-DD).")
    return parser


def cli_main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    start = parse_date(args.start_date)
    end = parse_date(args.end_date)
    if end <= start:
        raise SystemExit("end-date must be greater than start-date")

    options = SeedOptions(
        db_path=Path(args.db),
        summary_path=Path(args.summary),
        items=args.items,
        people=args.people,
        projects=args.projects,
        random_seed=args.seed,
        start_date=start,
        end_date=end,
    )
    generator = SeedGenerator(options)
    summary = generator.run()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    cli_main()
