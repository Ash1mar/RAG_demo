from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Tuple


def epoch_ms(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _exec_many(conn: sqlite3.Connection, sql: str, rows: Iterable[Tuple]) -> None:
    conn.executemany(sql, list(rows))


def main() -> None:
    """
    (Re)initialize a richer demo tasks DB with:
    - persons table (normalized people info).
    - tasks table (denormalized per-status rows; latest status derived by ts DESC).
    - task_latest view (one row per person+task, picking latest ts).
    - sample data covering TODO→DONE transitions, priorities, projects, tags, due dates.
    """
    data_dir = Path(os.getenv("DATA_DIR", "data"))
    data_dir.mkdir(parents=True, exist_ok=True)
    db_path = data_dir / "tasks.db"

    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("PRAGMA foreign_keys = ON")

        # Reset schema for idempotent re-run when columns change
        conn.execute("DROP VIEW IF EXISTS task_latest")
        conn.execute("DROP TABLE IF EXISTS tasks")
        conn.execute("DROP TABLE IF EXISTS persons")

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS persons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                display_name TEXT,
                team TEXT,
                role TEXT,
                created_ts INTEGER NOT NULL
            )
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id INTEGER NOT NULL,
                person TEXT NOT NULL,           -- denormalized for quick lookups
                owner TEXT,                     -- optional "owner"/requester for enterprise datasets
                org_name TEXT,
                division_name TEXT,
                post_name TEXT,
                is_read INTEGER,
                is_delegated INTEGER,
                task TEXT NOT NULL,
                project TEXT,
                tags TEXT,                      -- comma-separated for demo
                priority INTEGER NOT NULL DEFAULT 3,  -- 1=highest
                status TEXT NOT NULL,           -- TODO | IN_PROGRESS | BLOCKED | DONE
                status_note TEXT,
                description TEXT,
                created_ts INTEGER NOT NULL,
                due_ts INTEGER,
                ts INTEGER NOT NULL,            -- status change timestamp (epoch ms)
                updated_ts INTEGER NOT NULL,    -- alias of ts for clarity
                FOREIGN KEY(person_id) REFERENCES persons(id)
            )
            """
        )

        conn.execute(
            """
            CREATE VIEW IF NOT EXISTS task_latest AS
            SELECT t.*
            FROM tasks t
            JOIN (
                SELECT person, task, MAX(ts) AS max_ts
                FROM tasks
                GROUP BY person, task
            ) latest
            ON latest.person = t.person AND latest.task = t.task AND latest.max_ts = t.ts
            """
        )

        conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_person_task_ts ON tasks(person, task, ts DESC)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_project ON tasks(project)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_priority ON tasks(priority)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_due_ts ON tasks(due_ts)")

        # Clear existing sample rows for idempotency
        conn.execute("DELETE FROM tasks")
        conn.execute("DELETE FROM persons")

        now = datetime.now(timezone.utc)
        persons = [
            ("张三", "张三", "研发一部", "后端", epoch_ms(now)),
            ("李四", "李四", "研发一部", "测试", epoch_ms(now)),
            ("王五", "王五", "产品部", "产品", epoch_ms(now)),
        ]
        _exec_many(
            conn,
            "INSERT INTO persons(name, display_name, team, role, created_ts) VALUES(?, ?, ?, ?, ?)",
            persons,
        )

        # Helper to map person name -> id
        cur = conn.cursor()
        cur.execute("SELECT id, name FROM persons")
        pid_map = {name: pid for pid, name in cur.fetchall()}

        def row(
            name: str,
            task: str,
            status: str,
            ts: datetime,
            *,
            project: str = None,
            tags: str = None,
            priority: int = 3,
            due: datetime = None,
            note: str = None,
            desc: str = None,
        ) -> Tuple:
            pid = pid_map[name]
            ts_ms = epoch_ms(ts)
            return (
                pid,
                name,
                name,  # owner mirrors assignee for demo data
                None,  # org_name
                None,  # division_name
                None,  # post_name
                0,     # is_read
                0,     # is_delegated
                task,
                project,
                tags,
                priority,
                status,
                note,
                desc,
                epoch_ms(ts.replace(hour=9, minute=0, second=0, microsecond=0)),  # created_ts
                epoch_ms(due) if due else None,
                ts_ms,
                ts_ms,  # updated_ts mirrors ts
            )

        rows = [
            # 张三｜提交9月周报：TODO -> DONE
            row(
                "张三",
                "提交9月周报",
                "TODO",
                datetime(2024, 9, 28, 9, 0, tzinfo=timezone.utc),
                project="交付",
                tags="周报,交付",
                priority=2,
                due=datetime(2024, 9, 30, 10, 0, tzinfo=timezone.utc),
            ),
            row(
                "张三",
                "提交9月周报",
                "DONE",
                datetime(2024, 9, 30, 18, 0, tzinfo=timezone.utc),
                project="交付",
                tags="周报,交付",
                priority=2,
                due=datetime(2024, 9, 30, 10, 0, tzinfo=timezone.utc),
                note="邮件已发",
            ),
            # 张三｜E3D接口联调：TODO -> IN_PROGRESS
            row(
                "张三",
                "E3D接口联调",
                "TODO",
                datetime(2024, 10, 1, 10, 0, tzinfo=timezone.utc),
                project="E3D",
                tags="接口,联调",
                priority=1,
                desc="与前端联调接口",
            ),
            row(
                "张三",
                "E3D接口联调",
                "IN_PROGRESS",
                datetime(2024, 10, 8, 12, 0, tzinfo=timezone.utc),
                project="E3D",
                tags="接口,联调",
                priority=1,
                note="已打通鉴权",
            ),
            # 李四｜整理工艺包V2：TODO -> DONE
            row(
                "李四",
                "整理工艺包V2",
                "TODO",
                datetime(2024, 9, 20, 9, 30, tzinfo=timezone.utc),
                project="工艺",
                tags="整理,文档",
                priority=3,
                desc="收集V2文档",
            ),
            row(
                "李四",
                "整理工艺包V2",
                "DONE",
                datetime(2024, 9, 25, 10, 30, tzinfo=timezone.utc),
                project="工艺",
                tags="整理,文档",
                priority=3,
            ),
            # 王五｜需求澄清：TODO -> BLOCKED -> DONE
            row(
                "王五",
                "需求澄清",
                "TODO",
                datetime(2024, 10, 5, 8, 0, tzinfo=timezone.utc),
                project="RAG优化",
                tags="需求,沟通",
                priority=2,
            ),
            row(
                "王五",
                "需求澄清",
                "BLOCKED",
                datetime(2024, 10, 6, 14, 0, tzinfo=timezone.utc),
                project="RAG优化",
                tags="需求,沟通",
                priority=2,
                note="等待客户反馈",
            ),
            row(
                "王五",
                "需求澄清",
                "DONE",
                datetime(2024, 10, 9, 17, 30, tzinfo=timezone.utc),
                project="RAG优化",
                tags="需求,沟通",
                priority=2,
                note="需求已确认",
            ),
        ]

        _exec_many(
            conn,
            """
            INSERT INTO tasks(
                person_id, person, owner, org_name, division_name, post_name, is_read, is_delegated,
                task, project, tags, priority, status, status_note,
                description, created_ts, due_ts, ts, updated_ts
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

        conn.commit()
        print(f"Initialized sample tasks DB at: {db_path}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
