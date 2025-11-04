from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from datetime import datetime, timezone


def epoch_ms(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def main() -> None:
    data_dir = Path(os.getenv("DATA_DIR", "data"))
    data_dir.mkdir(parents=True, exist_ok=True)
    db_path = data_dir / "tasks.db"

    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person TEXT NOT NULL,
                task TEXT NOT NULL,
                status TEXT NOT NULL,
                ts INTEGER NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_person_task_ts ON tasks(person, task, ts DESC)")

        # Clear existing sample rows for idempotency (only sample set by marker)
        # Use a simple marker in comments by task+person combination if needed. For now, upsert-like reset.
        conn.execute("DELETE FROM tasks")

        # Sample data covering: multiple people, multiple tasks, DONE/TODO, duplicate records (latest wins)
        rows = [
            # 张三｜提交9月周报：first TODO, then DONE (latest)
            ("张三", "提交9月周报", "TODO", epoch_ms(datetime(2024, 9, 28, 9, 0, 0, tzinfo=timezone.utc))),
            ("张三", "提交9月周报", "DONE", epoch_ms(datetime(2024, 9, 30, 18, 0, 0, tzinfo=timezone.utc))),
            # 张三｜E3D接口联调：current TODO
            ("张三", "E3D接口联调", "TODO", epoch_ms(datetime(2024, 10, 8, 12, 0, 0, tzinfo=timezone.utc))),
            # 李四｜整理工艺包V2：DONE
            ("李四", "整理工艺包V2", "DONE", epoch_ms(datetime(2024, 9, 25, 10, 30, 0, tzinfo=timezone.utc))),
        ]

        conn.executemany(
            "INSERT INTO tasks(person, task, status, ts) VALUES(?, ?, ?, ?)",
            rows,
        )
        conn.commit()
        print(f"Initialized sample tasks DB at: {db_path}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()

