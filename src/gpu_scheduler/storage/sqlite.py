"""SQLite persistence for tasks, assignments and an append-only history log."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

from ..models import Assignment, AssignmentState, Task, TaskStatus

SCHEMA_VERSION = 2

SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    command TEXT NOT NULL,
    required_gpus INTEGER NOT NULL,
    estimated_seconds REAL NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    notes TEXT NOT NULL DEFAULT '',
    config_path TEXT
);
CREATE TABLE IF NOT EXISTS assignments (
    id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    gpu_ids TEXT NOT NULL,
    preferred_gpu_ids TEXT NOT NULL,
    actual_gpu_ids TEXT,
    planned_start TEXT NOT NULL,
    planned_end TEXT NOT NULL,
    actual_start TEXT,
    actual_end TEXT,
    state TEXT NOT NULL,
    seq INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS history (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    at TEXT NOT NULL,
    event TEXT NOT NULL,
    task_id TEXT,
    assignment_id TEXT,
    detail TEXT
);
"""


def default_db_path() -> Path:
    return Path.home() / ".local" / "share" / "gpu-scheduler" / "state.db"


def _dt(s: str | None) -> datetime | None:
    return datetime.fromisoformat(s) if s else None


def _iso(d: datetime | None) -> str | None:
    return d.isoformat() if d else None


class SqliteStorage:
    def __init__(self, path: str | Path = ":memory:") -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = str(path)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self._migrate()
        self.conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        self.conn.commit()

    def _migrate(self) -> None:
        cols = {r["name"] for r in self.conn.execute("PRAGMA table_info(assignments)")}
        if "seq" not in cols:  # v1 -> v2: planned work becomes an ordered queue
            self.conn.execute("ALTER TABLE assignments ADD COLUMN seq INTEGER NOT NULL DEFAULT 0")
            rows = self.conn.execute(
                "SELECT id FROM assignments WHERE state='planned' ORDER BY planned_start"
            ).fetchall()
            for i, r in enumerate(rows):
                self.conn.execute("UPDATE assignments SET seq=? WHERE id=?", (i, r["id"]))

    def close(self) -> None:
        self.conn.close()

    # tasks -----------------------------------------------------------------
    def save_task(self, t: Task) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO tasks VALUES (?,?,?,?,?,?,?,?,?)",
            (
                t.id,
                t.name,
                t.command,
                t.required_gpus,
                t.estimated_duration.total_seconds(),
                t.status.value,
                t.created_at.isoformat(),
                t.notes,
                t.config_path,
            ),
        )
        self.conn.commit()

    def delete_task(self, task_id: str) -> None:
        self.conn.execute("DELETE FROM tasks WHERE id=?", (task_id,))
        self.conn.execute("DELETE FROM assignments WHERE task_id=?", (task_id,))
        self.conn.commit()

    def load_tasks(self) -> list[Task]:
        rows = self.conn.execute("SELECT * FROM tasks ORDER BY created_at, rowid").fetchall()
        return [
            Task(
                id=r["id"],
                name=r["name"],
                command=r["command"],
                required_gpus=r["required_gpus"],
                estimated_duration=timedelta(seconds=r["estimated_seconds"]),
                status=TaskStatus(r["status"]),
                created_at=datetime.fromisoformat(r["created_at"]),
                notes=r["notes"],
                config_path=r["config_path"],
            )
            for r in rows
        ]

    # assignments -----------------------------------------------------------
    def save_assignment(self, a: Assignment) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO assignments (id, task_id, gpu_ids, preferred_gpu_ids,"
            " actual_gpu_ids, planned_start, planned_end, actual_start, actual_end, state, seq)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                a.id,
                a.task_id,
                json.dumps(list(a.gpu_ids)),
                json.dumps(list(a.preferred_gpu_ids)),
                json.dumps(list(a.actual_gpu_ids)) if a.actual_gpu_ids is not None else None,
                _iso(a.planned_start),
                _iso(a.planned_end),
                _iso(a.actual_start),
                _iso(a.actual_end),
                a.state.value,
                a.seq,
            ),
        )
        self.conn.commit()

    def delete_assignment(self, assignment_id: str) -> None:
        self.conn.execute("DELETE FROM assignments WHERE id=?", (assignment_id,))
        self.conn.commit()

    def load_assignments(self) -> list[Assignment]:
        rows = self.conn.execute("SELECT * FROM assignments ORDER BY seq, planned_start").fetchall()
        out = []
        for r in rows:
            actual = r["actual_gpu_ids"]
            out.append(
                Assignment(
                    id=r["id"],
                    task_id=r["task_id"],
                    gpu_ids=tuple(json.loads(r["gpu_ids"])),
                    preferred_gpu_ids=tuple(json.loads(r["preferred_gpu_ids"])),
                    actual_gpu_ids=tuple(json.loads(actual)) if actual else None,
                    planned_start=datetime.fromisoformat(r["planned_start"]),
                    planned_end=datetime.fromisoformat(r["planned_end"]),
                    actual_start=_dt(r["actual_start"]),
                    actual_end=_dt(r["actual_end"]),
                    state=AssignmentState(r["state"]),
                    seq=r["seq"],
                )
            )
        return out

    # history ---------------------------------------------------------------
    def log_event(
        self,
        event: str,
        task_id: str | None = None,
        assignment_id: str | None = None,
        detail: dict | None = None,
    ) -> None:
        self.conn.execute(
            "INSERT INTO history (at, event, task_id, assignment_id, detail) VALUES (?,?,?,?,?)",
            (
                datetime.now().isoformat(timespec="seconds"),
                event,
                task_id,
                assignment_id,
                json.dumps(detail, default=str) if detail else None,
            ),
        )
        self.conn.commit()

    def history(self, task_id: str | None = None) -> list[dict]:
        if task_id:
            rows = self.conn.execute(
                "SELECT * FROM history WHERE task_id=? ORDER BY seq", (task_id,)
            ).fetchall()
        else:
            rows = self.conn.execute("SELECT * FROM history ORDER BY seq").fetchall()
        return [dict(r) for r in rows]
