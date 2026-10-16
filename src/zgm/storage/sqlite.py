"""SQLite persistence for tasks, assignments and an append-only history log."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

from ..models import Assignment, AssignmentState, LaunchSpec, Task, TaskStatus

SCHEMA_VERSION = 5

# planned times and estimates are nullable: no run history means "unknown"
TABLES = {
    "tasks": """CREATE TABLE tasks (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    command TEXT NOT NULL,
    required_gpus INTEGER NOT NULL,
    estimated_seconds REAL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    notes TEXT NOT NULL DEFAULT '',
    config_path TEXT,
    fingerprint TEXT NOT NULL DEFAULT '',
    launch TEXT
)""",
    "assignments": """CREATE TABLE assignments (
    id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    gpu_ids TEXT NOT NULL,
    preferred_gpu_ids TEXT NOT NULL,
    actual_gpu_ids TEXT,
    planned_start TEXT,
    planned_end TEXT,
    actual_start TEXT,
    actual_end TEXT,
    state TEXT NOT NULL,
    seq INTEGER NOT NULL DEFAULT 0,
    log_path TEXT,
    run_now INTEGER NOT NULL DEFAULT 0
)""",
    "history": """CREATE TABLE history (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    at TEXT NOT NULL,
    event TEXT NOT NULL,
    task_id TEXT,
    assignment_id TEXT,
    detail TEXT
)""",
    # shared by every zgm on this database (TUI, daemon, submit)
    "settings": """CREATE TABLE settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
)""",
}


def default_db_path() -> Path:
    return Path.home() / ".local" / "share" / "zgm" / "state.db"


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
        self._migrate()
        self.conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        self.conn.commit()

    def _columns(self, table: str) -> dict[str, bool]:
        """column name -> NOT NULL flag (empty if the table doesn't exist)."""
        return {r["name"]: bool(r["notnull"])
                for r in self.conn.execute(f"PRAGMA table_info({table})")}

    def _migrate(self) -> None:
        """Create missing tables; rebuild any whose columns differ from TABLES."""
        old_assignment_cols = self._columns("assignments")
        for table, ddl in TABLES.items():
            old = self._columns(table)
            if not old:
                self.conn.execute(ddl)
                continue
            self.conn.execute(ddl.replace(f"CREATE TABLE {table}", "CREATE TEMP TABLE _target"))
            target = self._columns("_target")
            self.conn.execute("DROP TABLE _target")
            if target == old:
                continue
            common = ", ".join(c for c in target if c in old)
            self.conn.execute(f"ALTER TABLE {table} RENAME TO {table}_old")
            self.conn.execute(ddl)
            self.conn.execute(f"INSERT INTO {table} ({common}) SELECT {common} FROM {table}_old")
            self.conn.execute(f"DROP TABLE {table}_old")
        if old_assignment_cols and "seq" not in old_assignment_cols:
            # v1 -> v2: planned work becomes an ordered queue, in planned-start order
            rows = self.conn.execute(
                "SELECT id FROM assignments WHERE state='planned' ORDER BY planned_start"
            ).fetchall()
            for i, r in enumerate(rows):
                self.conn.execute("UPDATE assignments SET seq=? WHERE id=?", (i, r["id"]))
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def data_version(self) -> int:
        """Changes when another connection (e.g. ``zgm submit``) commits."""
        return self.conn.execute("PRAGMA data_version").fetchone()[0]

    # tasks -----------------------------------------------------------------
    def save_task(self, t: Task) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO tasks (id, name, command, required_gpus, estimated_seconds,"
            " status, created_at, notes, config_path, fingerprint, launch)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                t.id,
                t.name,
                t.command,
                t.required_gpus,
                t.estimated_duration.total_seconds() if t.estimated_duration else None,
                t.status.value,
                t.created_at.isoformat(),
                t.notes,
                t.config_path,
                t.fingerprint,
                json.dumps(t.launch.to_dict()) if t.launch else None,
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
                estimated_duration=(timedelta(seconds=r["estimated_seconds"])
                                    if r["estimated_seconds"] else None),
                status=TaskStatus(r["status"]),
                created_at=datetime.fromisoformat(r["created_at"]),
                notes=r["notes"],
                config_path=r["config_path"],
                fingerprint=r["fingerprint"],
                launch=LaunchSpec.from_dict(json.loads(r["launch"])) if r["launch"] else None,
            )
            for r in rows
        ]

    # assignments -----------------------------------------------------------
    def save_assignment(self, a: Assignment) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO assignments (id, task_id, gpu_ids, preferred_gpu_ids,"
            " actual_gpu_ids, planned_start, planned_end, actual_start, actual_end, state, seq,"
            " log_path, run_now) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
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
                a.log_path,
                int(a.run_now),
            ),
        )
        self.conn.commit()

    def assignment_state(self, assignment_id: str) -> AssignmentState | None:
        """The stored state, which another process may have changed."""
        row = self.conn.execute("SELECT state FROM assignments WHERE id=?",
                                (assignment_id,)).fetchone()
        return AssignmentState(row["state"]) if row else None

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
                    planned_start=_dt(r["planned_start"]),
                    planned_end=_dt(r["planned_end"]),
                    actual_start=_dt(r["actual_start"]),
                    actual_end=_dt(r["actual_end"]),
                    state=AssignmentState(r["state"]),
                    seq=r["seq"],
                    log_path=r["log_path"],
                    run_now=bool(r["run_now"]),
                )
            )
        return out

    # settings --------------------------------------------------------------
    def get_setting(self, key: str, default: str | None = None) -> str | None:
        row = self.conn.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        return row["value"] if row else default

    def set_setting(self, key: str, value: str) -> None:
        self.conn.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?,?)",
                          (key, value))
        self.conn.commit()

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
