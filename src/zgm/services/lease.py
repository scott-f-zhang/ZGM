"""ExecutorLease: at most one process runs the queue for a database.

The executor — ``zgm daemon``, or a TUI with ``--runtime`` when no daemon is
running — holds an OS lock on ``<db>.executor.lock`` for as long as it lives,
and describes itself in ``<db>.executor.json``. If it exits or dies, the OS
drops the lock and another zgm can take over; tasks keep running meanwhile
(see LocalRuntime) and the new executor attaches to them.
"""

from __future__ import annotations

import json
import os
import socket
from datetime import datetime
from pathlib import Path

from .filelock import FileLock, is_locked


class ExecutorLease:
    def __init__(self, db_path: str | Path) -> None:
        self.in_memory = str(db_path) == ":memory:"
        db = Path(db_path).expanduser() if not self.in_memory else Path("memory")
        self.lock_path = db.with_name(db.name + ".executor.lock")
        self.info_path = db.with_name(db.name + ".executor.json")
        self._lock = FileLock(self.lock_path)
        self._info: dict | None = None
        self._memory_held = False

    @property
    def held(self) -> bool:
        return self._memory_held or self._lock.held

    def acquire(self, role: str, runtime: str, log_dir: str | None = None,
                timeout: float = 0.0) -> bool:
        """Become the executor. False if another process already is."""
        info = {"role": role, "pid": os.getpid(), "host": socket.gethostname(),
                "runtime": runtime, "log_dir": log_dir,
                "since": datetime.now().isoformat(timespec="seconds")}
        if self.in_memory:  # a private in-memory database has no other processes
            self._memory_held, self._info = True, info
            return True
        if not self._lock.acquire(timeout):
            return False
        self._info = info
        tmp = self.info_path.with_name(self.info_path.name + ".tmp")
        tmp.write_text(json.dumps(info), encoding="utf-8")
        os.replace(tmp, self.info_path)
        return True

    def release(self) -> None:
        if self._lock.held:
            self.info_path.unlink(missing_ok=True)
            self._lock.release()
        self._memory_held = False
        self._info = None

    def holder(self) -> dict | None:
        """Who runs the queue now (this process included), or None."""
        if self.held:
            return self._info
        if self.in_memory or not is_locked(self.lock_path):
            return None
        try:
            return json.loads(self.info_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"role": "unknown"}


def describe(holder: dict | None) -> str:
    if not holder:
        return "nobody runs the queue"
    who = "zgm daemon" if holder.get("role") == "daemon" else "a zgm TUI"
    return f"{who} (pid {holder.get('pid')}, {holder.get('runtime')} runtime)"
