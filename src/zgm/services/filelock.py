"""Exclusive file locks that the OS drops when the holder dies.

Used to tell whether a process is alive without trusting PIDs (which get
reused): a task's runner holds ``<run>.lock`` while it runs, and the process
that executes the queue holds ``<db>.executor.lock``. Locks conflict across
processes and across separate opens within one process.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import IO

if os.name == "nt":
    import msvcrt

    def _try(f: IO) -> bool:
        f.seek(0)
        try:
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            return False
        return True

    def _unlock(f: IO) -> None:
        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
else:
    import fcntl

    def _try(f: IO) -> bool:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return False
        return True

    def _unlock(f: IO) -> None:
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)


class FileLock:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._file: IO | None = None

    @property
    def held(self) -> bool:
        return self._file is not None

    def acquire(self, timeout: float = 0.0) -> bool:
        """Take the lock; wait up to ``timeout`` seconds. False if someone holds it."""
        if self._file is not None:
            return True
        self.path.parent.mkdir(parents=True, exist_ok=True)
        f = open(self.path, "a+b")
        deadline = time.monotonic() + timeout
        while not _try(f):
            if time.monotonic() >= deadline:
                f.close()
                return False
            time.sleep(0.05)
        self._file = f
        return True

    def release(self) -> None:
        if self._file is None:
            return
        try:
            _unlock(self._file)
        except OSError:
            pass
        self._file.close()
        self._file = None


def is_locked(path: str | Path) -> bool:
    """Whether some process (or another open in this one) holds the lock now."""
    path = Path(path)
    if not path.exists():
        return False
    probe = FileLock(path)
    if probe.acquire():
        probe.release()
        return False
    return True
