"""Runtime interface: how a planned assignment would actually be executed.

The TUI and scheduler depend only on this protocol, never on Ray directly, so
planning stays ours and execution can be swapped (mock, Ray, ...).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from ..models import Assignment, Task


class RunState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class RunHandle:
    assignment_id: str
    ref: object  # runtime-specific handle


class Runtime(Protocol):
    name: str

    def available(self) -> bool: ...

    def submit(self, task: Task, assignment: Assignment) -> RunHandle: ...

    def cancel(self, handle: RunHandle) -> None: ...

    def status(self, handle: RunHandle) -> RunState: ...
