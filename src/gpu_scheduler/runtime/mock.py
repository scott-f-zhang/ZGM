"""MockRuntime: pretends to run tasks; finishes after a scaled-down estimate."""

from __future__ import annotations

from datetime import datetime, timedelta

from ..models import Assignment, Task
from .base import RunHandle, RunState


class MockRuntime:
    name = "mock"

    def __init__(self, speedup: float = 3600.0) -> None:
        # speedup=3600 -> a 1h estimate "runs" for 1s of wall time
        self.speedup = speedup
        self._runs: dict[str, dict] = {}

    def available(self) -> bool:
        return True

    def submit(self, task: Task, assignment: Assignment) -> RunHandle:
        self._runs[assignment.id] = {
            "started": datetime.now(),
            # no history: pretend it takes an hour (mock only)
            "seconds": (task.estimated_duration or timedelta(hours=1)).total_seconds()
            / self.speedup,
            "cancelled": False,
        }
        assignment.actual_start = datetime.now()
        assignment.actual_gpu_ids = assignment.preferred_gpu_ids
        return RunHandle(assignment.id, assignment.id)

    def cancel(self, handle: RunHandle) -> None:
        if handle.assignment_id in self._runs:
            self._runs[handle.assignment_id]["cancelled"] = True

    def status(self, handle: RunHandle) -> RunState:
        run = self._runs.get(handle.assignment_id)
        if run is None:
            return RunState.PENDING
        if run["cancelled"]:
            return RunState.CANCELLED
        elapsed = (datetime.now() - run["started"]).total_seconds()
        return RunState.SUCCEEDED if elapsed >= run["seconds"] else RunState.RUNNING
