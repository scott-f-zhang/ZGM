"""Assignment: a task's place on the GPU timeline.

Planned times come from the user (start) plus the task's estimate. Actual times
are only ever written by a runtime when execution really starts/ends, so the
estimate is never mistaken for a completion time.

``preferred_gpu_ids`` is what the user asked for. ``actual_gpu_ids`` is what the
runtime really got: Ray treats GPUs as logical resources and picks devices
itself, so the two may differ once execution is wired up.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from .task import new_id


class AssignmentState(str, Enum):
    PLANNED = "planned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


ACTIVE_STATES = {AssignmentState.PLANNED, AssignmentState.RUNNING}


@dataclass
class Assignment:
    task_id: str
    gpu_ids: tuple[int, ...]
    planned_start: datetime
    planned_end: datetime
    id: str = field(default_factory=new_id)
    actual_start: datetime | None = None
    actual_end: datetime | None = None
    preferred_gpu_ids: tuple[int, ...] = ()
    actual_gpu_ids: tuple[int, ...] | None = None
    state: AssignmentState = AssignmentState.PLANNED

    def __post_init__(self) -> None:
        self.gpu_ids = tuple(sorted(self.gpu_ids))
        if not self.preferred_gpu_ids:
            self.preferred_gpu_ids = self.gpu_ids
        if self.planned_end <= self.planned_start:
            raise ValueError("planned_end must be after planned_start")

    @property
    def is_active(self) -> bool:
        return self.state in ACTIVE_STATES

    @property
    def is_historical(self) -> bool:
        return self.actual_end is not None

    @property
    def display_gpu_ids(self) -> tuple[int, ...]:
        return tuple(self.actual_gpu_ids) if self.actual_gpu_ids else self.gpu_ids

    def display_interval(self, now: datetime) -> tuple[datetime, datetime]:
        """Interval to draw: real times when known, otherwise the plan."""
        if self.actual_start is not None:
            if self.actual_end is not None:
                return self.actual_start, self.actual_end
            expected = self.actual_start + (self.planned_end - self.planned_start)
            return self.actual_start, max(now, expected)
        return self.planned_start, self.planned_end

    def overlaps(self, other: "Assignment", now: datetime | None = None) -> bool:
        now = now or datetime.now()
        if not set(self.gpu_ids) & set(other.gpu_ids):
            return False
        a0, a1 = self.display_interval(now)
        b0, b1 = other.display_interval(now)
        return a0 < b1 and b0 < a1
