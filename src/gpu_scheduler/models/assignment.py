"""Assignment: a task's place on the GPU timeline.

Planned work is an ordered queue: ``seq`` is the user's chosen order and the
planned start/end are *derived* from it (a task starts when all of its GPUs are
free). Actual times are only ever written by a runtime when execution really
starts/ends, so the estimate is never mistaken for a completion time.

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
    planned_start: datetime | None  # None: unknown (no run history to estimate from)
    planned_end: datetime | None
    id: str = field(default_factory=new_id)
    actual_start: datetime | None = None
    actual_end: datetime | None = None
    preferred_gpu_ids: tuple[int, ...] = ()
    actual_gpu_ids: tuple[int, ...] | None = None
    state: AssignmentState = AssignmentState.PLANNED
    seq: int = 0  # position in the planned queue (global across GPUs)

    def __post_init__(self) -> None:
        self.gpu_ids = tuple(sorted(self.gpu_ids))
        if not self.preferred_gpu_ids:
            self.preferred_gpu_ids = self.gpu_ids
        if (self.planned_start is not None and self.planned_end is not None
                and self.planned_end <= self.planned_start):
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

    @property
    def expected_duration(self):
        if self.planned_start is None or self.planned_end is None:
            return None
        return self.planned_end - self.planned_start

    def expected_end(self, now: datetime) -> datetime | None:
        """When running/planned work should finish, or None if unknown."""
        if self.actual_end is not None:
            return self.actual_end
        if self.actual_start is not None:
            dur = self.expected_duration
            return max(now, self.actual_start + dur) if dur is not None else None
        return self.planned_end

    def display_interval(self, now: datetime) -> tuple[datetime | None, datetime | None]:
        """Interval to draw: real times when known, otherwise the plan.

        Running work with no estimate is drawn up to ``now``.
        """
        if self.actual_start is not None:
            if self.actual_end is not None:
                return self.actual_start, self.actual_end
            return self.actual_start, self.expected_end(now) or now
        return self.planned_start, self.planned_end

    def overlaps(self, other: "Assignment", now: datetime | None = None) -> bool:
        """True if both intervals are known and share time on a common GPU."""
        now = now or datetime.now()
        if not set(self.gpu_ids) & set(other.gpu_ids):
            return False
        a0, a1 = self.display_interval(now)
        b0, b1 = other.display_interval(now)
        if None in (a0, a1, b0, b1):
            return False
        return a0 < b1 and b0 < a1
