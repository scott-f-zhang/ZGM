"""Planning logic: an ordered queue of planned work with derived start times.

The user decides *which GPUs* and *in what order*; the start time of each
planned assignment is derived by list scheduling in ``seq`` order: a task starts
as soon as every GPU it needs is free (after running work and after the planned
work queued before it on those GPUs). Overlaps therefore cannot be created by
placement; ``ConflictValidator`` remains as an invariant check.

Keyboard placement and mouse drag-and-drop both go through
``AssignmentService.assign`` / ``move`` so they share validation and persistence.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime

from ..models import Assignment, AssignmentState, Task, TaskStatus
from .task_store import TaskStore

GHOST = "__ghost__"


@dataclass
class Result:
    ok: bool
    assignment: Assignment | None = None
    conflicts: list[Assignment] = field(default_factory=list)
    error: str | None = None

    @property
    def message(self) -> str:
        if self.ok:
            return "ok"
        return self.error or "invalid placement"


@dataclass
class Slot:
    """One item in the future part of the timeline (one fixed-width column)."""

    key: str  # assignment id, or GHOST for a placement preview
    task_id: str
    gpu_ids: tuple[int, ...]
    start: datetime | None  # None: unknown (something ahead has no run history)
    end: datetime | None
    kind: str  # "running" | "planned" | "ghost"


class ConflictValidator:
    def find_conflicts(
        self,
        candidate: Assignment,
        existing: Iterable[Assignment],
        now: datetime | None = None,
    ) -> list[Assignment]:
        now = now or datetime.now()
        return [
            a
            for a in existing
            if a.id != candidate.id
            and a.state != AssignmentState.CANCELLED
            and candidate.overlaps(a, now)
        ]

    def validate_shape(
        self, task: Task, gpu_ids: Sequence[int], known_gpus: Iterable[int] | None
    ) -> str | None:
        if len(set(gpu_ids)) != len(gpu_ids):
            return "duplicate GPU ids"
        if len(gpu_ids) != task.required_gpus:
            return f"task needs {task.required_gpus} GPU(s), got {len(gpu_ids)}"
        if known_gpus is not None:
            known = set(known_gpus)
            missing = [g for g in gpu_ids if g not in known]
            if missing:
                return f"unknown GPU id(s): {missing}"
        return None


class AssignmentService:
    def __init__(
        self,
        store: TaskStore,
        gpu_ids: Iterable[int] | None,
        validator: ConflictValidator | None = None,
    ) -> None:
        self.store = store
        self.gpu_ids = sorted(gpu_ids) if gpu_ids is not None else None
        self.validator = validator or ConflictValidator()

    # queue views -----------------------------------------------------------
    def planned(self) -> list[Assignment]:
        return sorted(
            (a for a in self.store.assignments.values() if a.state == AssignmentState.PLANNED),
            key=lambda a: (a.seq, a.planned_start),
        )

    def running(self) -> list[Assignment]:
        return sorted(
            (a for a in self.store.assignments.values() if a.state == AssignmentState.RUNNING),
            key=lambda a: a.actual_start or a.planned_start,
        )

    def _duration(self, task_id: str):
        task = self.store.tasks.get(task_id)
        return task.estimated_duration if task else None

    def simulate(
        self,
        order: Sequence[tuple[str, str, tuple[int, ...], str]],
        now: datetime | None = None,
    ) -> list[Slot]:
        """List-schedule ``order`` (key, task_id, gpu_ids, kind) after running work.

        A GPU's free time becomes unknown (None) once anything on it has no
        estimate; everything queued behind it on that GPU is then unknown too.
        """
        now = (now or datetime.now()).replace(second=0, microsecond=0)
        free_at: dict[int, datetime | None] = {}
        slots: list[Slot] = []

        def ready(gpus) -> datetime | None:
            times = [free_at.get(g, now) for g in gpus]
            return None if None in times else max([now, *times])

        for a in self.running():
            end = a.expected_end(now)
            for g in a.display_gpu_ids:
                free_at[g] = None if end is None or free_at.get(g, now) is None else max(
                    free_at.get(g, now), end)
            slots.append(Slot(a.id, a.task_id, a.display_gpu_ids, a.actual_start, end, "running"))
        for key, task_id, gpus, kind in order:
            if task_id not in self.store.tasks:
                continue
            dur = self._duration(task_id)
            start = ready(gpus)
            end = start + dur if start is not None and dur is not None else None
            for g in gpus:
                free_at[g] = end
            slots.append(Slot(key, task_id, tuple(gpus), start, end, kind))
        return slots

    def schedule(self, now: datetime | None = None) -> list[Slot]:
        return self.simulate(
            [(a.id, a.task_id, a.gpu_ids, "planned") for a in self.planned()], now
        )

    def recompute(self, now: datetime | None = None, persist: bool = True) -> None:
        """Re-derive planned start/end from the queue order."""
        for slot in self.schedule(now):
            if slot.kind != "planned":
                continue
            a = self.store.assignments[slot.key]
            if (a.planned_start, a.planned_end) != (slot.start, slot.end):
                a.planned_start, a.planned_end = slot.start, slot.end
                if persist:
                    self.store.save_assignment(a, None)

    def preview(
        self,
        task_id: str,
        gpu_ids: Sequence[int],
        position: int | None,
        assignment_id: str | None = None,
        now: datetime | None = None,
    ) -> tuple[list[Slot], Slot | None]:
        """Schedule as if ``task_id`` were inserted at ``position`` (moving ``assignment_id``)."""
        base = [a for a in self.planned() if a.id != assignment_id]
        pos = len(base) if position is None else max(0, min(len(base), position))
        order = [(a.id, a.task_id, a.gpu_ids, "planned") for a in base]
        order.insert(pos, (GHOST, task_id, tuple(sorted(gpu_ids)), "ghost"))
        slots = self.simulate(order, now)
        ghost = next((s for s in slots if s.key == GHOST), None)
        return slots, ghost

    # queries ---------------------------------------------------------------
    def check(
        self,
        task_id: str,
        gpu_ids: Sequence[int],
        assignment_id: str | None = None,
    ) -> Result:
        """Validate a placement (GPU count/ids and task state) without persisting it."""
        task = self.store.tasks.get(task_id)
        if task is None:
            return Result(False, error=f"unknown task {task_id}")
        if task.status in (TaskStatus.RUNNING, TaskStatus.COMPLETED):
            return Result(False, error=f"{task.name} is {task.status.value}")
        if assignment_id:
            a = self.store.assignments.get(assignment_id)
            if a is None or a.state != AssignmentState.PLANNED:
                return Result(False, error="only planned assignments can be moved")
        err = self.validator.validate_shape(task, list(gpu_ids), self.gpu_ids)
        if err:
            return Result(False, error=err)
        return Result(True)

    def suggest_gpus(self, task: Task, anchor_gpu: int) -> tuple[int, ...]:
        """Contiguous window of ``required_gpus`` starting at the anchor (clamped)."""
        all_ids = self.gpu_ids if self.gpu_ids is not None else list(
            range(anchor_gpu + task.required_gpus))
        n = task.required_gpus
        if n >= len(all_ids):  # too many GPUs requested: check() reports it
            return tuple(all_ids)
        idx = all_ids.index(anchor_gpu) if anchor_gpu in all_ids else 0
        idx = min(idx, len(all_ids) - n)
        return tuple(all_ids[idx: idx + n])

    def _task_name(self, task_id: str) -> str:
        t = self.store.tasks.get(task_id)
        return t.name if t else task_id

    # commands --------------------------------------------------------------
    def _write_order(self, order: list[Assignment], event_for: dict[str, str]) -> None:
        for i, a in enumerate(order):
            a.seq = i
        self.recompute(persist=False)
        for a in order:
            self.store.save_assignment(a, event_for.get(a.id))

    def assign(
        self, task_id: str, gpu_ids: Sequence[int], position: int | None = None
    ) -> Result:
        """Queue ``task_id`` on ``gpu_ids`` at ``position`` (None = end of the queue)."""
        existing = self.store.active_assignment(task_id)
        if existing is not None:
            return self.move(existing.id, gpu_ids, position)
        res = self.check(task_id, gpu_ids)
        if not res.ok:
            return res
        task = self.store.tasks[task_id]
        a = Assignment(task_id=task_id, gpu_ids=tuple(gpu_ids), planned_start=None,
                       planned_end=None)  # times are derived by recompute()
        order = self.planned()
        pos = len(order) if position is None else max(0, min(len(order), position))
        order.insert(pos, a)
        self.store.assignments[a.id] = a
        self._write_order(order, {a.id: "assigned"})
        if task.status in (TaskStatus.PENDING, TaskStatus.FAILED, TaskStatus.CANCELLED):
            task.status = TaskStatus.PLANNED
            self.store.update_task(task, quiet=True)
        self.store.notify()
        return Result(True, a)

    def move(
        self,
        assignment_id: str,
        gpu_ids: Sequence[int] | None = None,
        position: int | None = None,
    ) -> Result:
        """Change GPUs and/or queue position of a planned assignment."""
        a = self.store.assignments.get(assignment_id)
        if a is None:
            return Result(False, error=f"unknown assignment {assignment_id}")
        gpus = tuple(sorted(gpu_ids)) if gpu_ids is not None else a.gpu_ids
        res = self.check(a.task_id, gpus, assignment_id=a.id)
        if not res.ok:
            return res
        order = [x for x in self.planned() if x.id != a.id]
        pos = len(order) if position is None else max(0, min(len(order), position))
        order.insert(pos, a)
        a.gpu_ids = gpus
        a.preferred_gpu_ids = gpus
        self._write_order(order, {a.id: "moved"})
        self.store.notify()
        return Result(True, a)

    def shift(self, assignment_id: str, delta: int) -> Result:
        """Move a planned assignment ``delta`` places earlier (<0) or later (>0)."""
        order = self.planned()
        idx = next((i for i, x in enumerate(order) if x.id == assignment_id), None)
        if idx is None:
            return Result(False, error="only planned assignments can be reordered")
        return self.move(assignment_id, None, max(0, idx + delta))

    def unassign(self, assignment_id: str) -> Result:
        a = self.store.assignments.get(assignment_id)
        if a is None:
            return Result(False, error=f"unknown assignment {assignment_id}")
        if a.state != AssignmentState.PLANNED:
            return Result(False, error=f"cannot unassign a {a.state.value} assignment")
        self.store.delete_assignment(assignment_id)
        self._write_order(self.planned(), {})
        task = self.store.tasks.get(a.task_id)
        if task and task.status == TaskStatus.PLANNED and not self.store.active_assignment(task.id):
            task.status = TaskStatus.PENDING
            self.store.update_task(task, quiet=True)
        self.store.notify()
        return Result(True, a)
