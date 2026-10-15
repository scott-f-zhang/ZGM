"""Planning logic: conflict validation and the single AssignmentService.

Both keyboard placement and mouse drag-and-drop end up in
``AssignmentService.assign``/``move`` so they share validation and persistence.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from ..models import Assignment, AssignmentState, Task, TaskStatus
from .task_store import TaskStore


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
        return self.error or "conflict"


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

    # queries ---------------------------------------------------------------
    def check(
        self,
        task_id: str,
        gpu_ids: Sequence[int],
        start: datetime,
        assignment_id: str | None = None,
    ) -> Result:
        """Validate a placement without persisting it (used for previews)."""
        task = self.store.tasks.get(task_id)
        if task is None:
            return Result(False, error=f"unknown task {task_id}")
        err = self.validator.validate_shape(task, list(gpu_ids), self.gpu_ids)
        if err:
            return Result(False, error=err)
        cand = Assignment(
            task_id=task_id,
            gpu_ids=tuple(gpu_ids),
            planned_start=start,
            planned_end=start + task.estimated_duration,
        )
        if assignment_id:
            cand.id = assignment_id
        conflicts = self.validator.find_conflicts(cand, self.store.assignments.values())
        if conflicts:
            names = ", ".join(self._task_name(c.task_id) for c in conflicts)
            return Result(False, cand, conflicts, f"conflicts with {names}")
        return Result(True, cand)

    def suggest_gpus(self, task: Task, anchor_gpu: int, start: datetime) -> tuple[int, ...]:
        """anchor + following GPUs; prefers ones free at ``start``."""
        all_ids = self.gpu_ids if self.gpu_ids is not None else list(range(anchor_gpu + task.required_gpus))
        n = task.required_gpus
        if n > len(all_ids):
            return tuple(all_ids)
        # contiguous window starting at anchor (wrap to fit)
        idx = all_ids.index(anchor_gpu) if anchor_gpu in all_ids else 0
        idx = min(idx, len(all_ids) - n)
        window = tuple(all_ids[idx: idx + n])
        if self.check(task.id, window, start).ok:
            return window
        # otherwise anchor + any free others
        end = start + task.estimated_duration
        free = [g for g in all_ids if g != anchor_gpu and self._gpu_free(g, start, end)]
        if anchor_gpu in all_ids and len(free) >= n - 1:
            return tuple(sorted([anchor_gpu, *free[: n - 1]]))
        return window

    def next_free_slot(
        self, gpu_ids: Sequence[int], after: datetime, duration: timedelta
    ) -> datetime:
        t = after
        now = datetime.now()
        for _ in range(1000):
            blockers = [
                a
                for a in self.store.assignments.values()
                if a.state != AssignmentState.CANCELLED
                and set(a.display_gpu_ids) & set(gpu_ids)
                and a.display_interval(now)[0] < t + duration
                and t < a.display_interval(now)[1]
            ]
            if not blockers:
                return t
            t = max(b.display_interval(now)[1] for b in blockers)
        return t

    def _gpu_free(self, gpu: int, start: datetime, end: datetime) -> bool:
        now = datetime.now()
        for a in self.store.assignments_on_gpu(gpu):
            if a.state == AssignmentState.CANCELLED:
                continue
            s, e = a.display_interval(now)
            if s < end and start < e:
                return False
        return True

    def _task_name(self, task_id: str) -> str:
        t = self.store.tasks.get(task_id)
        return t.name if t else task_id

    # commands --------------------------------------------------------------
    def assign(self, task_id: str, gpu_ids: Sequence[int], start: datetime) -> Result:
        existing = self.store.active_assignment(task_id)
        if existing is not None:
            return self.move(existing.id, gpu_ids, start)
        res = self.check(task_id, gpu_ids, start)
        if not res.ok:
            return res
        a = res.assignment
        assert a is not None
        self.store.save_assignment(a, "assigned")
        task = self.store.tasks[task_id]
        if task.status == TaskStatus.PENDING:
            task.status = TaskStatus.PLANNED
            self.store.update_task(task, quiet=True)
        self.store.notify()
        return res

    def move(self, assignment_id: str, gpu_ids: Sequence[int], start: datetime) -> Result:
        a = self.store.assignments.get(assignment_id)
        if a is None:
            return Result(False, error=f"unknown assignment {assignment_id}")
        if a.state != AssignmentState.PLANNED:
            return Result(False, error=f"cannot move a {a.state.value} assignment")
        res = self.check(a.task_id, gpu_ids, start, assignment_id=a.id)
        if not res.ok:
            return res
        a.gpu_ids = tuple(sorted(gpu_ids))
        a.preferred_gpu_ids = a.gpu_ids
        a.planned_start = start
        a.planned_end = res.assignment.planned_end  # type: ignore[union-attr]
        self.store.save_assignment(a, "moved")
        self.store.notify()
        return Result(True, a)

    def unassign(self, assignment_id: str) -> Result:
        a = self.store.assignments.get(assignment_id)
        if a is None:
            return Result(False, error=f"unknown assignment {assignment_id}")
        if a.state != AssignmentState.PLANNED:
            return Result(False, error=f"cannot unassign a {a.state.value} assignment")
        self.store.delete_assignment(assignment_id)
        task = self.store.tasks.get(a.task_id)
        if task and task.status == TaskStatus.PLANNED and not self.store.active_assignment(task.id):
            task.status = TaskStatus.PENDING
            self.store.update_task(task, quiet=True)
        self.store.notify()
        return Result(True, a)
