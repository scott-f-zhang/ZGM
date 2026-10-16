"""TaskStore: in-memory state of tasks and assignments, persisted via storage."""

from __future__ import annotations

import statistics
from collections import defaultdict
from collections.abc import Callable
from datetime import datetime, timedelta

from ..models import Assignment, AssignmentState, Task, TaskStatus
from ..storage.sqlite import SqliteStorage

Listener = Callable[[], None]

# tasks that are not on the timeline (yet, or any more): these fill the top queue
WAITING = (TaskStatus.PENDING, TaskStatus.FAILED, TaskStatus.CANCELLED)


class TaskStore:
    def __init__(self, storage: SqliteStorage | None = None) -> None:
        self.storage = storage or SqliteStorage(":memory:")
        self.tasks: dict[str, Task] = {t.id: t for t in self.storage.load_tasks()}
        self.assignments: dict[str, Assignment] = {
            a.id: a for a in self.storage.load_assignments()
        }
        self._listeners: list[Listener] = []
        self._data_version = self.storage.data_version()
        self.refresh_estimates()

    def sync(self) -> bool:
        """Reload if another process (``zgm submit``) changed the database."""
        version = self.storage.data_version()
        if version == self._data_version:
            return False
        self._data_version = version
        self.tasks = {t.id: t for t in self.storage.load_tasks()}
        self.assignments = {a.id: a for a in self.storage.load_assignments()}
        self.notify()
        return True

    # estimates -------------------------------------------------------------
    def run_durations(self) -> dict[str, list[timedelta]]:
        """fingerprint -> durations of completed runs (actual start to actual end)."""
        runs: dict[str, list[timedelta]] = defaultdict(list)
        for a in self.assignments.values():
            if a.state != AssignmentState.COMPLETED or not (a.actual_start and a.actual_end):
                continue
            task = self.tasks.get(a.task_id)
            if task and task.fingerprint:
                runs[task.fingerprint].append(a.actual_end - a.actual_start)
        return runs

    def refresh_estimates(self) -> None:
        """Estimate = median of past completed runs with the same fingerprint, else None."""
        runs = self.run_durations()
        for t in self.tasks.values():
            if t.status == TaskStatus.COMPLETED:
                continue
            past = runs.get(t.fingerprint, [])
            t.estimate_runs = len(past)
            t.estimated_duration = (
                timedelta(seconds=round(statistics.median(d.total_seconds() for d in past)))
                if past else None
            )

    # observers -------------------------------------------------------------
    def subscribe(self, fn: Listener) -> None:
        self._listeners.append(fn)

    def unsubscribe(self, fn: Listener) -> None:
        if fn in self._listeners:
            self._listeners.remove(fn)

    def notify(self) -> None:
        self.refresh_estimates()
        for fn in list(self._listeners):
            fn()

    # tasks -----------------------------------------------------------------
    def add_task(self, task: Task) -> Task:
        self.tasks[task.id] = task
        self.refresh_estimates()
        self.storage.save_task(task)
        self.storage.log_event("task_created", task.id, detail={"name": task.name})
        self.notify()
        return task

    def update_task(self, task: Task, *, quiet: bool = False) -> None:
        self.tasks[task.id] = task
        self.storage.save_task(task)
        if not quiet:
            self.notify()

    def remove_task(self, task_id: str) -> None:
        self.tasks.pop(task_id, None)
        for aid in [a.id for a in self.assignments.values() if a.task_id == task_id]:
            del self.assignments[aid]
        self.storage.delete_task(task_id)
        self.storage.log_event("task_deleted", task_id)
        self.notify()

    def waiting_tasks(self) -> list[Task]:
        """Tasks for the top queue: not planned, running or finished. Oldest first."""
        return sorted((t for t in self.tasks.values() if t.status in WAITING),
                      key=lambda t: t.created_at)

    # assignments -----------------------------------------------------------
    def save_assignment(self, a: Assignment, event: str | None) -> bool:
        """Persist ``a``; ``event`` (if given) is appended to the history log.
        False if another process had already moved it on (see SqliteStorage):
        nothing is written, and the next ``sync`` brings the current state."""
        self.assignments[a.id] = a
        if not self.storage.save_assignment(a):
            self._data_version = None  # stale: reload on the next sync
            return False
        if event is None:
            return True
        self.storage.log_event(
            event,
            a.task_id,
            a.id,
            {"gpus": list(a.gpu_ids), "start": a.planned_start, "end": a.planned_end},
        )
        return True

    def update_planned(self, a: Assignment) -> None:
        """Persist a queued assignment's GPUs, order and times, if it is still queued."""
        self.assignments[a.id] = a
        if not self.storage.update_planned(a):
            self._data_version = None  # it started or went away elsewhere: reload

    def delete_assignment(self, assignment_id: str) -> Assignment | None:
        a = self.assignments.pop(assignment_id, None)
        if a:
            self.storage.delete_assignment(assignment_id)
            self.storage.log_event("unassigned", a.task_id, a.id)
        return a

    def assignments_for_task(self, task_id: str) -> list[Assignment]:
        return sorted(
            (a for a in self.assignments.values() if a.task_id == task_id),
            key=lambda a: a.planned_start,
        )

    def active_assignment(self, task_id: str) -> Assignment | None:
        for a in self.assignments_for_task(task_id):
            if a.is_active:
                return a
        return None

    def assignments_on_gpu(self, gpu_id: int) -> list[Assignment]:
        return [a for a in self.assignments.values() if gpu_id in a.display_gpu_ids]


def add_completed_run(
    store: TaskStore, template: Task, gpu_ids: tuple[int, ...],
    start: datetime, duration: timedelta, name: str | None = None,
) -> Task:
    """Record a finished run of the same workload as ``template`` (demo/tests)."""
    done = Task(name or template.name, template.command, template.required_gpus,
                status=TaskStatus.COMPLETED, fingerprint=template.fingerprint,
                config_path=template.config_path, launch=template.launch)
    done.created_at = start - timedelta(minutes=5)
    store.add_task(done)
    store.save_assignment(Assignment(
        task_id=done.id, gpu_ids=gpu_ids, planned_start=start, planned_end=start + duration,
        actual_start=start, actual_end=start + duration, actual_gpu_ids=gpu_ids,
        state=AssignmentState.COMPLETED,
    ), "completed")
    return done


def seed_demo(store: TaskStore, now: datetime | None = None) -> None:
    """Populate an empty store with example tasks and some run history.

    Estimates come only from history, so some demo tasks have one (resnet50,
    bert) and the rest show "no data".
    """
    if store.tasks:
        return
    from .scheduler import AssignmentService  # local: avoid cycle

    now = (now or datetime.now()).replace(second=0, microsecond=0)
    demo = [
        Task("resnet50-baseline", "python train.py --config configs/resnet50_base.yaml", 1),
        Task("vit-b16-pretrain", "python train.py --model vit_b16 --epochs 90", 2),
        Task("bert-finetune-sst2", "python finetune.py --task sst2", 1),
        Task("llama-lora-sweep", "python lora.py --sweep configs/lora.yaml", 4),
        Task("eval-imagenet-c", "python eval.py --suite imagenet-c", 1),
        Task("diffusion-ablation", "python ddpm.py --ablate attn", 2),
    ]
    # earlier runs of the same workloads (days ago) feed the estimates
    add_completed_run(store, demo[0], (0,), now - timedelta(days=2), timedelta(hours=2, minutes=50))
    add_completed_run(store, demo[0], (1,), now - timedelta(days=1), timedelta(hours=3, minutes=5))
    add_completed_run(store, demo[2], (2,), now - timedelta(days=1), timedelta(minutes=85))
    # one recent finished run, visible in the history zone
    prep = Task("data-preprocess", "python prep.py --shards 64", 1)
    add_completed_run(store, prep, (3,), now - timedelta(hours=3, minutes=-5),
                      timedelta(hours=1, minutes=30))

    for t in demo:
        t.created_at = now - timedelta(minutes=10)
        store.add_task(t)
    svc = AssignmentService(store, gpu_ids=None)
    svc.assign(demo[0].id, (0,))
    svc.assign(demo[1].id, (1, 2))
    svc.assign(demo[2].id, (0,))
    store.notify()
