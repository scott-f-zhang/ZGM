"""TaskStore: in-memory state of tasks and assignments, persisted via storage."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta

from ..models import Assignment, Task, TaskStatus
from ..storage.sqlite import SqliteStorage

Listener = Callable[[], None]


class TaskStore:
    def __init__(self, storage: SqliteStorage | None = None) -> None:
        self.storage = storage or SqliteStorage(":memory:")
        self.tasks: dict[str, Task] = {t.id: t for t in self.storage.load_tasks()}
        self.assignments: dict[str, Assignment] = {
            a.id: a for a in self.storage.load_assignments()
        }
        self._listeners: list[Listener] = []

    # observers -------------------------------------------------------------
    def subscribe(self, fn: Listener) -> None:
        self._listeners.append(fn)

    def unsubscribe(self, fn: Listener) -> None:
        if fn in self._listeners:
            self._listeners.remove(fn)

    def notify(self) -> None:
        for fn in list(self._listeners):
            fn()

    # tasks -----------------------------------------------------------------
    def add_task(self, task: Task) -> Task:
        self.tasks[task.id] = task
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

    def ordered_tasks(self) -> list[Task]:
        """Unscheduled first, then by creation time."""
        rank = {TaskStatus.PENDING: 0, TaskStatus.PLANNED: 1, TaskStatus.RUNNING: 2}
        return sorted(self.tasks.values(), key=lambda t: (rank.get(t.status, 3), t.created_at))

    # assignments -----------------------------------------------------------
    def save_assignment(self, a: Assignment, event: str | None) -> None:
        """Persist ``a``; ``event`` (if given) is appended to the history log."""
        self.assignments[a.id] = a
        self.storage.save_assignment(a)
        if event is None:
            return
        self.storage.log_event(
            event,
            a.task_id,
            a.id,
            {"gpus": list(a.gpu_ids), "start": a.planned_start, "end": a.planned_end},
        )

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


def seed_demo(store: TaskStore, now: datetime | None = None) -> None:
    """Populate an empty store with a few example tasks."""
    if store.tasks:
        return
    from .scheduler import AssignmentService  # local: avoid cycle

    now = (now or datetime.now()).replace(second=0, microsecond=0)
    demo = [
        Task("resnet50-baseline", "python train.py --config configs/resnet50_base.yaml",
             1, timedelta(hours=3)),
        Task("vit-b16-pretrain", "python train.py --model vit_b16 --epochs 90",
             2, timedelta(hours=6)),
        Task("bert-finetune-sst2", "python finetune.py --task sst2", 1, timedelta(minutes=90)),
        Task("llama-lora-sweep", "python lora.py --sweep configs/lora.yaml",
             4, timedelta(hours=4)),
        Task("eval-imagenet-c", "python eval.py --suite imagenet-c", 1, timedelta(minutes=45)),
        Task("diffusion-ablation", "python ddpm.py --ablate attn", 2, timedelta(hours=2)),
    ]
    for t in demo:
        t.created_at = now - timedelta(minutes=10)
        store.add_task(t)
    svc = AssignmentService(store, gpu_ids=None)
    svc.assign(demo[0].id, (0,))
    svc.assign(demo[1].id, (1, 2))
    svc.assign(demo[2].id, (0,))

    # one finished run, so the timeline shows history drawn from actual times
    done = Task("data-preprocess", "python prep.py --shards 64", 1, timedelta(hours=2),
                status=TaskStatus.COMPLETED)
    done.created_at = now - timedelta(hours=4)
    store.add_task(done)
    from ..models import AssignmentState

    hist = Assignment(
        task_id=done.id,
        gpu_ids=(3,),
        planned_start=now - timedelta(hours=3),
        planned_end=now - timedelta(hours=1),
        actual_start=now - timedelta(hours=3, minutes=-5),
        actual_end=now - timedelta(hours=1, minutes=25),
        actual_gpu_ids=(3,),
        state=AssignmentState.COMPLETED,
    )
    store.save_assignment(hist, "completed")
    store.notify()
