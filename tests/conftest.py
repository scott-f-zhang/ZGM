from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from zgm.models import Task
from zgm.services.scheduler import AssignmentService
from zgm.services.task_store import TaskStore, add_completed_run
from zgm.storage.sqlite import SqliteStorage

T0 = datetime(2030, 1, 1, 9, 0)


@pytest.fixture
def store() -> TaskStore:
    return TaskStore(SqliteStorage(":memory:"))


@pytest.fixture
def svc(store: TaskStore) -> AssignmentService:
    return AssignmentService(store, [0, 1, 2, 3])


def make_task(store: TaskStore, name: str = "t", gpus: int = 1,
              hours: float | None = 1) -> Task:
    """A pending task. ``hours`` records one past completed run of the same
    workload (long ago, on GPU 0), so the task's estimate is ``hours``;
    ``None`` leaves it with no history."""
    task = Task(name=name, command=f"echo {name}", required_gpus=gpus)
    if hours is not None:
        add_completed_run(store, task, tuple(range(gpus)), T0 - timedelta(days=30),
                          timedelta(hours=hours), name=f"{name} (past run)")
    return store.add_task(task)


def planned_count(store: TaskStore) -> int:
    return sum(1 for a in store.assignments.values() if a.is_active)
