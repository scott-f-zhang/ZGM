from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from gpu_scheduler.models import Task
from gpu_scheduler.services.scheduler import AssignmentService
from gpu_scheduler.services.task_store import TaskStore
from gpu_scheduler.storage.sqlite import SqliteStorage

T0 = datetime(2030, 1, 1, 9, 0)


@pytest.fixture
def store() -> TaskStore:
    return TaskStore(SqliteStorage(":memory:"))


@pytest.fixture
def svc(store: TaskStore) -> AssignmentService:
    return AssignmentService(store, [0, 1, 2, 3])


def make_task(store: TaskStore, name: str = "t", gpus: int = 1, hours: float = 1) -> Task:
    return store.add_task(
        Task(name=name, command=f"echo {name}", required_gpus=gpus,
             estimated_duration=timedelta(hours=hours))
    )
