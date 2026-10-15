from __future__ import annotations

from datetime import timedelta

import pytest

from conftest import T0, make_task
from gpu_scheduler.models import Task, TaskStatus, format_duration, parse_duration
from gpu_scheduler.services.task_store import TaskStore, seed_demo
from gpu_scheduler.storage.sqlite import SqliteStorage


def test_add_remove_round_trip(tmp_path):
    db = tmp_path / "q.db"
    store = TaskStore(SqliteStorage(db))
    t = store.add_task(Task("train", "python train.py", 2, timedelta(minutes=90),
                            config_path="/x/a.yaml"))
    store.storage.close()

    loaded = TaskStore(SqliteStorage(db))
    got = loaded.tasks[t.id]
    assert (got.name, got.required_gpus, got.estimated_duration, got.config_path) == (
        "train", 2, timedelta(minutes=90), "/x/a.yaml")
    loaded.remove_task(t.id)
    assert TaskStore(SqliteStorage(db)).tasks == {}


def test_remove_task_drops_assignments(store, svc):
    t = make_task(store)
    svc.assign(t.id, [0], T0)
    store.remove_task(t.id)
    assert not store.assignments


def test_ordering_puts_unscheduled_first(store, svc):
    a, b = make_task(store, "a"), make_task(store, "b")
    svc.assign(a.id, [0], T0)
    assert [t.name for t in store.ordered_tasks()] == ["b", "a"]


def test_listeners_fire(store):
    calls = []
    store.subscribe(lambda: calls.append(1))
    make_task(store)
    assert calls


def test_task_validation():
    with pytest.raises(ValueError):
        Task("x", "y", required_gpus=0)
    with pytest.raises(ValueError):
        Task("x", "y", estimated_duration=timedelta(0))


@pytest.mark.parametrize("text,minutes", [
    ("90", 90), ("90m", 90), ("2h", 120), ("1h30m", 90), ("1.5h", 90), ("1d", 1440),
])
def test_parse_duration(text, minutes):
    assert parse_duration(text) == timedelta(minutes=minutes)


@pytest.mark.parametrize("bad", ["", "abc", "h", "3x", "0"])
def test_parse_duration_rejects(bad):
    with pytest.raises(ValueError):
        parse_duration(bad)


def test_format_duration():
    assert format_duration(timedelta(minutes=90)) == "1h30m"
    assert format_duration(timedelta(hours=2)) == "2h"
    assert format_duration(timedelta(minutes=45)) == "45m"


def test_seed_demo_is_consistent(store):
    seed_demo(store)
    assert len(store.tasks) == 7
    planned = [t for t in store.tasks.values() if t.status == TaskStatus.PLANNED]
    assert len(planned) == 3
    seed_demo(store)  # idempotent on a non-empty store
    assert len(store.tasks) == 7
