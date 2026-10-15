from __future__ import annotations

from datetime import timedelta

import pytest

from conftest import T0, make_task, planned_count
from gpu_scheduler.models import Task, TaskStatus, format_duration, parse_duration
from gpu_scheduler.services.task_store import TaskStore, seed_demo
from gpu_scheduler.storage.sqlite import SqliteStorage


def test_add_remove_round_trip(tmp_path):
    db = tmp_path / "q.db"
    store = TaskStore(SqliteStorage(db))
    t = store.add_task(Task("train", "python train.py", 2, config_path="/x/a.yaml",
                            fingerprint="cfg:abc"))
    store.storage.close()

    loaded = TaskStore(SqliteStorage(db))
    got = loaded.tasks[t.id]
    assert (got.name, got.required_gpus, got.config_path, got.fingerprint) == (
        "train", 2, "/x/a.yaml", "cfg:abc")
    assert got.estimated_duration is None  # no history
    loaded.remove_task(t.id)
    assert TaskStore(SqliteStorage(db)).tasks == {}


def test_remove_task_drops_assignments(store, svc):
    t = make_task(store)
    svc.assign(t.id, [0])
    store.remove_task(t.id)
    assert planned_count(store) == 0


def test_queue_holds_only_waiting_tasks(store, svc):
    a, b, c = make_task(store, "a"), make_task(store, "b"), make_task(store, "c")
    svc.assign(a.id, [0])
    c.status = TaskStatus.CANCELLED  # stopped and put back: waits again
    assert [t.name for t in store.waiting_tasks()] == ["b", "c"]
    svc.unassign(store.active_assignment(a.id).id)
    assert {t.name for t in store.waiting_tasks()} == {"a", "b", "c"}


def test_listeners_fire(store):
    calls = []
    store.subscribe(lambda: calls.append(1))
    make_task(store)
    assert calls


def test_task_validation():
    with pytest.raises(ValueError):
        Task("x", "y", required_gpus=0)
    assert Task("x", "y").estimated_duration is None
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
    n = len(store.tasks)
    planned = {t.name: t for t in store.tasks.values() if t.status == TaskStatus.PLANNED}
    assert len(planned) == 3
    # estimates exist only where there is run history
    assert planned["resnet50-baseline"].estimate_runs == 2
    assert planned["resnet50-baseline"].estimated_duration == timedelta(hours=2, minutes=57, seconds=30)
    assert planned["vit-b16-pretrain"].estimated_duration is None
    assert len(store.waiting_tasks()) == 3
    seed_demo(store)  # idempotent on a non-empty store
    assert len(store.tasks) == n


def test_migrates_v1_database(tmp_path):
    import sqlite3

    db = tmp_path / "v1.db"
    conn = sqlite3.connect(db)
    conn.executescript("""
        CREATE TABLE tasks (id TEXT PRIMARY KEY, name TEXT NOT NULL, command TEXT NOT NULL,
            required_gpus INTEGER NOT NULL, estimated_seconds REAL NOT NULL, status TEXT NOT NULL,
            created_at TEXT NOT NULL, notes TEXT NOT NULL DEFAULT '', config_path TEXT);
        CREATE TABLE assignments (id TEXT PRIMARY KEY, task_id TEXT NOT NULL, gpu_ids TEXT NOT NULL,
            preferred_gpu_ids TEXT NOT NULL, actual_gpu_ids TEXT, planned_start TEXT NOT NULL,
            planned_end TEXT NOT NULL, actual_start TEXT, actual_end TEXT, state TEXT NOT NULL);
        PRAGMA user_version = 1;
    """)
    for tid, start in (("late", "2030-01-01T12:00:00"), ("early", "2030-01-01T09:00:00")):
        conn.execute("INSERT INTO tasks VALUES (?,?,?,?,?,?,?,?,?)",
                     (tid, tid, "echo", 1, 3600, "planned", "2030-01-01T00:00:00", "", None))
        conn.execute("INSERT INTO assignments VALUES (?,?,?,?,?,?,?,?,?,?)",
                     ("a-" + tid, tid, "[0]", "[0]", None, start,
                      start.replace("T09", "T10").replace("T12", "T13"), None, None, "planned"))
    conn.commit()
    conn.close()

    from gpu_scheduler.services.scheduler import AssignmentService

    store = TaskStore(SqliteStorage(db))
    svc = AssignmentService(store, range(4))
    assert [a.task_id for a in svc.planned()] == ["early", "late"]
    assert [a.seq for a in svc.planned()] == [0, 1]
    from gpu_scheduler.storage.sqlite import SCHEMA_VERSION

    assert store.storage.conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    assert store.tasks["early"].fingerprint.startswith("cmd:")  # derived for old rows


def test_estimate_is_median_of_matching_runs(store):
    from gpu_scheduler.services.task_store import add_completed_run

    t = make_task(store, "job", hours=None)
    assert t.estimated_duration is None and t.estimate_runs == 0
    for h in (1, 2, 6):
        add_completed_run(store, t, (0,), T0 - timedelta(days=h), timedelta(hours=h))
    store.notify()
    assert t.estimated_duration == timedelta(hours=2) and t.estimate_runs == 3
    # a different command, or the same command on more GPUs, is a different workload
    other = store.add_task(Task("job", "echo job --lr 0.1", 1))
    wider = store.add_task(Task("job", t.command, 2))
    assert other.estimated_duration is None and wider.estimated_duration is None
    # renaming keeps the history
    t.name = "renamed"
    store.update_task(t)
    assert t.estimated_duration == timedelta(hours=2)


def test_cancelled_or_failed_runs_do_not_count(store):
    from gpu_scheduler.models import Assignment, AssignmentState

    t = make_task(store, "job", hours=None)
    for state in (AssignmentState.CANCELLED, AssignmentState.FAILED):
        store.save_assignment(Assignment(t.id, (0,), T0, T0 + timedelta(hours=1),
                                         actual_start=T0, actual_end=T0 + timedelta(minutes=5),
                                         state=state), "x")
    store.notify()
    assert t.estimated_duration is None


def test_migrates_v2_not_null_columns(tmp_path):
    import sqlite3

    db = tmp_path / "v2.db"
    conn = sqlite3.connect(db)
    conn.executescript("""
        CREATE TABLE tasks (id TEXT PRIMARY KEY, name TEXT NOT NULL, command TEXT NOT NULL,
            required_gpus INTEGER NOT NULL, estimated_seconds REAL NOT NULL, status TEXT NOT NULL,
            created_at TEXT NOT NULL, notes TEXT NOT NULL DEFAULT '', config_path TEXT);
        CREATE TABLE assignments (id TEXT PRIMARY KEY, task_id TEXT NOT NULL, gpu_ids TEXT NOT NULL,
            preferred_gpu_ids TEXT NOT NULL, actual_gpu_ids TEXT, planned_start TEXT NOT NULL,
            planned_end TEXT NOT NULL, actual_start TEXT, actual_end TEXT, state TEXT NOT NULL,
            seq INTEGER NOT NULL DEFAULT 0);
        INSERT INTO tasks VALUES ('t1','x','echo',1,3600,'pending','2030-01-01T00:00:00','',NULL);
        PRAGMA user_version = 2;
    """)
    conn.commit()
    conn.close()
    store = TaskStore(SqliteStorage(db))
    assert store.tasks["t1"].name == "x"
    t = store.tasks["t1"]
    t.estimated_duration = None
    store.update_task(t)  # nullable now
    from gpu_scheduler.models import Assignment

    store.save_assignment(Assignment("t1", (0,), None, None), None)  # unknown times persist
    assert TaskStore(SqliteStorage(db)).assignments
