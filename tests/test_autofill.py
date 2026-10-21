"""Auto-fill: waiting tasks marked for it start on GPUs that have nothing to do."""

from __future__ import annotations

from datetime import datetime, timedelta

from zgm.models import Assignment, AssignmentState, Task, TaskStatus
from zgm.runtime.mock import MockRuntime
from zgm.services.executor import Executor
from zgm.services.scheduler import AssignmentService
from zgm.services.task_store import TaskStore


def setup(gpus=(0, 1, 2, 3), node_of=None, autofill=True):
    store = TaskStore()
    svc = AssignmentService(store, list(gpus), node_of=node_of)
    ex = Executor(store, svc, MockRuntime(speedup=1))  # runs last an hour
    ex.autofill = autofill
    return store, svc, ex


def waiting(store, name, gpus=1, minutes_ago=0, autofill=True):
    t = Task(name, f"echo {name}", gpus, autofill=autofill)
    t.created_at = datetime.now() - timedelta(minutes=minutes_ago)
    return store.add_task(t)


def gpus_of(store, task):
    a = store.active_assignment(task.id)
    return a.gpu_ids if a and a.state == AssignmentState.RUNNING else None


def test_only_tasks_marked_for_it():
    store, _, ex = setup(autofill=False)
    assert Executor(TaskStore(), ex.svc).autofill is True  # the switch is on by default
    manual = waiting(store, "manual", minutes_ago=5, autofill=False)
    marked = waiting(store, "marked")
    assert ex.tick() == []  # switched off
    ex.autofill = True
    assert [t.name for t in ex.tick()] == ["marked"]
    assert store.tasks[manual.id].status == TaskStatus.PENDING


def test_the_mark_is_stored(tmp_path):
    from zgm.storage.sqlite import SqliteStorage

    db = SqliteStorage(tmp_path / "s.db")
    TaskStore(db).add_task(Task("a", "echo a", autofill=True))
    (t,) = TaskStore(SqliteStorage(tmp_path / "s.db")).tasks.values()
    assert t.autofill


def test_oldest_waiting_tasks_go_to_idle_gpus_only():
    store, svc, ex = setup()
    queued = store.add_task(Task("queued", "echo q", 1))
    assert svc.assign(queued.id, (1,)).ok  # GPU 1 has queued work: not idle
    old, new = waiting(store, "old", minutes_ago=10), waiting(store, "new")
    third = waiting(store, "third", minutes_ago=5)
    started = [t.name for t in ex.tick()]
    assert started == ["queued", "old", "third", "new"]
    assert gpus_of(store, queued) == (1,)
    assert [gpus_of(store, t) for t in (old, third, new)] == [(0,), (2,), (3,)]
    assert ex.tick() == []  # nothing idle any more


def test_a_task_that_does_not_fit_is_passed_over():
    store, svc, ex = setup()
    busy = store.add_task(Task("busy", "echo b", 1))
    svc.assign(busy.id, (1,))
    big = waiting(store, "big", gpus=3, minutes_ago=10)  # GPUs 0, 2, 3 are not adjacent
    pair = waiting(store, "pair", gpus=2, minutes_ago=5)
    small = waiting(store, "small")
    assert [t.name for t in ex.tick()] == ["busy", "pair", "small"]
    assert gpus_of(store, pair) == (2, 3) and gpus_of(store, small) == (0,)
    assert store.tasks[big.id].status == TaskStatus.PENDING


def test_tasks_stay_on_one_host():
    node_of = {0: "node07", 1: "node08", 2: "node08"}
    store, svc, ex = setup(gpus=(0, 1, 2), node_of=node_of)
    pair = waiting(store, "pair", gpus=2)
    ex.tick()
    assert gpus_of(store, pair) == (1, 2)


def test_skips_failed_and_stopped_tasks():
    store, svc, ex = setup(gpus=(0,))
    failed = waiting(store, "failed", minutes_ago=30)
    failed.status = TaskStatus.FAILED
    stopped = waiting(store, "stopped", minutes_ago=20)  # back in the waiting list
    store.save_assignment(Assignment(
        task_id=stopped.id, gpu_ids=(0,), planned_start=None, planned_end=None,
        actual_start=datetime.now() - timedelta(hours=1), actual_end=datetime.now(),
        state=AssignmentState.CANCELLED), "stopped")
    again = waiting(store, "ran-before", minutes_ago=10)  # completed earlier: may run again
    store.save_assignment(Assignment(
        task_id=again.id, gpu_ids=(0,), planned_start=None, planned_end=None,
        actual_start=datetime.now() - timedelta(hours=2), actual_end=datetime.now(),
        state=AssignmentState.COMPLETED), "completed")
    assert [t.name for t in ex.tick()] == ["ran-before"]
    assert store.tasks[failed.id].status == TaskStatus.FAILED
    assert store.tasks[stopped.id].status == TaskStatus.PENDING


def test_waits_while_paused_and_avoids_gpus_used_elsewhere():
    store, svc, ex = setup(gpus=(0, 1))
    t = waiting(store, "a")
    ex.autostart = False
    assert ex.tick() == []
    ex.autostart = True
    svc.blocked = {0}  # another program uses GPU 0
    ex.tick()
    assert gpus_of(store, t) == (1,)


def test_a_task_that_cannot_start_is_not_tried_again():
    store, svc, ex = setup(gpus=(0,))

    class Broken(MockRuntime):
        def submit(self, task, assignment):
            raise RuntimeError("no such working directory")

    ex.runtime = Broken()
    t = waiting(store, "broken")
    assert ex.tick() == []
    assert store.tasks[t.id].status == TaskStatus.FAILED
    assert ex.tick() == [] and store.active_assignment(t.id) is None


def test_higher_priority_goes_first():
    store, _, ex = setup(gpus=(0, 1))
    waiting(store, "old", minutes_ago=30)
    low = waiting(store, "low", minutes_ago=20)
    low.priority = -1
    urgent = waiting(store, "urgent", minutes_ago=1)
    urgent.priority = 5
    assert [t.name for t in ex.tick()] == ["urgent", "old"]  # then "low", when a GPU frees up


def test_priority_is_stored(tmp_path):
    from zgm.storage.sqlite import SqliteStorage

    TaskStore(SqliteStorage(tmp_path / "s.db")).add_task(Task("a", "echo a", priority=3))
    (t,) = TaskStore(SqliteStorage(tmp_path / "s.db")).tasks.values()
    assert t.priority == 3
