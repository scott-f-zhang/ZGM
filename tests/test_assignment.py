from __future__ import annotations

import random
from datetime import datetime, timedelta

import pytest

from conftest import T0, make_task, planned_count
from zgm.models import Assignment, AssignmentState, TaskStatus
from zgm.services import gpu_discovery
from zgm.services.gpu_discovery import MockGPUProvider, make_provider
from zgm.services.scheduler import ConflictValidator


def starts(svc, now=T0):
    """task name -> (start, end) for the current queue, scheduled at ``now``."""
    return {svc.store.tasks[s.task_id].name: (s.start, s.end) for s in svc.schedule(now)}


def names(svc):
    return [svc.store.tasks[a.task_id].name for a in svc.planned()]


def test_append_starts_now_then_queues_behind(store, svc):
    a, b = make_task(store, "a"), make_task(store, "b", hours=2)
    assert svc.assign(a.id, [0]).ok
    assert svc.assign(b.id, [0]).ok
    s = starts(svc)
    assert s["a"] == (T0, T0 + timedelta(hours=1))
    assert s["b"] == (T0 + timedelta(hours=1), T0 + timedelta(hours=3))
    assert store.tasks[a.id].status == TaskStatus.PLANNED


def test_different_gpus_run_in_parallel(store, svc):
    a, b = make_task(store, "a"), make_task(store, "b")
    svc.assign(a.id, [0])
    svc.assign(b.id, [1])
    s = starts(svc)
    assert s["a"][0] == s["b"][0] == T0


def test_multi_gpu_waits_for_latest_gpu(store, svc):
    a, b = make_task(store, "a", hours=1), make_task(store, "b", hours=3)
    c = make_task(store, "c", gpus=2)
    svc.assign(a.id, [0])
    svc.assign(b.id, [1])
    svc.assign(c.id, [0, 1])
    assert starts(svc)["c"][0] == T0 + timedelta(hours=3)


def test_insert_at_position_shifts_later_work(store, svc):
    a, b, c = (make_task(store, n) for n in "abc")
    svc.assign(a.id, [0])
    svc.assign(b.id, [0])
    assert svc.assign(c.id, [0], position=0).ok
    assert names(svc) == ["c", "a", "b"]
    s = starts(svc)
    assert s["c"][0] == T0 and s["a"][0] == T0 + timedelta(hours=1)
    assert [a.seq for a in svc.planned()] == [0, 1, 2]


def test_move_reorders_and_changes_gpu(store, svc):
    a, b, c = (make_task(store, n) for n in "abc")
    for t in (a, b, c):
        svc.assign(t.id, [0])
    aid = store.active_assignment(c.id).id
    assert svc.move(aid, [0], 0).ok
    assert names(svc) == ["c", "a", "b"]
    assert svc.move(aid, [2], 2).ok
    assert names(svc) == ["a", "b", "c"]
    s = starts(svc)
    assert s["c"][0] == T0  # alone on GPU 2
    # assign() on an already-planned task moves rather than duplicating
    assert svc.assign(c.id, [1], 0).ok
    assert planned_count(store) == 3 and names(svc)[0] == "c"


def test_shift(store, svc):
    a, b, c = (make_task(store, n) for n in "abc")
    for t in (a, b, c):
        svc.assign(t.id, [0])
    svc.shift(store.active_assignment(c.id).id, -1)
    assert names(svc) == ["a", "c", "b"]
    svc.shift(store.active_assignment(a.id).id, +5)
    assert names(svc) == ["c", "b", "a"]


def test_unassign_pulls_later_work_forward(store, svc):
    a, b = make_task(store, "a"), make_task(store, "b")
    svc.assign(a.id, [0])
    svc.assign(b.id, [0])
    assert svc.unassign(store.active_assignment(a.id).id).ok
    assert store.tasks[a.id].status == TaskStatus.PENDING
    assert starts(svc)["b"][0] == T0
    assert [x.seq for x in svc.planned()] == [0]


def test_running_work_blocks_its_gpus(store, svc):
    a, b = make_task(store, "a", hours=2), make_task(store, "b")
    run = Assignment(a.id, (0,), T0, T0 + timedelta(hours=2), actual_start=T0,
                     state=AssignmentState.RUNNING)
    store.save_assignment(run, "started")
    svc.assign(b.id, [0])
    assert starts(svc, T0)["b"][0] == T0 + timedelta(hours=2)
    # overrunning: queued work waits until "now"
    late = T0 + timedelta(hours=5)
    assert starts(svc, late)["b"][0] == late
    kinds = [s.kind for s in svc.schedule(T0)]
    assert kinds == ["running", "planned"]


def test_shape_errors_are_refused(store, svc):
    t = make_task(store, gpus=2)
    for gpus in ([0], [0, 0], [0, 9]):
        res = svc.assign(t.id, gpus)
        assert not res.ok and res.error
    assert planned_count(store) == 0
    assert svc.assign(t.id, [2, 1]).assignment.gpu_ids == (1, 2)


def test_completed_task_cannot_be_planned(store, svc):
    t = make_task(store)
    t.status = TaskStatus.COMPLETED
    assert not svc.assign(t.id, [0]).ok


def test_preview_does_not_persist(store, svc):
    a, b = make_task(store, "a"), make_task(store, "b")
    svc.assign(a.id, [0])
    slots, ghost = svc.preview(b.id, [0], 0, now=T0)
    assert ghost.start == T0 and [s.kind for s in slots] == ["ghost", "planned"]
    assert slots[1].start == T0 + timedelta(hours=1)
    assert planned_count(store) == 1


def test_suggest_gpus(store, svc):
    t2, t4 = make_task(store, gpus=2), make_task(store, gpus=4)
    assert svc.suggest_gpus(t2, 1) == (1, 2)
    assert svc.suggest_gpus(t2, 3) == (2, 3)  # clamped to fit
    assert svc.suggest_gpus(t4, 2) == (0, 1, 2, 3)


def test_random_operations_never_overlap(store, svc):
    rng = random.Random(7)
    tasks = [make_task(store, f"t{i}", gpus=rng.choice([1, 1, 2, 4]), hours=rng.choice([1, 2, 3]))
             for i in range(12)]
    for _ in range(80):
        t = rng.choice(tasks)
        op = rng.random()
        a = store.active_assignment(t.id)
        if op < 0.6:
            first = rng.randrange(0, 5 - t.required_gpus)
            svc.assign(t.id, list(range(first, first + t.required_gpus)),
                       rng.randrange(0, len(svc.planned()) + 1))
        elif a is not None and op < 0.8:
            svc.unassign(a.id)
        elif a is not None:
            svc.shift(a.id, rng.choice([-2, -1, 1, 2]))
    svc.recompute(T0)
    planned = svc.planned()
    assert [a.seq for a in planned] == list(range(len(planned)))
    v = ConflictValidator()
    for a in planned:
        assert not v.find_conflicts(a, planned, T0)


def test_planned_assignment_persists(tmp_path):
    from zgm.services.scheduler import AssignmentService
    from zgm.services.task_store import TaskStore
    from zgm.storage.sqlite import SqliteStorage

    db = tmp_path / "s.db"
    store = TaskStore(SqliteStorage(db))
    svc = AssignmentService(store, range(4))
    t, u = make_task(store, "t", gpus=2), make_task(store, "u")
    svc.assign(t.id, [1, 2])
    svc.assign(u.id, [1], 0)
    store.storage.close()

    again = TaskStore(SqliteStorage(db))
    svc2 = AssignmentService(again, range(4))
    assert names(svc2) == ["u", "t"]
    a = again.active_assignment(t.id)
    assert a.gpu_ids == (1, 2) and a.actual_end is None
    assert again.tasks[t.id].status == TaskStatus.PLANNED
    events = [h["event"] for h in again.storage.history(t.id)]
    assert events[:2] == ["task_created", "assigned"]


def test_gpu_discovery_falls_back_to_mock(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("no NVML")

    monkeypatch.delenv("ZGM_MOCK_GPUS", raising=False)
    monkeypatch.setattr(gpu_discovery, "NVMLProvider", boom)
    monkeypatch.setattr(gpu_discovery, "AppleGPUProvider", boom)
    provider = make_provider()
    assert isinstance(provider, MockGPUProvider)
    assert [g.id for g in provider.discover()] == [0, 1, 2, 3]


def test_mock_gpu_count_override(monkeypatch):
    monkeypatch.setenv("ZGM_MOCK_GPUS", "2")
    assert len(make_provider().discover()) == 2
    assert len(make_provider(6).discover()) == 6


def test_invalid_assignment_interval():
    with pytest.raises(ValueError):
        Assignment("x", (0,), T0, T0)


def test_validator_detects_overlap():
    a = Assignment("a", (0,), T0, T0 + timedelta(hours=2))
    b = Assignment("b", (0, 1), T0 + timedelta(hours=1), T0 + timedelta(hours=3))
    c = Assignment("c", (0,), T0 + timedelta(hours=2), T0 + timedelta(hours=3))
    v = ConflictValidator()
    assert v.find_conflicts(a, [b, c], datetime(2000, 1, 1)) == [b]


def test_unknown_estimate_makes_later_starts_unknown(store, svc):
    a = make_task(store, "fresh", hours=None)  # never run before
    b = make_task(store, "after", hours=1)
    c = make_task(store, "elsewhere", hours=1)
    assert a.estimated_duration is None and b.estimated_duration == timedelta(hours=1)
    svc.assign(a.id, [0])
    svc.assign(b.id, [0])
    svc.assign(c.id, [1])
    s = starts(svc)
    assert s["fresh"] == (T0, None)  # starts now, end unknown
    assert s["after"] == (None, None)  # behind unknown work on GPU 0
    assert s["elsewhere"] == (T0, T0 + timedelta(hours=1))  # other GPU unaffected
    svc.recompute(T0)
    assert store.active_assignment(b.id).planned_start is None


def test_running_without_estimate_blocks_with_unknown_end(store, svc):
    a, b = make_task(store, "a", hours=None), make_task(store, "b")
    store.save_assignment(Assignment(a.id, (0,), T0, None, actual_start=T0,
                                     state=AssignmentState.RUNNING), "started")
    svc.assign(b.id, [0])
    kinds = {s.kind: s for s in svc.schedule(T0 + timedelta(hours=1))}
    assert kinds["running"].end is None and kinds["planned"].start is None
