from __future__ import annotations

from datetime import timedelta

import pytest

from conftest import T0, make_task
from gpu_scheduler.models import Assignment, AssignmentState, TaskStatus
from gpu_scheduler.services import gpu_discovery
from gpu_scheduler.services.gpu_discovery import MockGPUProvider, make_provider


def test_assign_without_conflict(store, svc):
    t = make_task(store)
    res = svc.assign(t.id, [0], T0)
    assert res.ok
    assert res.assignment.planned_end == T0 + timedelta(hours=1)
    assert store.tasks[t.id].status == TaskStatus.PLANNED
    assert len(store.assignments) == 1


def test_overlap_on_same_gpu_is_rejected_and_not_persisted(store, svc):
    a, b = make_task(store, "a"), make_task(store, "b")
    assert svc.assign(a.id, [0], T0).ok
    res = svc.assign(b.id, [0], T0 + timedelta(minutes=30))
    assert not res.ok
    assert [c.task_id for c in res.conflicts] == [a.id]
    assert "a" in res.message
    assert len(store.assignments) == 1
    assert store.tasks[b.id].status == TaskStatus.PENDING


def test_adjacent_intervals_are_allowed(store, svc):
    a, b = make_task(store, "a"), make_task(store, "b")
    assert svc.assign(a.id, [0], T0).ok
    assert svc.assign(b.id, [0], T0 + timedelta(hours=1)).ok


def test_different_gpu_is_fine(store, svc):
    a, b = make_task(store, "a"), make_task(store, "b")
    assert svc.assign(a.id, [0], T0).ok
    assert svc.assign(b.id, [1], T0).ok


def test_multi_gpu_needs_exact_count(store, svc):
    t = make_task(store, gpus=2)
    assert not svc.assign(t.id, [0], T0).ok
    assert not svc.assign(t.id, [0, 0], T0).ok
    assert not svc.assign(t.id, [0, 9], T0).ok
    res = svc.assign(t.id, [2, 1], T0)
    assert res.ok and res.assignment.gpu_ids == (1, 2)


def test_multi_gpu_conflicts_if_any_row_overlaps(store, svc):
    a, b = make_task(store, "a"), make_task(store, "b", gpus=2)
    assert svc.assign(a.id, [1], T0).ok
    res = svc.assign(b.id, [0, 1], T0 + timedelta(minutes=15))
    assert not res.ok
    assert svc.assign(b.id, [2, 3], T0 + timedelta(minutes=15)).ok


def test_move_excludes_self_and_reassign_moves(store, svc):
    t = make_task(store, hours=2)
    a = svc.assign(t.id, [0], T0).assignment
    # overlaps its own old slot: fine
    res = svc.move(a.id, [0], T0 + timedelta(hours=1))
    assert res.ok and store.assignments[a.id].planned_start == T0 + timedelta(hours=1)
    # assign() on an already-planned task moves rather than duplicating
    res = svc.assign(t.id, [3], T0)
    assert res.ok and len(store.assignments) == 1
    assert store.assignments[a.id].gpu_ids == (3,)


def test_unassign_restores_pending(store, svc):
    t = make_task(store)
    a = svc.assign(t.id, [0], T0).assignment
    assert svc.unassign(a.id).ok
    assert store.tasks[t.id].status == TaskStatus.PENDING
    assert not store.assignments


def test_cancelled_assignments_do_not_conflict(store, svc):
    a, b = make_task(store, "a"), make_task(store, "b")
    asg = svc.assign(a.id, [0], T0).assignment
    asg.state = AssignmentState.CANCELLED
    assert svc.assign(b.id, [0], T0).ok


def test_history_uses_actual_times(store, svc):
    """A completed run that overran its estimate blocks the real interval."""
    a, b = make_task(store, "a"), make_task(store, "b")
    done = Assignment(a.id, (0,), T0, T0 + timedelta(hours=1),
                      actual_start=T0, actual_end=T0 + timedelta(hours=3),
                      state=AssignmentState.COMPLETED)
    store.save_assignment(done, "completed")
    assert not svc.assign(b.id, [0], T0 + timedelta(hours=2)).ok
    assert svc.assign(b.id, [0], T0 + timedelta(hours=3)).ok


def test_next_free_slot_and_suggest(store, svc):
    a, b = make_task(store, "a"), make_task(store, "b", gpus=2)
    svc.assign(a.id, [0], T0)
    assert svc.next_free_slot([0], T0, timedelta(hours=1)) == T0 + timedelta(hours=1)
    assert svc.next_free_slot([1], T0, timedelta(hours=1)) == T0
    # anchor 0 is busy at T0 for 2-GPU task window (0,1) -> falls back to anchor+free
    assert svc.suggest_gpus(store.tasks[b.id], 3, T0) == (2, 3)
    assert svc.suggest_gpus(store.tasks[b.id], 1, T0) == (1, 2)


def test_planned_assignment_persists(tmp_path):
    from gpu_scheduler.services.scheduler import AssignmentService
    from gpu_scheduler.services.task_store import TaskStore
    from gpu_scheduler.storage.sqlite import SqliteStorage

    db = tmp_path / "s.db"
    store = TaskStore(SqliteStorage(db))
    t = make_task(store, gpus=2)
    AssignmentService(store, range(4)).assign(t.id, [1, 2], T0)
    store.storage.close()

    again = TaskStore(SqliteStorage(db))
    (a,) = again.assignments.values()
    assert a.gpu_ids == (1, 2) and a.planned_start == T0 and a.actual_end is None
    assert again.tasks[t.id].status == TaskStatus.PLANNED
    events = [h["event"] for h in again.storage.history(t.id)]
    assert events == ["task_created", "assigned"]


def test_gpu_discovery_falls_back_to_mock(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("no NVML")

    monkeypatch.delenv("GPU_SCHEDULER_MOCK_GPUS", raising=False)
    monkeypatch.setattr(gpu_discovery, "NVMLProvider", boom)
    provider = make_provider()
    assert isinstance(provider, MockGPUProvider)
    gpus = provider.discover()
    assert [g.id for g in gpus] == [0, 1, 2, 3]


def test_mock_gpu_count_override(monkeypatch):
    monkeypatch.setenv("GPU_SCHEDULER_MOCK_GPUS", "2")
    assert len(make_provider().discover()) == 2
    assert len(make_provider(6).discover()) == 6


def test_invalid_assignment_interval():
    with pytest.raises(ValueError):
        Assignment("x", (0,), T0, T0)
