"""Several zgm processes write the same database (the daemon, TUIs, zgm submit,
zgm stop). A process working from a copy read before another one changed
things must not undo those changes. Each "process" here has its own
connection, like separate processes do."""

from __future__ import annotations

from test_dispatch import FakeRuntime
from zgm.models import Assignment, AssignmentState, Task
from zgm.services.executor import Executor
from zgm.services.scheduler import AssignmentService
from zgm.services.task_store import TaskStore
from zgm.storage.sqlite import SqliteStorage


class Proc:
    def __init__(self, db, runtime=None):
        self.store = TaskStore(SqliteStorage(db))
        self.svc = AssignmentService(self.store, [0, 1, 2, 3])
        self.executor = Executor(self.store, self.svc, runtime)

    def tick(self):
        self.store.sync()
        return self.executor.tick()

    def state(self, task_id):
        return self.store.storage.assignment_state(
            next(a.id for a in self.store.assignments_for_task(task_id)))


def queued(proc, name, gpus=(0,)):
    t = proc.store.add_task(Task(name, f"echo {name}", len(gpus)))
    assert proc.svc.assign(t.id, gpus).ok
    return t


def test_a_stale_queue_change_does_not_restart_a_running_task(tmp_path):
    """The bug: zgm submit, holding a copy from before the daemon started X,
    rewrote the queue with X still 'planned', and the daemon started X again."""
    db = tmp_path / "s.db"
    rt = FakeRuntime()
    daemon = Proc(db, rt)
    x = queued(daemon, "x")
    submit = Proc(db)  # reads the queue now: x is planned
    daemon.tick()  # starts x
    assert rt.started == ["x"]
    queued(submit, "y", gpus=(0,))  # rewrites the queue from its stale copy
    assert submit.state(x.id) == AssignmentState.RUNNING
    daemon.tick()
    daemon.tick()
    assert rt.started == ["x"]  # not again


def test_a_stale_move_of_a_started_task_is_refused(tmp_path):
    db = tmp_path / "s.db"
    rt = FakeRuntime()
    daemon = Proc(db, rt)
    x = queued(daemon, "x")
    tui = Proc(db)
    daemon.tick()
    a = tui.store.active_assignment(x.id)  # still "planned" in the TUI's copy
    tui.svc.move(a.id, (1,))
    assert tui.state(x.id) == AssignmentState.RUNNING
    tui.store.sync()  # the refused write made it reload
    assert tui.store.active_assignment(x.id).state == AssignmentState.RUNNING


def test_a_stale_queue_change_does_not_bring_back_a_removed_task(tmp_path):
    db = tmp_path / "s.db"
    first = Proc(db)
    gone = queued(first, "gone")
    kept = queued(first, "kept")
    second = Proc(db)
    first.svc.unassign(first.store.active_assignment(gone.id).id)  # e.g. zgm stop
    second.svc.shift(second.store.active_assignment(kept.id).id, -1)  # stale reorder
    fresh = Proc(db)
    assert fresh.store.active_assignment(gone.id) is None
    assert fresh.store.active_assignment(kept.id) is not None


def test_stopping_a_run_that_just_completed_keeps_it_completed(tmp_path):
    db = tmp_path / "s.db"
    rt = FakeRuntime()
    daemon = Proc(db, rt)
    x = queued(daemon, "x")
    daemon.tick()
    client = Proc(db)  # sees x running
    rt.finish("x")
    daemon.tick()  # records the completion
    client.executor.stop(client.store.tasks[x.id], requeue=True)
    assert Proc(db).state(x.id) == AssignmentState.COMPLETED


def test_executor_skips_what_another_process_already_started(tmp_path):
    db = tmp_path / "s.db"
    first, second = Proc(db, FakeRuntime()), Proc(db, FakeRuntime())
    x = queued(first, "x")
    second.store.sync()
    a1 = first.store.active_assignment(x.id)
    a2 = second.store.active_assignment(x.id)
    assert first.executor.start(first.store.tasks[x.id], a1)
    assert not second.executor.start(second.store.tasks[x.id], a2)
    assert first.executor.runtime.started == ["x"] and second.executor.runtime.started == []


def test_state_only_moves_forward(tmp_path):
    storage = SqliteStorage(tmp_path / "s.db")
    a = Assignment("t", (0,), None, None)
    assert storage.save_assignment(a)
    a.state = AssignmentState.RUNNING
    assert storage.save_assignment(a)
    a.state = AssignmentState.PLANNED
    assert not storage.save_assignment(a)
    for final in (AssignmentState.FAILED, AssignmentState.FAILED):
        a.state = final
        assert storage.save_assignment(a)  # the same final state again is fine
    a.state = AssignmentState.CANCELLED
    assert not storage.save_assignment(a)
    assert storage.assignment_state(a.id) == AssignmentState.FAILED
