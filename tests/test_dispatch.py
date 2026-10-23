"""Auto-dispatch: queued tasks start as their GPUs free up, in queue order."""

from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

import pytest

from conftest import make_task
from zgm.app import ZGMApp
from zgm.models import AssignmentState, LaunchSpec, Task, TaskStatus
from zgm.runtime.base import RunHandle, RunState
from zgm.runtime.local import LocalRuntime

EXAMPLES = Path(__file__).parent.parent / "examples" / "configs"
SIZE = (140, 42)


# the rule --------------------------------------------------------------------------
def names(svc, assignments):
    return [svc.store.tasks[a.task_id].name for a in assignments]


def test_ready_in_queue_order_one_per_gpu(store, svc):
    for name, gpu in [("a", 0), ("b", 1), ("c", 0)]:
        svc.assign(make_task(store, name).id, [gpu])
    assert names(svc, svc.ready_to_start()) == ["a", "b"]  # c waits behind a on GPU 0


def test_running_work_blocks_its_gpus(store, svc):
    a = make_task(store, "a")
    svc.assign(a.id, [1])
    run = store.active_assignment(a.id)
    run.state = AssignmentState.RUNNING
    svc.assign(make_task(store, "b").id, [1])
    svc.assign(make_task(store, "c").id, [2])
    assert names(svc, svc.ready_to_start()) == ["c"]


def test_no_backfill_past_a_waiting_multi_gpu_task(store, svc):
    """d (GPU 0,1) waits for GPU 0; e (GPU 1) is queued after d, so it waits too,
    exactly as the timeline shows it."""
    busy = make_task(store, "busy")
    svc.assign(busy.id, [0])
    store.active_assignment(busy.id).state = AssignmentState.RUNNING
    svc.assign(make_task(store, "d", gpus=2).id, [0, 1])
    svc.assign(make_task(store, "e").id, [1])
    svc.assign(make_task(store, "f").id, [3])
    assert names(svc, svc.ready_to_start()) == ["f"]


# the app ---------------------------------------------------------------------------
class FakeRuntime:
    """Runs nothing; tests decide when each run ends."""

    name = "fake"

    def __init__(self, fail_tasks: tuple[str, ...] = ()) -> None:
        self.state: dict[str, RunState] = {}
        self.fail_tasks = fail_tasks
        self.started: list[str] = []

    def available(self) -> bool:
        return True

    def submit(self, task: Task, assignment) -> RunHandle:
        if task.name in self.fail_tasks:
            raise ValueError("working directory not found: /nope")
        self.started.append(task.name)
        self.state[task.name] = RunState.RUNNING
        return RunHandle(assignment.id, task.name)

    def finish(self, name: str, state: RunState = RunState.SUCCEEDED) -> None:
        self.state[name] = state

    def cancel(self, handle: RunHandle) -> None:
        self.state[handle.ref] = RunState.CANCELLED

    def status(self, handle: RunHandle) -> RunState:
        return self.state[handle.ref]


@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    d = tmp_path / "configs"
    shutil.copytree(EXAMPLES, d)
    return d


def make_app(config_dir, runtime, autostart=True) -> ZGMApp:
    return ZGMApp(mock_gpus=4, config_dir=config_dir, stats_interval=0, runtime=runtime,
                  autostart=autostart)


def queue(app, *specs) -> dict[str, Task]:
    out = {}
    for name, gpus in specs:
        t = app.store.add_task(Task(name, f"echo {name}", len(gpus)))
        assert app.assignment_service.assign(t.id, gpus).ok
        out[name] = t
    return out


def tick(app) -> None:
    app.poll_runs()
    app.dispatch()


async def test_queue_drains_as_gpus_free_up(config_dir):
    rt = FakeRuntime()
    app = make_app(config_dir, rt)
    async with app.run_test(size=SIZE) as pilot:
        t = queue(app, ("a", [0]), ("b", [1]), ("c", [0]), ("d", [0, 1]))
        tick(app)
        assert rt.started == ["a", "b"]
        rt.finish("a")
        tick(app)
        assert rt.started == ["a", "b", "c"]
        rt.finish("b", RunState.FAILED)  # a failure frees its GPUs as well
        tick(app)
        assert rt.started == ["a", "b", "c"]  # d still needs GPU 0
        rt.finish("c")
        tick(app)
        assert rt.started == ["a", "b", "c", "d"]
        assert app.store.tasks[t["a"].id].status == TaskStatus.COMPLETED
        assert app.store.tasks[t["b"].id].status == TaskStatus.FAILED
        assert app.store.tasks[t["d"].id].status == TaskStatus.RUNNING
        await pilot.pause()


async def test_pause_and_resume(config_dir):
    rt = FakeRuntime()
    app = make_app(config_dir, rt, autostart=False)
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause()
        queue(app, ("a", [0]))
        tick(app)
        assert rt.started == []
        await pilot.press("p")
        await pilot.pause()
        assert app.autostart and rt.started == ["a"]
        await pilot.press("p")
        queue(app, ("b", [1]))
        tick(app)
        assert rt.started == ["a"]


async def test_launch_failure_does_not_block_the_queue(config_dir):
    rt = FakeRuntime(fail_tasks=("bad",))
    app = make_app(config_dir, rt)
    async with app.run_test(size=SIZE) as pilot:
        t = queue(app, ("bad", [0]), ("next", [0]))
        tick(app)
        assert app.store.tasks[t["bad"].id].status == TaskStatus.FAILED
        assert app.store.active_assignment(t["bad"].id) is None
        tick(app)
        assert rt.started == ["next"]
        await pilot.pause()


async def test_run_now_skips_the_queue_but_not_a_busy_gpu(config_dir):
    rt = FakeRuntime()
    app = make_app(config_dir, rt, autostart=False)
    async with app.run_test(size=SIZE) as pilot:
        t = queue(app, ("a", [0]), ("b", [1]), ("c", [0]))
        app.run_selected(t["b"])  # ahead of a: fine, GPU 1 is free
        assert rt.started == ["b"]
        app.run_selected(t["a"])
        app.run_selected(t["c"])  # GPU 0 is busy with a
        assert rt.started == ["b", "a"]
        assert app.store.tasks[t["c"].id].status == TaskStatus.PLANNED
        await pilot.pause()


async def test_no_dispatch_while_placing(config_dir):
    rt = FakeRuntime()
    app = make_app(config_dir, rt, autostart=False)
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause()
        queue(app, ("a", [0]))
        await pilot.press("t")  # timeline: cursor on a's card
        await pilot.press("enter")  # pick it up
        await pilot.pause()
        assert app.screen.placement is not None
        app.autostart = True
        tick(app)
        assert rt.started == []
        await pilot.press("escape")
        await pilot.pause()
        tick(app)
        assert rt.started == ["a"]


async def test_status_line_shows_state(config_dir):
    app = make_app(config_dir, FakeRuntime(), autostart=False)
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause()
        status = app.screen.query_one("#status-line")
        assert "PAUSED" in str(status.render())
        await pilot.press("p")
        await pilot.pause()
        assert "AUTO" in str(status.render())


# end to end: real processes, one after the other on the same GPU ----------------------
async def test_local_runtime_runs_the_queue_in_order(config_dir, tmp_path):
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "job.py").write_text(
        "import os, time\ntime.sleep(0.5)\nprint('ran on', os.environ['CUDA_VISIBLE_DEVICES'])\n",
        encoding="utf-8")
    app = make_app(config_dir, LocalRuntime(tmp_path / "logs"))
    async with app.run_test(size=SIZE) as pilot:
        tasks = []
        for name in ("first", "second"):
            t = app.store.add_task(Task(name, launch=LaunchSpec(
                cwd=str(proj), python=sys.executable, script="job.py")))
            app.assignment_service.assign(t.id, [2])
            tasks.append(t)
        deadline = time.time() + 30
        while time.time() < deadline:
            tick(app)
            if all(app.store.tasks[t.id].status == TaskStatus.COMPLETED for t in tasks):
                break
            await pilot.pause(0.1)
        a1, a2 = (app.store.assignments_for_task(t.id)[0] for t in tasks)
        assert a1.state == a2.state == AssignmentState.COMPLETED
        assert a2.actual_start >= a1.actual_end  # one GPU: strictly one after the other
        assert "ran on 2" in app.log_path(tasks[1]).read_text(encoding="utf-8")
