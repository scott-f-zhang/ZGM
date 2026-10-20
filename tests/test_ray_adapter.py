from __future__ import annotations

import time
from datetime import datetime, timedelta

import pytest

from zgm.models import Assignment, Task
from zgm.runtime.base import RunState
from zgm.runtime.ray_adapter import RayRuntime

ray = pytest.importorskip("ray")
pytestmark = pytest.mark.ray


@pytest.fixture(scope="module")
def runtime():
    rt = RayRuntime(num_cpus=1, include_dashboard=False, log_to_driver=False)
    yield rt
    ray.shutdown()


def wait(rt, handle, timeout=60):
    deadline = time.time() + timeout
    while time.time() < deadline:
        state = rt.status(handle)
        if state not in (RunState.PENDING, RunState.RUNNING):
            return state
        time.sleep(0.2)
    raise TimeoutError


def test_submit_and_status(runtime):
    task = Task("echo", "echo hello-from-ray", 1, timedelta(minutes=1))
    now = datetime.now()
    a = Assignment(task.id, (0,), now, now + task.estimated_duration)
    # num_gpus=0: this container has no GPU; the GPU path is identical apart from the resource
    handle = runtime.submit(task, a, num_gpus=0)
    assert wait(runtime, handle) == RunState.SUCCEEDED
    result = runtime.result(handle)
    assert "hello-from-ray" in result["stdout_tail"]
    assert result["gpu_ids"] == []


def test_failed_command(runtime):
    task = Task("fail", "exit 3", 1, timedelta(minutes=1))
    now = datetime.now()
    a = Assignment(task.id, (0,), now, now + task.estimated_duration)
    handle = runtime.submit(task, a, num_gpus=0)
    assert wait(runtime, handle) == RunState.FAILED


# the whole path: executor -> RayRuntime -> launcher -> the user's script --------------------
def test_executor_runs_user_scripts_on_ray(tmp_path, monkeypatch):
    import sys

    # Ray refuses more logical GPUs than CUDA_VISIBLE_DEVICES lists (e.g. under Slurm)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    from zgm.models import AssignmentState, LaunchSpec, TaskStatus
    from zgm.services.executor import Executor
    from zgm.services.scheduler import AssignmentService
    from zgm.services.task_store import TaskStore
    from zgm.storage.sqlite import SqliteStorage

    (tmp_path / "train.py").write_text(
        "import os, sys\n"
        "print('cwd', os.getcwd())\n"
        "print('cuda', os.environ.get('CUDA_VISIBLE_DEVICES'), 'zgm', os.environ['ZGM_GPU_IDS'])\n"
        "print('args', sys.argv[1:], os.environ['RUN_TAG'])\n"
        "sys.exit(int(sys.argv[1]))\n", encoding="utf-8")
    ray.shutdown()
    rt = RayRuntime(num_cpus=2, num_gpus=4, include_dashboard=False)  # 4 logical GPUs
    try:
        store = TaskStore(SqliteStorage(":memory:"))
        svc = AssignmentService(store, [0, 1, 2, 3])
        ex = Executor(store, svc, rt)
        tasks = {}
        for name, code, gpus in [("ok", 0, [0, 1]), ("bad", 3, [2])]:
            spec = LaunchSpec(cwd=str(tmp_path), python=sys.executable, script="train.py",
                              args=[str(code)], env={"RUN_TAG": "{name}"})
            t = store.add_task(Task(name, launch=spec, required_gpus=len(gpus)))
            svc.assign(t.id, gpus)
            tasks[name] = t
        assert sorted(t.name for t in ex.dispatch()) == ["bad", "ok"]
        deadline = time.time() + 60
        while ex.runs and time.time() < deadline:
            ex.poll()
            time.sleep(0.2)
        ok, bad = store.tasks[tasks["ok"].id], store.tasks[tasks["bad"].id]
        assert (ok.status, bad.status) == (TaskStatus.COMPLETED, TaskStatus.FAILED)
        (a,) = store.assignments_for_task(ok.id)
        assert a.state == AssignmentState.COMPLETED
        assert len(a.actual_gpu_ids) == 2  # what Ray actually gave it
    finally:
        ray.shutdown()


def test_executor_stops_a_ray_run(tmp_path):
    import sys

    from zgm.models import AssignmentState, LaunchSpec
    from zgm.runtime.base import RunState
    from zgm.services.executor import Executor
    from zgm.services.scheduler import AssignmentService
    from zgm.services.task_store import TaskStore
    from zgm.storage.sqlite import SqliteStorage

    (tmp_path / "long.py").write_text("import time\ntime.sleep(120)\n", encoding="utf-8")
    ray.shutdown()
    rt = RayRuntime(num_cpus=2, num_gpus=1, include_dashboard=False)
    try:
        store = TaskStore(SqliteStorage(":memory:"))
        svc = AssignmentService(store, [0])
        ex = Executor(store, svc, rt)
        t = store.add_task(Task("long", launch=LaunchSpec(cwd=str(tmp_path),
                                                          python=sys.executable,
                                                          script="long.py")))
        svc.assign(t.id, [0])
        ex.dispatch()
        (a,) = store.assignments_for_task(t.id)
        handle = ex.runs[a.id]
        time.sleep(2)
        assert ex.stop(t, requeue=True)
        assert wait(rt, handle) == RunState.CANCELLED
        assert a.state == AssignmentState.CANCELLED
        # the user's script must not outlive the stop (it would keep holding the GPU)
        import psutil

        def leftovers():
            found = []
            for p in psutil.process_iter():
                try:
                    if p.environ().get("ZGM_TASK_ID") == t.id:
                        found.append(p.pid)
                except (psutil.Error, OSError):
                    pass
            return found

        assert leftovers() == []
    finally:
        ray.shutdown()
