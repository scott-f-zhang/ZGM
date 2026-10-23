"""GPUs used by programs outside zgm: NVML's process lists, told apart from
zgm's own task processes, keep new work off those GPUs."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
import types
from datetime import datetime
from pathlib import Path

import psutil
import pytest

from conftest import make_task
from test_dispatch import FakeRuntime
from zgm.app import ZGMApp
from zgm.models import Assignment, AssignmentState, GPUDevice, GPUProcess, LaunchSpec, Task
from zgm.runtime.local import LocalRuntime, read_run
from zgm.services import gpu_discovery, launcher
from zgm.services.gpu_usage import GPUWatch, zgm_pids
from zgm.storage.sqlite import SqliteStorage

PY = sys.executable
EXAMPLES = Path(__file__).parent.parent / "examples" / "configs"


def gpus(n=4) -> list[GPUDevice]:
    return [GPUDevice(i, f"GPU-{i}", "Test GPU", 24 * 1024**3) for i in range(n)]


class Provider:
    """Stands in for NVML: which processes are on which GPU."""

    def __init__(self, procs=None, broken=()):
        self.procs: dict[int, list[GPUProcess]] = procs or {}
        self.broken = set(broken)

    def processes(self, gpu_id):
        if gpu_id in self.broken:
            raise RuntimeError("Not Supported")
        return list(self.procs.get(gpu_id, []))


def sleeper(env=None) -> subprocess.Popen:
    return subprocess.Popen([PY, "-c", "import time; time.sleep(60)"],
                            env={**os.environ, **(env or {})})


# NVML ----------------------------------------------------------------------------------
def fake_pynvml(compute, mps=None, compute_error=None):
    class NVMLError(Exception):
        pass

    class NVMLError_NotSupported(NVMLError):
        pass

    def proc(pid, mem):
        return types.SimpleNamespace(pid=pid, usedGpuMemory=mem)

    def get_compute(h):
        if compute_error:
            raise compute_error(NVMLError_NotSupported)
        return [proc(*p) for p in compute.get(h, [])]

    def get_mps(h):
        if mps is None:
            raise NVMLError_NotSupported()
        return [proc(*p) for p in mps.get(h, [])]

    return types.SimpleNamespace(
        NVMLError=NVMLError, NVMLError_NotSupported=NVMLError_NotSupported,
        nvmlInit=lambda: None, nvmlDeviceGetHandleByIndex=lambda i: i,
        nvmlDeviceGetComputeRunningProcesses=get_compute,
        nvmlDeviceGetMPSComputeRunningProcesses=get_mps,
    )


def nvml_provider(monkeypatch, nv) -> gpu_discovery.NVMLProvider:
    monkeypatch.setitem(sys.modules, "pynvml", nv)
    return gpu_discovery.NVMLProvider()


def test_nvml_lists_compute_and_mps_processes(monkeypatch):
    me = os.getpid()
    nv = fake_pynvml(compute={0: [(me, 2 * 1024**3), (4242, None)]},  # None: WDDM
                     mps={0: [(me, 3 * 1024**3)], 1: [(5151, 1024**2)]})
    p = nvml_provider(monkeypatch, nv)
    procs = {x.pid: x for x in p.processes(0)}
    assert set(procs) == {me, 4242}
    assert procs[me].used_memory == 3 * 1024**3  # same pid on both lists: larger one
    assert procs[me].name == psutil.Process(me).name()
    assert procs[4242].used_memory is None
    assert [x.pid for x in p.processes(1)] == [5151]


def test_nvml_without_mps_still_lists_compute(monkeypatch):
    p = nvml_provider(monkeypatch, fake_pynvml(compute={0: [(1234, 10)]}, mps=None))
    assert [x.pid for x in p.processes(0)] == [1234]


def test_nvml_that_cannot_list_processes_raises(monkeypatch):
    p = nvml_provider(monkeypatch, fake_pynvml({}, compute_error=lambda e: e()))
    with pytest.raises(Exception):
        p.processes(0)


# telling zgm's processes from everyone else's ------------------------------------------------
TREE = ("import subprocess, sys, time\n"
        "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        "time.sleep(60)\n")


@pytest.fixture
def running_task(tmp_path, store):
    """A real zgm run: runner -> task -> a grandchild (like a data loader worker)."""
    (tmp_path / "tree.py").write_text(TREE, encoding="utf-8")
    t = store.add_task(Task("tree", launch=LaunchSpec(cwd=str(tmp_path), python=PY,
                                                      script="tree.py")))
    a = Assignment(t.id, (0,), None, None)
    rt = LocalRuntime(tmp_path / "logs")
    handle = rt.submit(t, a)
    a.state, a.actual_start = AssignmentState.RUNNING, datetime.now()
    store.save_assignment(a, "started")
    child = read_run(handle.ref)["child_pid"]
    deadline = time.monotonic() + 15
    while not psutil.Process(child).children() and time.monotonic() < deadline:
        time.sleep(0.05)
    grandchild = psutil.Process(child).children()[0].pid
    yield t, read_run(handle.ref)["runner_pid"], child, grandchild
    rt.cancel(handle)


def test_zgm_task_processes_are_not_external(store, running_task):
    task, runner, child, grandchild = running_task
    assert {runner, child, grandchild} <= zgm_pids(store)
    foreign = sleeper()
    try:
        devices = gpus(2)
        provider = Provider({0: [GPUProcess(child, used_memory=1), GPUProcess(grandchild),
                                 GPUProcess(foreign.pid, "python", 5 * 1024**3)]})
        busy, changes = GPUWatch(provider, devices).refresh(store)
        assert busy == {0}
        assert [p.pid for p in devices[0].external] == [foreign.pid]
        assert "GPU 0 is in use outside zgm" in changes[0]
    finally:
        foreign.kill()


def test_task_id_in_the_environment_marks_zgm_processes(store, running_task):
    """A task's process whose parent exited (so it is outside the tree) is still
    zgm's; the same variable for a task that isn't running doesn't count."""
    task, *_ = running_task
    ours, stale = sleeper({"ZGM_TASK_ID": task.id}), sleeper({"ZGM_TASK_ID": "finished1"})
    try:
        devices = gpus(1)
        provider = Provider({0: [GPUProcess(ours.pid), GPUProcess(stale.pid)]})
        GPUWatch(provider, devices).refresh(store)
        assert [p.pid for p in devices[0].external] == [stale.pid]
    finally:
        ours.kill()
        stale.kill()


def test_short_lived_outside_use_blocks_but_is_reported_only_if_it_lasts(store):
    """An exiting process that no longer tells its task looks like another
    program for a few seconds: block the GPU, but say so only if it stays."""
    clock = [0.0]
    devices = gpus(1)
    watch = GPUWatch(Provider(), devices, report_after=30, clock=lambda: clock[0])
    assert watch.refresh(store) == (set(), [])
    other = sleeper()
    try:
        watch.provider.procs = {0: [GPUProcess(other.pid)]}
        clock[0] = 5
        assert watch.refresh(store) == ({0}, [])  # blocked, not reported yet
        watch.provider.procs = {}
        clock[0] = 10
        assert watch.refresh(store) == (set(), [])  # gone: nothing to say

        watch.provider.procs = {0: [GPUProcess(other.pid)]}
        clock[0] = 20
        watch.refresh(store)
        clock[0] = 55
        busy, changes = watch.refresh(store)
        assert busy == {0} and "GPU 0 is in use outside zgm" in changes[0]
        watch.provider.procs = {}
        assert watch.refresh(store)[1] == ["GPU 0 is free of other programs again"]
    finally:
        other.kill()


def test_an_ended_tasks_exiting_processes_block_quietly(store):
    """Right after a run ends its vLLM engine may still hold the GPU for a few
    seconds: the GPU waits, but that is not reported as another program."""
    done = store.add_task(Task("done", "echo"))
    leftover = sleeper({"ZGM_TASK_ID": done.id})
    try:
        devices = gpus(1)
        watch = GPUWatch(Provider({0: [GPUProcess(leftover.pid)]}), devices)
        busy, changes = watch.refresh(store)
        assert busy == {0} and changes == []
        watch.provider.procs = {}
        busy, changes = watch.refresh(store)
        assert busy == set() and changes == []
    finally:
        leftover.kill()


def test_gpu_whose_processes_cannot_be_listed_is_reported_not_blocked(store):
    devices = gpus(2)
    busy, changes = GPUWatch(Provider(broken={1}), devices).refresh(store)
    assert busy == set()
    assert not devices[1].processes_visible
    assert "can't list its processes" in changes[0]


def test_changes_are_reported_once(store):
    devices = gpus(1)
    provider = Provider({0: [GPUProcess(4321, "game")]})
    watch = GPUWatch(provider, devices)
    assert len(watch.refresh(store)[1]) == 1
    assert watch.refresh(store)[1] == []  # still busy: nothing new to say
    provider.procs = {}
    busy, changes = watch.refresh(store)
    assert busy == set() and "free of other programs" in changes[0]


# the scheduler keeps new work off those GPUs ---------------------------------------------------
def names(svc, assignments):
    return [svc.store.tasks[a.task_id].name for a in assignments]


def test_blocked_gpu_holds_its_queue_in_order(store, svc):
    svc.assign(make_task(store, "a").id, [0])
    svc.assign(make_task(store, "b", gpus=2).id, [0, 1])
    svc.assign(make_task(store, "c").id, [1])
    svc.assign(make_task(store, "d").id, [2])
    svc.blocked = {0}
    # a waits for GPU 0; b waits too, and c (queued after b on GPU 1) waits for b
    assert names(svc, svc.ready_to_start()) == ["d"]
    slots = {svc.store.tasks[s.task_id].name: s for s in svc.schedule()}
    assert slots["a"].start is None and slots["d"].start is not None
    svc.blocked = set()
    assert names(svc, svc.ready_to_start()) == ["a", "d"]


def test_best_gpus_avoids_blocked(store, svc):
    svc.blocked = {0, 1}
    assert svc.best_gpus(make_task(store, "new")) in {(2,), (3,)}
    assert svc.best_gpus(make_task(store, "pair", gpus=2)) == (2, 3)


def test_launcher_numbers_gpus_like_nvidia_smi(tmp_path):
    t = Task("job", launch=LaunchSpec(cwd=str(tmp_path), command="echo"))
    env = launcher.build_process(t, (2,)).env
    assert env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID" and env["CUDA_VISIBLE_DEVICES"] == "2"


# end to end: the TUI with a GPU taken by another program ----------------------------------------
async def test_tui_waits_for_a_gpu_used_elsewhere(tmp_path, monkeypatch):
    config_dir = tmp_path / "configs"
    shutil.copytree(EXAMPLES, config_dir)
    monkeypatch.setenv("ZGM_MOCK_EXTERNAL", "1:9000:4321")  # 9000 MiB on GPU 1
    rt = FakeRuntime()
    app = ZGMApp(storage=SqliteStorage(":memory:"), mock_gpus=4, config_dir=config_dir,
                 stats_interval=0, runtime=rt)
    async with app.run_test(size=(140, 42)) as pilot:
        await pilot.pause()
        assert app.assignment_service.blocked == {1}
        a = app.store.add_task(Task("a", "echo a"))
        b = app.store.add_task(Task("b", "echo b"))
        app.assignment_service.assign(a.id, [1])
        app.assignment_service.assign(b.id, [2])
        app.tick()
        await pilot.pause()
        assert rt.started == ["b"]  # a waits for GPU 1
        message, _ = app.executor.request_run_now(a)
        assert "in use outside zgm" in message
        app.tick()
        assert rt.started == ["b"]  # not even r overrides it
        row = next(r for r in app.screen.query("GPURow") if r.gpu.id == 1)
        label = "".join(seg.text for seg in row._label_line(3))
        assert "other: 1 proc 8.8G" in label
        app.provider.fake_processes.clear()  # the other program exits
        app.tick()
        await pilot.pause()
        assert rt.started == ["b", "a"]


def test_a_process_with_a_blanked_environment_takes_its_parents_task(monkeypatch):
    """vLLM's EngineCore sets its process title, which blanks /proc/PID/environ:
    the task comes from the nearest ancestor that has it."""
    import psutil

    from zgm.probe import task_of

    parent, child = sleeper({"ZGM_TASK_ID": "task1"}), sleeper()
    up = {child.pid: parent.pid}
    try:
        real = psutil.Process.environ
        monkeypatch.setattr(psutil.Process, "environ",
                            lambda self: {} if self.pid == child.pid else real(self))
        monkeypatch.setattr(psutil.Process, "parent",
                            lambda self: psutil.Process(up[self.pid]) if self.pid in up else None)
        assert task_of(child.pid) == "task1"
        up.clear()
        assert task_of(child.pid) is None
    finally:
        parent.kill()
        child.kill()
