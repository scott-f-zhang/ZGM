"""Cluster mode's core, with a made-up backend and transport: groups -> GPUs,
hosts in placement, groups that go, remote runs, the TUI panel.

Nothing here is Slurm's (that is test_slurm_backend.py): ``FakeBackend`` and
``FakeTransport`` stand in for any source of hosts and any way to reach them.
The remote-run tests put a fake ``ssh`` on PATH that runs the script right
here, so the remote runner really runs.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from zgm.cluster import transport as transports
from zgm.cluster.base import Group
from zgm.cluster.core import ClusterProvider, ClusterWatch
from zgm.cluster.transport import SshTransport
from zgm.models import Assignment, GPUProcess, LaunchSpec, Task, TaskStatus
from zgm.runtime.base import RunState
from zgm.runtime.local import LocalRuntime, Placement, read_run, run_state
from zgm.runtime.runner import write_json
from zgm.services.gpu_discovery import MockGPUProvider
from zgm.services.gpu_usage import external
from zgm.services.scheduler import AssignmentService
from zgm.services.task_store import TaskStore
from zgm.storage.sqlite import SqliteStorage

PY = sys.executable
ROOT = Path(__file__).parent.parent
pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="remote hosts run POSIX shells")

PROBES: dict[str, dict] = {}  # host -> what zgm.probe would print there


def probe_of(n: int, host: str, procs: dict | None = None) -> dict:
    return {"host": host, "gpus": [
        {"index": i, "uuid": f"GPU-{host}-{i}", "name": "NVIDIA Test GPU 24GB",
         "memory_total": 24 * 2**30, "memory_used": 2**30, "utilization": 5,
         "processes_visible": True, "processes": (procs or {}).get(i, [])}
        for i in range(n)]}


class FakeTransport:
    """Answers ``zgm.probe`` from PROBES; anything else succeeds."""

    def __init__(self, host: str) -> None:
        self.host = host

    def run(self, script, timeout=60):
        if "zgm.probe" not in script:
            return subprocess.CompletedProcess(script, 0, "", "")
        if self.host not in PROBES:
            return subprocess.CompletedProcess(script, 255, "", "Permission denied (publickey)")
        return subprocess.CompletedProcess(script, 0, json.dumps(PROBES[self.host]), "")


class FakeBackend:
    name = "fake"
    interval = 0.0
    host_env = ("FAKE_",)

    def __init__(self) -> None:
        self.listed: list[Group] = []

    def groups(self):
        return list(self.listed)

    def local_key(self):
        return "here"

    def title(self, key):
        return f"group {key}"

    def columns(self, row, local):
        return [row["key"], row["state"], row["host"] or "-", "this one" if local else ""]

    def order(self, row):
        return (row["key"],)


def group(key, host=None, usable=True, note=""):
    return Group(key, host if host is not None else key, usable, note, transport="fake")


@pytest.fixture
def cluster(monkeypatch):
    """(store, svc, watch, provider, backend): this process runs in group "here",
    with one mock GPU."""
    monkeypatch.setitem(transports.TRANSPORTS, "fake", FakeTransport)
    PROBES.clear()
    store = TaskStore(SqliteStorage(":memory:"))
    store.storage.set_setting("cluster", "fake")
    local = MockGPUProvider(1)
    backend = FakeBackend()
    watch = ClusterWatch(store, backend, local)
    provider = ClusterProvider(store.storage, backend, local)
    svc = AssignmentService(store, [])
    yield store, svc, watch, provider, backend
    watch.close()


def update(store, svc, watch, provider, gpus):
    """One executor tick, with listing and probes due."""
    watch._probed.clear()
    messages = [m for m, _ in watch.refresh(svc, wait=True)]
    provider.invalidate()
    provider.sync(gpus)
    svc.set_gpus([g.id for g in gpus], provider.node_of(gpus))
    return messages


# groups -> GPU ids ------------------------------------------------------------------------
def test_usable_groups_become_gpus(cluster):
    store, svc, watch, provider, backend = cluster
    backend.listed = [group("here"), group("b"), group("c", host="", usable=False)]
    PROBES["b"] = probe_of(4, "b")
    gpus = []
    messages = update(store, svc, watch, provider, gpus)
    assert [(g.id, g.host, g.local_index) for g in gpus] == [
        (0, "here", 0), (1, "b", 0), (2, "b", 1), (3, "b", 2), (4, "b", 3)]
    assert any("b (group b): 4 x Test GPU 24GB added as GPU 1-4" in m for m in messages)
    assert provider.locate([0]) == Placement("here", "here", [0], None, ("FAKE_",))
    far = provider.locate([3, 4])
    assert (far.group, far.visible, far.remote, far.transport.host) == ("b", [2, 3], True, "b")
    with pytest.raises(ValueError, match="one host"):
        provider.locate([0, 1])
    lines = {line.split()[0]: line for line in provider.lines()}
    assert "this one" in lines["here"] and "GPU 0" in lines["here"]
    assert "GPU 1-4" in lines["b"] and "down" in lines["c"]
    assert "cluster: fake" in provider.describe(gpus)


def test_tasks_stay_on_one_host(cluster):
    store, svc, watch, provider, backend = cluster
    backend.listed = [group("here"), group("b")]
    PROBES["b"] = probe_of(2, "b")
    update(store, svc, watch, provider, [])
    pair = store.add_task(Task("pair", "echo", 2))
    assert svc.best_gpus(pair) == (1, 2)
    assert not svc.assign(pair.id, [0, 1]).ok  # GPU 0 is on another host
    assert svc.suggest_gpus(pair, 0) == (0,)  # the anchor's host has only one
    with pytest.raises(ValueError, match="no node has 3"):
        svc.best_gpus(store.add_task(Task("three", "echo", 3)))


def test_a_group_that_goes_takes_its_queue_back(cluster):
    store, svc, watch, provider, backend = cluster
    backend.listed = [group("here"), group("b")]
    PROBES["b"] = probe_of(4, "b")
    update(store, svc, watch, provider, gpus := [])
    queued = store.add_task(Task("queued", "echo"))
    running = store.add_task(Task("running", "echo"))
    assert svc.assign(queued.id, [3]).ok and svc.assign(running.id, [4]).ok
    a = store.active_assignment(running.id)
    a.state, a.actual_start = a.state.RUNNING, datetime.now()  # as the executor would
    store.save_assignment(a, "started")

    backend.listed = [group("here")]  # b went away
    messages = update(store, svc, watch, provider, gpus)
    assert store.tasks[queued.id].status == TaskStatus.PENDING
    assert any("queued: back to the waiting queue" in m for m in messages)
    assert [g.id for g in gpus] == [0]
    assert 4 in svc.unavailable()  # still running there: nothing new starts on it

    a.state, a.actual_end = a.state.FAILED, datetime.now()  # its run is gone
    store.save_assignment(a, "lost")
    messages = update(store, svc, watch, provider, gpus)
    assert any("ended: GPU 1-4 removed" in m for m in messages)
    backend.listed.append(group("e"))
    PROBES["e"] = probe_of(2, "e")
    update(store, svc, watch, provider, gpus)
    assert [(g.id, g.host) for g in gpus] == [(0, "here"), (1, "e"), (2, "e")]  # ids reused


def test_unusable_and_unreachable_groups(cluster):
    store, svc, watch, provider, backend = cluster
    backend.listed = [group("here"), group("b", usable=False, note="not for us: not used"),
                      group("f")]
    messages = update(store, svc, watch, provider, gpus := [])
    assert [g.id for g in gpus] == [0]
    rows = {r["key"]: r for r in store.storage.gpu_groups("fake")}
    assert rows["b"]["state"] == "down" and rows["b"]["note"] == "not for us: not used"
    assert rows["f"]["problem"].startswith("probe failed: f: Permission denied")
    assert any("can't read its GPUs" in m and "zgm check" in m for m in messages)
    assert any(line.endswith("probe failed: f: Permission denied (publickey)")
               for line in provider.lines())


def test_remote_processes_are_judged_by_their_task_id(cluster):
    store, svc, watch, provider, backend = cluster
    mine = store.add_task(Task("mine", "echo"))
    backend.listed = [group("b")]
    PROBES["b"] = probe_of(1, "b", {0: [
        {"pid": 10, "name": "python", "used_memory": 2**30, "zgm_task": mine.id},
        {"pid": 11, "name": "other", "used_memory": 2**30, "zgm_task": None}]})
    update(store, svc, watch, provider, gpus := [])
    procs = provider.processes(gpus[0].id)
    assert {p.host for p in procs} == {"b"}
    assert [p.pid for p in external(procs, own_pids={10, 11}, running_tasks={mine.id})] == [11]


def test_processes_from_a_stale_probe_are_unknown(cluster):
    store, svc, watch, provider, backend = cluster
    backend.listed = [group("b")]
    PROBES["b"] = probe_of(1, "b")
    update(store, svc, watch, provider, gpus := [])
    row = store.storage.gpu_groups("fake")[0]
    row["probed_at"] = (datetime.now() - timedelta(minutes=5)).isoformat()
    store.storage.save_gpu_groups([row])
    provider.invalidate()
    with pytest.raises(RuntimeError, match="no recent probe"):
        provider.processes(gpus[0].id)


def test_external_keeps_local_rules_for_local_processes():
    procs = [GPUProcess(os.getpid(), "me")]
    assert external(procs, own_pids={os.getpid()}, running_tasks=set()) == []


# remote runs over ssh ----------------------------------------------------------------------
FAKE_SSH = """#!/bin/sh
# fake ssh: skip the options and the host, run the script here
while [ "$1" = "-o" ]; do shift 2; done
shift
exec sh -c "$1"
"""


@pytest.fixture
def fake_bin(tmp_path, monkeypatch):
    d = tmp_path / "bin"
    d.mkdir()
    (d / "ssh").write_text(FAKE_SSH)
    (d / "ssh").chmod(0o755)
    monkeypatch.setenv("PATH", f"{d}{os.pathsep}{os.environ['PATH']}")
    return d


JOB = """import os, sys, time
print("cvd", os.environ["CUDA_VISIBLE_DEVICES"], "zgm", os.environ["ZGM_GPU_IDS"],
      "fake", sorted(k for k in os.environ if k.startswith("FAKE_")), flush=True)
time.sleep(float(sys.argv[1]))
"""


def remote_runtime(tmp_path):
    return LocalRuntime(tmp_path / "logs", locate=lambda ids: Placement(
        "b", "b", [i - 3 for i in ids], SshTransport("b"), ("FAKE_",)))


def test_remote_run_heartbeat_and_stop(tmp_path, fake_bin, monkeypatch):
    monkeypatch.setenv("FAKE_GROUP", "here")  # describes zgm's group: must not reach the task
    (tmp_path / "job.py").write_text(JOB)
    rt = remote_runtime(tmp_path)
    t = Task("far", launch=LaunchSpec(cwd=str(tmp_path), python=PY, script="job.py",
                                      args=["60"]))
    a = Assignment(t.id, (4, 5), datetime.now(), None)
    handle = rt.submit(t, a)
    deadline = time.time() + 20
    while "cvd" not in rt.log_path(a.id).read_text() and time.time() < deadline:
        time.sleep(0.2)
    info = read_run(handle.ref)
    assert info["remote"] and info["host"] and rt.status(handle) == RunState.RUNNING
    assert "cvd 1,2 zgm 4,5 fake []" in rt.log_path(a.id).read_text()
    rt.cancel(handle)
    assert rt.status(handle) == RunState.CANCELLED
    info = read_run(handle.ref)
    assert info["exit_code"] != 0
    log = rt.log_path(a.id).read_text()
    assert "# zgm: stopped" in log and "exited with code" in log
    with pytest.raises(OSError):
        os.kill(info["child_pid"], 0)


def test_remote_run_completes(tmp_path, fake_bin):
    (tmp_path / "job.py").write_text(JOB)
    rt = remote_runtime(tmp_path)
    t = Task("far", launch=LaunchSpec(cwd=str(tmp_path), python=PY, script="job.py",
                                      args=["0"]))
    handle = rt.submit(t, Assignment(t.id, (3,), datetime.now(), None))
    deadline = time.time() + 20
    while rt.status(handle) == RunState.RUNNING and time.time() < deadline:
        time.sleep(0.2)
    assert rt.status(handle) == RunState.SUCCEEDED
    assert rt.result(handle)["returncode"] == 0


def test_remote_run_without_heartbeat_is_lost(tmp_path):
    base = tmp_path / "abc"
    old = (datetime.now() - timedelta(minutes=5)).isoformat(timespec="seconds")
    write_json(tmp_path / "abc.run.json", {"remote": True, "runner_pid": 1, "heartbeat": old})
    assert run_state(base) == RunState.FAILED
    write_json(tmp_path / "abc.run.json", {"remote": True, "runner_pid": 1,
                                           "heartbeat": datetime.now().isoformat()})
    assert run_state(base) == RunState.RUNNING


def test_unreachable_host_is_a_launch_error(tmp_path, monkeypatch):
    d = tmp_path / "bin"
    d.mkdir()
    (d / "ssh").write_text("#!/bin/sh\necho 'Permission denied (publickey)' >&2\nexit 255\n")
    (d / "ssh").chmod(0o755)
    monkeypatch.setenv("PATH", f"{d}{os.pathsep}{os.environ['PATH']}")
    rt = remote_runtime(tmp_path)
    t = Task("far", launch=LaunchSpec(cwd=str(tmp_path), command="true"))
    with pytest.raises(ValueError, match="b: could not start the runner: Permission denied"):
        rt.submit(t, Assignment(t.id, (3,), datetime.now(), None))


def test_probe_command_reports_json():
    out = subprocess.run([PY, "-m", "zgm.probe"], capture_output=True, text=True, timeout=60,
                         env={**os.environ, "ZGM_MOCK_GPUS": "2"})
    data = json.loads(out.stdout)
    assert out.returncode == 0 and len(data["gpus"]) == 2 and data["host"]
    assert {"index", "uuid", "processes", "processes_visible"} <= set(data["gpus"][0])


# the TUI -----------------------------------------------------------------------------------
def group_row(key, host, base=None, probe=None, state="up"):
    now = datetime.now().isoformat(timespec="seconds")
    return {"backend": "fake", "key": key, "state": state, "host": host, "transport": "fake",
            "note": "", "problem": "", "info": {}, "gpu_base": base, "probe": probe,
            "probed_at": now if probe else None, "seen_at": now}


async def test_tui_shows_groups_and_follows_new_gpus(tmp_path, monkeypatch):
    from zgm.app import ZGMApp
    from zgm.widgets.cluster_panel import ClusterPanel
    from zgm.widgets.gpu_row import GPURow

    monkeypatch.setattr("zgm.cluster.core.load_backend", lambda name: FakeBackend())
    db = tmp_path / "s.db"
    other = SqliteStorage(db)  # stands in for the daemon
    other.set_setting("cluster", "fake")
    other.save_gpu_groups([group_row("here", "here", 0, probe_of(1, "here")),
                           group_row("later", "", state="down")])
    app = ZGMApp(storage=SqliteStorage(db), mock_gpus=1, stats_interval=0)
    async with app.run_test(size=(140, 46)) as pilot:
        await pilot.pause()
        panel = app.screen.query_one(ClusterPanel)
        assert panel.display and "fake" in str(panel.border_title)
        text = str(panel.render())
        assert "this one" in text and "later" in text
        assert len(app.screen.query(GPURow)) == 1
        assert app.gpu_source.startswith("fake · ")

        other.save_gpu_groups([group_row("b", "b", 1, probe_of(2, "b"))])
        app.tick()
        await pilot.pause()
        await pilot.pause()
        rows = list(app.screen.query(GPURow))
        assert [r.gpu.id for r in rows] == [0, 1, 2]
        assert "gpu1 b" in rows[1].render_line(0).text
        assert app.assignment_service.node_of == {0: "here", 1: "b", 2: "b"}
