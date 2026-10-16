"""Running the queue in the background: detached runs, the executor lease,
clients acting on the executor's runs, and the zgm daemon command."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from zgm import daemon
from zgm.app import ZGMApp
from zgm.models import Assignment, AssignmentState, LaunchSpec, Task, TaskStatus
from zgm.runtime.base import RunState
from zgm.runtime.local import LocalRuntime, read_run, run_state
from zgm.services.executor import Executor
from zgm.services.filelock import FileLock, is_locked
from zgm.services.lease import ExecutorLease
from zgm.services.scheduler import AssignmentService
from zgm.services.task_store import TaskStore
from zgm.storage.sqlite import SqliteStorage

PY = sys.executable
ROOT = Path(__file__).parent.parent
EXAMPLES = ROOT / "examples" / "configs"


def wait_for(condition, timeout=30.0, step=0.05):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(step)
    return False


def sleeper(tmp_path: Path, seconds: float, code: int = 0) -> LaunchSpec:
    script = tmp_path / "job.py"
    script.write_text("import sys, time\nprint('working', flush=True)\n"
                      "time.sleep(float(sys.argv[1]))\nsys.exit(int(sys.argv[2]))\n",
                      encoding="utf-8")
    return LaunchSpec(cwd=str(tmp_path), python=PY, script="job.py",
                      args=[str(seconds), str(code)])


class Zgm:
    """One zgm process's view of a database: its own connection and executor."""

    def __init__(self, db: Path, log_dir: Path | None = None) -> None:
        self.store = TaskStore(SqliteStorage(db))
        self.svc = AssignmentService(self.store, [0, 1, 2, 3])
        runtime = LocalRuntime(log_dir) if log_dir else None
        self.executor = Executor(self.store, self.svc, runtime)

    def tick(self):
        self.store.sync()
        return self.executor.tick()

    def queue(self, name: str, spec: LaunchSpec, gpus=(0,)) -> Task:
        t = self.store.add_task(Task(name, launch=spec, required_gpus=len(gpus)))
        assert self.svc.assign(t.id, gpus).ok
        return t

    def state(self, task: Task) -> tuple[TaskStatus, AssignmentState | None]:
        self.store.sync()
        t = self.store.tasks.get(task.id)
        runs = self.store.assignments_for_task(task.id)
        return (t.status if t else None), (runs[-1].state if runs else None)


# locks and the lease --------------------------------------------------------------------
# holds the lock until stdin closes, then dies without releasing it (the OS must)
HOLD = ("import os, sys; from zgm.services.filelock import FileLock\n"
        "lock = FileLock(sys.argv[1]); lock.acquire(); print('held', flush=True)\n"
        "sys.stdin.read(); os._exit(0)\n")


def test_file_lock_across_processes(tmp_path):
    path = tmp_path / "x.lock"
    proc = subprocess.Popen([PY, "-c", HOLD, str(path)], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, text=True, cwd=ROOT)
    try:
        assert proc.stdout.readline().strip() == "held"
        assert is_locked(path)
        assert not FileLock(path).acquire()
    finally:
        proc.stdin.close()
        proc.wait(timeout=10)
    assert wait_for(lambda: not is_locked(path), 5)  # the OS dropped it with the process
    lock = FileLock(path)
    assert lock.acquire()
    assert is_locked(path)  # also seen from a second open in the same process
    lock.release()
    assert not is_locked(path)


def test_lease_one_executor_at_a_time(tmp_path):
    db = tmp_path / "s.db"
    first, second = ExecutorLease(db), ExecutorLease(db)
    assert first.acquire("daemon", "local", "/logs")
    assert not second.acquire("tui", "local")
    assert second.holder()["role"] == "daemon" and second.holder()["log_dir"] == "/logs"
    first.release()
    assert second.holder() is None
    assert second.acquire("tui", "local")
    assert ExecutorLease(":memory:").acquire("tui", "mock")  # private db: always yours


# runs outlive whoever started them ---------------------------------------------------------
STARTER = """
import sys
from datetime import datetime
from zgm.models import Assignment, LaunchSpec, Task
from zgm.runtime.local import LocalRuntime
spec = LaunchSpec(cwd=sys.argv[2], python=sys.executable, script="job.py", args=["1.5", "0"])
task = Task("detached", launch=spec)
a = Assignment(task.id, (1,), datetime.now(), None)
LocalRuntime(sys.argv[1]).submit(task, a)
print(a.id, a.log_path, flush=True)
"""


def test_run_outlives_the_process_that_started_it(tmp_path):
    sleeper(tmp_path, 1.5)
    out = subprocess.run([PY, "-c", STARTER, str(tmp_path / "logs"), str(tmp_path)],
                         capture_output=True, text=True, check=True, cwd=ROOT).stdout.split()
    aid, log_path = out[0], out[1]
    # the starter has exited; a different runtime instance picks the run up from its files
    a = Assignment("t", (1,), None, None, id=aid, state=AssignmentState.RUNNING,
                   log_path=log_path)
    rt = LocalRuntime(tmp_path / "elsewhere")
    handle = rt.attach(a)
    assert handle is not None and rt.status(handle) == RunState.RUNNING
    assert wait_for(lambda: rt.status(handle) == RunState.SUCCEEDED)
    result = rt.result(handle)
    assert result["gpu_ids"] == [1] and result["returncode"] == 0
    assert "working" in Path(log_path).read_text(encoding="utf-8")
    assert not Path(log_path).with_suffix(".spec.json").exists()  # env not left on disk


def test_run_state_from_files(tmp_path):
    base = tmp_path / "abc"
    (tmp_path / "abc.run.json").write_text(json.dumps({"runner_pid": 1}), encoding="utf-8")
    assert run_state(base) == RunState.FAILED  # no exit, nobody holds the lock: lost
    (tmp_path / "abc.cancel").write_text("", encoding="utf-8")
    assert run_state(base) == RunState.CANCELLED
    (tmp_path / "abc.run.json").write_text(json.dumps({"exit_code": 0}), encoding="utf-8")
    assert run_state(base) == RunState.SUCCEEDED  # it finished before the stop landed


# the executor: adopt, follow, record ---------------------------------------------------------
def test_executor_adopts_runs_and_records_their_real_end(tmp_path):
    db, logs = tmp_path / "s.db", tmp_path / "logs"
    first = Zgm(db, logs)
    t = first.queue("job", sleeper(tmp_path, 0.5))
    assert [x.name for x in first.tick()] == ["job"]
    del first  # this executor goes away while the task runs
    time.sleep(1.5)  # it ends while nobody watches
    second = Zgm(db, logs)
    second.tick()
    assert second.state(t) == (TaskStatus.COMPLETED, AssignmentState.COMPLETED)
    (a,) = second.store.assignments_for_task(t.id)
    ended = datetime.fromisoformat(read_run(Path(a.log_path).with_suffix(""))["ended"])
    assert a.actual_end == ended  # when it ended, not when it was noticed
    assert (a.actual_end - a.actual_start).total_seconds() < 1.4


def test_executor_marks_runs_without_a_trace_lost(tmp_path):
    z = Zgm(tmp_path / "s.db", tmp_path / "logs")
    t = z.store.add_task(Task("ghost", "echo"))
    a = Assignment(t.id, (2,), None, None, state=AssignmentState.RUNNING,
                   actual_start=datetime.now())
    z.store.save_assignment(a, "started")
    t.status = TaskStatus.RUNNING
    z.store.update_task(t)
    z.tick()
    assert z.state(t) == (TaskStatus.FAILED, AssignmentState.FAILED)
    assert "lost" in [e["event"] for e in z.store.storage.history(t.id)]
    assert 2 not in z.svc.busy_gpus()


# clients act through the database ----------------------------------------------------------
def test_client_stops_a_run_of_the_executor(tmp_path):
    db, logs = tmp_path / "s.db", tmp_path / "logs"
    executor, client = Zgm(db, logs), Zgm(db, logs)
    t = executor.queue("long", sleeper(tmp_path, 60))
    executor.tick()
    client.store.sync()
    assert client.executor.stop(client.store.tasks[t.id], requeue=True)
    executor.tick()
    executor.tick()
    assert executor.state(t) == (TaskStatus.PENDING, AssignmentState.CANCELLED)
    (a,) = executor.store.assignments_for_task(t.id)
    assert run_state(Path(a.log_path).with_suffix("")) == RunState.CANCELLED
    assert not executor.executor.runs


def test_client_run_now_and_pause(tmp_path):
    db, logs = tmp_path / "s.db", tmp_path / "logs"
    executor, client = Zgm(db, logs), Zgm(db, logs)
    client.executor.autostart = False  # p, from a client
    a = executor.queue("a", sleeper(tmp_path, 0.2))
    b = executor.queue("b", sleeper(tmp_path, 0.2), gpus=(1,))
    assert executor.tick() == []
    client.store.sync()
    message, _ = client.executor.request_run_now(client.store.tasks[b.id])
    assert "Starting b" in message
    assert [x.name for x in executor.tick()] == ["b"]  # r works while paused
    assert executor.state(a)[0] == TaskStatus.PLANNED


# the TUI: executor or client ---------------------------------------------------------------
@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    d = tmp_path / "configs"
    shutil.copytree(EXAMPLES, d)
    return d


async def test_tui_is_a_client_while_a_daemon_runs_then_takes_over(tmp_path, config_dir):
    db = tmp_path / "s.db"
    other = ExecutorLease(db)
    assert other.acquire("daemon", "local", str(tmp_path / "logs"))
    app = ZGMApp(storage=SqliteStorage(db), mock_gpus=4, config_dir=config_dir,
                 stats_interval=0, runtime=LocalRuntime(tmp_path / "logs"))
    async with app.run_test(size=(140, 42)) as pilot:
        await pilot.pause()
        assert not app.is_executor and app.can_control_runs
        status = app.screen.query_one("#status-line")
        assert "run by zgm daemon" in str(status.render())
        other.release()  # the daemon exits
        app.tick()
        await pilot.pause()
        assert app.is_executor
        assert "local runtime" in str(status.render())
    assert not ExecutorLease(db).holder()  # released when the TUI closed


async def test_tui_without_runtime_plans_only(tmp_path, config_dir):
    app = ZGMApp(storage=SqliteStorage(tmp_path / "s.db"), mock_gpus=4,
                 config_dir=config_dir, stats_interval=0)
    async with app.run_test(size=(140, 42)) as pilot:
        await pilot.pause()
        assert not app.can_control_runs
        assert "PLAN ONLY" in str(app.screen.query_one("#status-line").render())


# zgm daemon ------------------------------------------------------------------------------
def daemon_args(db: Path, **kw):
    base = dict(db=str(db), runtime="local", log_dir=None, mock_gpus=4, interval=0.2,
                ray_address=None,
                autostart=None)
    return SimpleNamespace(**{**base, **kw})


def test_daemon_runs_the_queue_and_hands_over(tmp_path, capsys):
    db = tmp_path / "s.db"
    proc = subprocess.Popen([PY, "-m", "zgm", "daemon", "run", "--db", str(db),
                             "--mock-gpus", "4", "--interval", "0.2"], cwd=ROOT,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        assert wait_for(lambda: (ExecutorLease(db).holder() or {}).get("role") == "daemon")
        z = Zgm(db)
        first = z.queue("first", sleeper(tmp_path, 3))
        second = z.queue("second", sleeper(tmp_path, 0.2))  # same GPU: after first
        assert wait_for(lambda: z.state(first)[0] == TaskStatus.RUNNING)
        assert daemon.stop(daemon_args(db)) == 0  # stop the daemon mid-run
        assert ExecutorLease(db).holder() is None
        assert z.state(first)[0] == TaskStatus.RUNNING  # still marked, and still running
        (a,) = z.store.assignments_for_task(first.id)
        assert run_state(Path(a.log_path).with_suffix("")) == RunState.RUNNING
    finally:
        proc.kill()
        proc.wait()
    capsys.readouterr()

    # a new daemon picks up the running task and carries on with the queue
    proc = subprocess.Popen([PY, "-m", "zgm", "daemon", "run", "--db", str(db),
                             "--mock-gpus", "4", "--interval", "0.2"], cwd=ROOT,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        assert wait_for(lambda: z.state(second)[0] == TaskStatus.COMPLETED, 30)
        assert z.state(first) == (TaskStatus.COMPLETED, AssignmentState.COMPLETED)
        assert daemon.status(daemon_args(db)) == 0
        out = capsys.readouterr().out
        assert "zgm daemon" in out and "running:    0" in out
    finally:
        proc.terminate() if sys.platform != "win32" else daemon.stop(daemon_args(db))
        proc.wait(timeout=15)
    log = proc.stdout.read()
    assert "following 1 task(s) already running" in log
    assert "first completed" in log and "Started second" in log


def test_daemon_start_and_stop_commands(tmp_path, capsys):
    db = tmp_path / "s.db"
    try:
        assert daemon.start(daemon_args(db)) == 0
        assert "started: zgm daemon" in capsys.readouterr().out
        assert daemon.start(daemon_args(db)) == 0
        assert "already running" in capsys.readouterr().out
    finally:
        assert daemon.stop(daemon_args(db)) == 0
    assert "stopped" in capsys.readouterr().out
    assert daemon.stop(daemon_args(db)) == 0
    assert "not running" in capsys.readouterr().out
    assert "runs the queue" in (tmp_path / "daemon.log").read_text(encoding="utf-8")


def test_stop_refuses_to_kill_a_tui(tmp_path, capsys):
    db = tmp_path / "s.db"
    tui = ExecutorLease(db)
    tui.acquire("tui", "local")
    assert daemon.stop(daemon_args(db)) == 1
    assert "close that TUI" in capsys.readouterr().err
    tui.release()
