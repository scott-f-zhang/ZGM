"""LocalRuntime: run tasks as processes on this machine that outlive zgm.

Unlike Ray, the planned GPUs are exactly the GPUs used: the process gets
``CUDA_VISIBLE_DEVICES`` set to the assignment's GPU ids.

Each task is started through a detached runner (``zgm.runtime.runner``) and
leaves files next to its log, ``<log_dir>/<assignment id>.*``:

- ``.log``: the task's stdout and stderr, written as they come;
- ``.run.json``: runner/child pids, start time, then exit code and end time;
- ``.lock``: held by the runner while it lives (so liveness never relies on pids);
- ``.cancel``: written when someone stops the task;
- ``.config.yaml``: the ``{config}`` snapshot, if the task uses one.

So the state of a run is on disk, not in this object: any zgm process (a
restarted TUI, the daemon) can ``attach`` to a run another one started.

Cluster mode: ``locate`` says where an assignment's GPUs are. GPUs on another
host get a runner started there, through the host's transport (e.g. ssh), in
``--remote`` mode (see runner.py): it marks ``run.json`` with ``remote``, beats
instead of holding the lock, and stops the task when ``.cancel`` appears. The
log dir must be on a filesystem both hosts share.
"""

from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from ..models import Assignment, Task
from ..services.filelock import is_locked
from ..services.launcher import build_process
from .base import RunHandle, RunState

START_TIMEOUT = 30.0  # seconds for a runner to come up and report
REMOTE_START_TIMEOUT = 60.0  # the same over ssh and a shared filesystem
STALE = 60.0  # a remote runner without a heartbeat for this long is gone


@dataclass
class Placement:
    """Where an assignment's GPUs are (cluster mode)."""

    group: str
    host: str
    visible: list[int]  # the GPUs' indices on their host: CUDA_VISIBLE_DEVICES
    # how to reach the host (cluster/transport.py), or None: this process runs
    # in the group, so the task starts here
    transport: object | None = None
    drop_env: tuple[str, ...] = ()  # variable prefixes a task elsewhere must not get

    @property
    def remote(self) -> bool:
        return self.transport is not None


Locate = Callable[[Sequence[int]], "Placement | None"]


def _file(base: Path, suffix: str) -> Path:
    return base.with_name(base.name + suffix)


def read_run(base: Path) -> dict | None:
    try:
        return json.loads(_file(base, ".run.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def heartbeat_age(info: dict) -> float:
    """Seconds since a remote runner last reported (inf if it never did)."""
    try:
        return (datetime.now() - datetime.fromisoformat(info["heartbeat"])).total_seconds()
    except (KeyError, TypeError, ValueError):
        return float("inf")


def _alive(base: Path, info: dict | None) -> bool:
    if info is not None and info.get("remote"):
        return "exit_code" not in info and heartbeat_age(info) < STALE
    return is_locked(_file(base, ".lock"))


def run_state(base: Path) -> RunState:
    """The state of the run whose files start with ``base``."""
    info = read_run(base)
    cancelled = _file(base, ".cancel").exists()
    if info is not None and "exit_code" in info:
        if info["exit_code"] == 0:
            return RunState.SUCCEEDED
        return RunState.CANCELLED if cancelled else RunState.FAILED
    if _alive(base, info):
        return RunState.RUNNING
    # the runner is gone without recording an exit: stopped, or lost (killed, reboot)
    return RunState.CANCELLED if cancelled else RunState.FAILED


def cancel_run(base: Path) -> None:
    """Stop the run's whole process tree. Marks it first, so whoever reads its
    state next sees "cancelled" rather than "failed"."""
    _file(base, ".cancel").write_text("", encoding="utf-8")
    info = read_run(base) or {}
    if info.get("remote"):  # its runner sees .cancel, stops the task and says so
        deadline = time.monotonic() + 15
        while _alive(base, info) and time.monotonic() < deadline:
            time.sleep(0.2)
            info = read_run(base) or info
        return
    pids = [p for p in (info.get("runner_pid"), info.get("child_pid")) if p]
    if not pids or not is_locked(_file(base, ".lock")):
        return  # not running (any more)
    if os.name == "nt":
        for pid in pids:  # the runner's tree, and the child in case it outlived it
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(pid)], capture_output=True)
    else:
        try:
            os.killpg(pids[0], signal.SIGTERM)  # the runner leads the task's process group
            deadline = time.monotonic() + 3
            while is_locked(_file(base, ".lock")) and time.monotonic() < deadline:
                time.sleep(0.05)
            if is_locked(_file(base, ".lock")):
                # macOS answers a dead group with EPERM, not ESRCH
                os.killpg(pids[0], signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    deadline = time.monotonic() + 5
    while is_locked(_file(base, ".lock")) and time.monotonic() < deadline:
        time.sleep(0.05)
    try:  # the runner was killed too, so it can't say so itself
        with open(_file(base, ".log"), "a", encoding="utf-8") as log:
            log.write(f"\n# zgm: stopped at {datetime.now():%Y-%m-%d %H:%M:%S}\n")
    except OSError:
        pass


def _detached() -> dict:
    """Popen options so the runner survives zgm, its terminal and its job."""
    if os.name != "nt":
        return {"start_new_session": True}
    flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
    return {"creationflags": flags | subprocess.CREATE_BREAKAWAY_FROM_JOB}


def spawn_detached(argv: list[str], log: Path, **kw) -> subprocess.Popen:
    """Start ``argv`` on its own: new process group/session, no console window,
    out of the parent's job object where allowed. Output is appended to ``log``."""
    with open(log, "ab") as out:
        options = _detached()
        try:
            return subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=out,
                                    stderr=subprocess.STDOUT, **options, **kw)
        except OSError:
            if os.name != "nt":
                raise
            # the job we are in doesn't allow breakaway: start inside it instead
            options["creationflags"] &= ~subprocess.CREATE_BREAKAWAY_FROM_JOB
            return subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=out,
                                    stderr=subprocess.STDOUT, **options, **kw)


class LocalRuntime:
    name = "local"

    def __init__(self, log_dir: str | Path, locate: Locate | None = None) -> None:
        self.log_dir = Path(log_dir)
        self.locate = locate  # cluster mode: where GPUs are (cluster/core.py)

    def available(self) -> bool:
        return True

    def base(self, assignment: Assignment) -> Path:
        if assignment.log_path:
            return Path(assignment.log_path).with_suffix("")
        return self.log_dir / assignment.id

    def log_path(self, assignment_id: str) -> Path:
        return self.log_dir / f"{assignment_id}.log"

    def submit(self, task: Task, assignment: Assignment) -> RunHandle:
        """Start the task; raises ValueError if it can't be built or started."""
        gpu_ids = tuple(assignment.preferred_gpu_ids)
        base = self.log_dir / assignment.id
        place = self.locate(gpu_ids) if self.locate else None
        proc = build_process(task, gpu_ids, snapshot=_file(base, ".config.yaml"),
                             visible=place.visible if place else None,
                             drop_env=place.drop_env if place and place.remote else None)
        self.log_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        for suffix in (".run.json", ".cancel"):
            _file(base, suffix).unlink(missing_ok=True)
        spec_path = _file(base, ".spec.json")
        fd = os.open(spec_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"name": task.name, "gpu_ids": list(gpu_ids), "args": proc.args, "cwd": proc.cwd, "env": proc.env,
                       "display": proc.display()}, f)
        log = _file(base, ".log")
        if place and place.remote:
            self._start_remote(place.transport, base, log)
            timeout, poll = REMOTE_START_TIMEOUT, (lambda: None)
        else:
            runner = spawn_detached([sys.executable, "-m", "zgm.runtime.runner", str(base)], log)
            timeout, poll = START_TIMEOUT, runner.poll
        deadline = time.monotonic() + timeout
        while (info := read_run(base)) is None:
            if poll() is not None or time.monotonic() > deadline:
                spec_path.unlink(missing_ok=True)
                where = (f"; is {self.log_dir} shared with {place.host}? (zgm check)"
                         if place and place.remote else "")
                raise ValueError(f"the runner did not start; see {log}{where}")
            time.sleep(0.05)
        if "error" in info:
            raise ValueError(f"could not start: {info['error']}")
        assignment.actual_gpu_ids = gpu_ids
        assignment.log_path = str(log)
        return RunHandle(assignment.id, base)

    @staticmethod
    def _start_remote(transport, base: Path, log: Path) -> None:
        """Start the runner on the transport's host, detached there, so it lives
        as long as that host (or, with Slurm, that job) does."""
        runner = shlex.join([sys.executable, "-m", "zgm.runtime.runner", "--remote", str(base)])
        # not "cd / && ... &": that backgrounds a subshell holding ssh's output open.
        # macOS has no setsid; nohup is enough there (and works on Linux too).
        script = (f"cd / || exit 1; if command -v setsid >/dev/null 2>&1; then "
                  f"setsid {runner} </dev/null >>{shlex.quote(str(log))} 2>&1 & else "
                  f"nohup {runner} </dev/null >>{shlex.quote(str(log))} 2>&1 & fi")
        host = transport.host
        try:
            done = transport.run(script, timeout=60)
        except (OSError, subprocess.TimeoutExpired) as e:
            raise ValueError(f"{host}: {e}") from None
        if done.returncode != 0:
            detail = (done.stderr or done.stdout).strip().splitlines()[-1:] or ["no output"]
            raise ValueError(f"{host}: could not start the runner: {detail[0]}")

    def attach(self, assignment: Assignment) -> RunHandle | None:
        """A handle to a run started earlier (by this or another zgm), if it has one."""
        base = self.base(assignment)
        return RunHandle(assignment.id, base) if read_run(base) is not None else None

    def cancel(self, handle: RunHandle) -> None:
        cancel_run(handle.ref)

    def status(self, handle: RunHandle) -> RunState:
        return run_state(handle.ref)

    def result(self, handle: RunHandle) -> dict:
        info = read_run(handle.ref) or {}
        return {"returncode": info.get("exit_code"), "gpu_ids": info.get("gpu_ids", []),
                "started": info.get("started"), "ended": info.get("ended"),
                "log": str(_file(handle.ref, ".log"))}
