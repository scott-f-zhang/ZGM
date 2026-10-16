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
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from ..models import Assignment, Task
from ..services.filelock import is_locked
from ..services.launcher import build_process
from .base import RunHandle, RunState

START_TIMEOUT = 30.0  # seconds for a runner to come up and report


def _file(base: Path, suffix: str) -> Path:
    return base.with_name(base.name + suffix)


def read_run(base: Path) -> dict | None:
    try:
        return json.loads(_file(base, ".run.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def run_state(base: Path) -> RunState:
    """The state of the run whose files start with ``base``."""
    info = read_run(base)
    cancelled = _file(base, ".cancel").exists()
    if info is not None and "exit_code" in info:
        if info["exit_code"] == 0:
            return RunState.SUCCEEDED
        return RunState.CANCELLED if cancelled else RunState.FAILED
    if is_locked(_file(base, ".lock")):
        return RunState.RUNNING
    # the runner is gone without recording an exit: stopped, or lost (killed, reboot)
    return RunState.CANCELLED if cancelled else RunState.FAILED


def cancel_run(base: Path) -> None:
    """Stop the run's whole process tree. Marks it first, so whoever reads its
    state next sees "cancelled" rather than "failed"."""
    _file(base, ".cancel").write_text("", encoding="utf-8")
    info = read_run(base) or {}
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
            os.killpg(pids[0], signal.SIGKILL)
        except ProcessLookupError:
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

    def __init__(self, log_dir: str | Path) -> None:
        self.log_dir = Path(log_dir)

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
        proc = build_process(task, gpu_ids, snapshot=_file(base, ".config.yaml"))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        for suffix in (".run.json", ".cancel"):
            _file(base, suffix).unlink(missing_ok=True)
        spec_path = _file(base, ".spec.json")
        fd = os.open(spec_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"name": task.name, "gpu_ids": list(gpu_ids), "args": proc.args, "cwd": proc.cwd, "env": proc.env,
                       "display": proc.display()}, f)
        log = _file(base, ".log")
        runner = spawn_detached([sys.executable, "-m", "zgm.runtime.runner", str(base)], log)
        deadline = time.monotonic() + START_TIMEOUT
        while (info := read_run(base)) is None:
            if runner.poll() is not None or time.monotonic() > deadline:
                spec_path.unlink(missing_ok=True)
                raise ValueError(f"the runner did not start; see {log}")
            time.sleep(0.02)
        if "error" in info:
            raise ValueError(f"could not start: {info['error']}")
        assignment.actual_gpu_ids = gpu_ids
        assignment.log_path = str(log)
        return RunHandle(assignment.id, base)

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
