"""Runner: starts one task's process and records how it ends, detached from zgm.

``python -m zgm.runtime.runner <run base>`` (``<base>`` = ``<log dir>/<assignment id>``):

- reads and deletes ``<base>.spec.json`` (argv, cwd, environment);
- holds ``<base>.lock`` for as long as it lives: any zgm process can tell it is
  alive, and the OS drops the lock if it dies, however it dies;
- starts the task with stdout/stderr appended to ``<base>.log``;
- writes ``<base>.run.json``: pids and start time, then the exit code and end time.

Because it runs on its own (new process group, no console window), the task
outlives the zgm that started it; another zgm picks it up from these files.

``--remote``: the runner was started on another host (cluster mode), so
the zgm that follows it can't see its lock or signal its pids. Instead it
writes a ``heartbeat`` time into ``run.json`` every few seconds (no heartbeat
for a while = it is gone), and stops the task itself when ``<base>.cancel``
appears. All of this goes through the shared filesystem.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from ..services.filelock import FileLock

HEARTBEAT = 5.0  # seconds between heartbeats of a remote runner


def write_json(path: Path, data: dict) -> None:
    """Atomically: readers see the old or the new file, never half of one.

    On Windows the replace fails while a reader has the file open, so retry."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data), encoding="utf-8")
    for attempt in range(200):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 199:
                raise
            time.sleep(0.01)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _stop_group(proc: subprocess.Popen) -> None:
    """TERM the task's process group, KILL it if it hasn't exited after 3 s."""
    for sig, wait in ((signal.SIGTERM, 3.0), (signal.SIGKILL, 5.0)):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=wait)
            return
        except subprocess.TimeoutExpired:
            continue


def _follow(proc: subprocess.Popen, base: Path, state_path: Path, state: dict, log) -> int:
    """Remote mode: wait for the task, beating and watching for ``.cancel``."""
    cancel = base.with_name(base.name + ".cancel")
    beat = 0.0
    while True:
        try:
            return proc.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            pass
        if cancel.exists():
            _stop_group(proc)
            log.write(f"\n# zgm: stopped at {datetime.now():%Y-%m-%d %H:%M:%S}\n")
            return proc.wait()
        if time.monotonic() - beat >= HEARTBEAT:
            beat = time.monotonic()
            state["heartbeat"] = _now()
            write_json(state_path, state)


def run(base: Path, remote: bool = False) -> int:
    spec_path = base.with_name(base.name + ".spec.json")
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    spec_path.unlink()  # it holds the task's whole environment
    if not remote:
        lock = FileLock(base.with_name(base.name + ".lock"))
        lock.acquire(timeout=10)
    state_path = base.with_name(base.name + ".run.json")
    state = {"runner_pid": os.getpid(), "gpu_ids": spec.get("gpu_ids", []), "started": _now()}
    if remote:
        state.update(remote=True, host=socket.gethostname(), heartbeat=state["started"])
    with open(base.with_name(base.name + ".log"), "a", encoding="utf-8",
              errors="replace") as log:
        log.write(f"# zgm: {spec['name']} started {state['started']}\n"
                  f"# cwd: {spec['cwd']}\n# run: {spec['display']}\n"
                  f"# CUDA_VISIBLE_DEVICES={spec['env'].get('CUDA_VISIBLE_DEVICES', '')}\n\n")
        log.flush()
        try:
            proc = subprocess.Popen(
                spec["args"], shell=isinstance(spec["args"], str), cwd=spec["cwd"],
                env=spec["env"], stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=remote,  # its own group, so a stop reaches all of it
            )
        except OSError as e:
            log.write(f"# zgm: could not start: {e}\n")
            write_json(state_path, {**state, "error": str(e), "exit_code": 127,
                                    "ended": _now()})
            return 127
        state["child_pid"] = proc.pid
        write_json(state_path, state)
        code = _follow(proc, base, state_path, state, log) if remote else proc.wait()
        log.write(f"\n# zgm: exited with code {code} at {_now()}\n")
    write_json(state_path, {**state, "exit_code": code, "ended": _now()})
    return code


if __name__ == "__main__":
    remote = "--remote" in sys.argv[1:]
    sys.exit(run(Path(next(a for a in sys.argv[1:] if a != "--remote")), remote))
