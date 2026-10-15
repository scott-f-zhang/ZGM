"""LocalRuntime: run a task as a child process on this machine.

Unlike Ray, the planned GPUs are exactly the GPUs used: the process gets
``CUDA_VISIBLE_DEVICES`` set to the assignment's GPU ids. stdout and stderr go
to ``<log_dir>/<assignment id>.log`` as the process writes them; a ``{config}``
snapshot goes next to it as ``<assignment id>.config.yaml``.
"""

from __future__ import annotations

import os
import signal
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import IO

from ..models import Assignment, Task
from ..services.launcher import build_process
from .base import RunHandle, RunState


@dataclass
class _Run:
    proc: subprocess.Popen
    log: IO[str]
    log_path: Path
    gpu_ids: tuple[int, ...]
    cancelled: bool = False
    closed: bool = field(default=False, repr=False)


class LocalRuntime:
    name = "local"

    def __init__(self, log_dir: str | Path) -> None:
        self.log_dir = Path(log_dir)

    def available(self) -> bool:
        return True

    def log_path(self, assignment_id: str) -> Path:
        return self.log_dir / f"{assignment_id}.log"

    def submit(self, task: Task, assignment: Assignment) -> RunHandle:
        """Start the process; raises ValueError if it can't be built or started."""
        gpu_ids = tuple(assignment.preferred_gpu_ids)
        spec = build_process(task, gpu_ids,
                             snapshot=self.log_dir / f"{assignment.id}.config.yaml")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        path = self.log_path(assignment.id)
        log = open(path, "w", encoding="utf-8", errors="replace")
        log.write(f"# zgm: {task.name} started {datetime.now():%Y-%m-%d %H:%M:%S}\n"
                  f"# cwd: {spec.cwd}\n# run: {spec.display()}\n"
                  f"# CUDA_VISIBLE_DEVICES={spec.env.get('CUDA_VISIBLE_DEVICES', '')}\n\n")
        log.flush()
        kwargs = {}
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True  # its own process group, killed as a whole
        try:
            proc = subprocess.Popen(
                spec.args, shell=spec.shell, cwd=spec.cwd, env=spec.env,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, **kwargs,
            )
        except OSError as e:
            log.write(f"# zgm: failed to start: {e}\n")
            log.close()
            raise ValueError(f"could not start: {e}") from e
        assignment.actual_gpu_ids = gpu_ids
        return RunHandle(assignment.id, _Run(proc, log, path, gpu_ids))

    def cancel(self, handle: RunHandle) -> None:
        run: _Run = handle.ref
        run.cancelled = True
        if run.proc.poll() is not None:
            return
        # kill the whole tree: shells, launchers and dataloader workers included
        if os.name == "nt":
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(run.proc.pid)],
                           capture_output=True)
        else:
            try:
                os.killpg(run.proc.pid, signal.SIGTERM)
                run.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(run.proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        self._close(run)

    def status(self, handle: RunHandle) -> RunState:
        run: _Run = handle.ref
        code = run.proc.poll()
        if run.cancelled:
            self._close(run)
            return RunState.CANCELLED
        if code is None:
            return RunState.RUNNING
        self._close(run)
        return RunState.SUCCEEDED if code == 0 else RunState.FAILED

    def result(self, handle: RunHandle) -> dict:
        run: _Run = handle.ref
        return {"returncode": run.proc.returncode, "gpu_ids": list(run.gpu_ids),
                "log": str(run.log_path)}

    def _close(self, run: _Run) -> None:
        if run.closed:
            return
        run.closed = True
        try:
            run.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        how = "stopped" if run.cancelled else f"exited with code {run.proc.returncode}"
        run.log.write(f"\n# zgm: {how} at {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        run.log.close()
