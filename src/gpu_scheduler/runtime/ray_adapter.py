"""RayRuntime: execute a task's shell command as a Ray task.

Caveat: Ray GPUs are *logical* resources. ``num_gpus=n`` reserves n GPUs and
sets CUDA_VISIBLE_DEVICES for the worker, but Ray's scheduler picks which
physical devices. So an assignment's ``preferred_gpu_ids`` (the user's plan)
may differ from ``actual_gpu_ids`` (reported back from the worker via
``ray.get_gpu_ids()``). Strict pinning to physical GPUs needs custom
per-device resources and is left for phase 2.
"""

from __future__ import annotations

from ..models import Assignment, Task
from .base import RunHandle, RunState


def _run_command(command: str) -> dict:
    import os
    import subprocess

    import ray

    gpu_ids = [int(g) for g in ray.get_gpu_ids()]
    proc = subprocess.run(command, shell=True, capture_output=True, text=True)
    return {
        "returncode": proc.returncode,
        "gpu_ids": gpu_ids,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }


class RayRuntime:
    name = "ray"

    def __init__(self, address: str | None = None, **init_kwargs) -> None:
        self.address = address
        self.init_kwargs = init_kwargs
        self._ray = None

    def available(self) -> bool:
        try:
            import ray  # noqa: F401
        except ImportError:
            return False
        return True

    def _ensure(self):
        if self._ray is None:
            import ray

            if not ray.is_initialized():
                ray.init(address=self.address, ignore_reinit_error=True, **self.init_kwargs)
            self._ray = ray
        return self._ray

    def submit(self, task: Task, assignment: Assignment, num_gpus: float | None = None) -> RunHandle:
        ray = self._ensure()
        n = task.required_gpus if num_gpus is None else num_gpus
        remote = ray.remote(num_gpus=n)(_run_command)
        ref = remote.remote(task.command)
        return RunHandle(assignment.id, ref)

    def cancel(self, handle: RunHandle) -> None:
        self._ensure().cancel(handle.ref, force=True)

    def status(self, handle: RunHandle) -> RunState:
        ray = self._ensure()
        ready, _ = ray.wait([handle.ref], timeout=0)
        if not ready:
            return RunState.RUNNING
        try:
            result = ray.get(handle.ref)
        except ray.exceptions.TaskCancelledError:
            return RunState.CANCELLED
        except Exception:
            return RunState.FAILED
        return RunState.SUCCEEDED if result["returncode"] == 0 else RunState.FAILED

    def result(self, handle: RunHandle) -> dict:
        return self._ensure().get(handle.ref)
