"""RayRuntime: execute a task's shell command as a Ray task.

Caveat: Ray GPUs are *logical* resources. ``num_gpus=n`` reserves n GPUs and
sets CUDA_VISIBLE_DEVICES for the worker, but Ray's scheduler picks which
physical devices. So an assignment's ``preferred_gpu_ids`` (the user's plan)
may differ from ``actual_gpu_ids`` (reported back from the worker via
``ray.get_gpu_ids()``). Strict pinning to physical GPUs needs custom
per-device resources.

Stopping a task kills the user's processes too: the worker kills its children
when cancelled, and if Ray has to kill the worker itself, the processes
carrying the task's ZGM_TASK_ID on this machine are killed after it.
"""

from __future__ import annotations

import os

from ..models import Assignment, Task
from ..services.launcher import build_process
from .base import RunHandle, RunState


def _kill_tree(pid: int) -> None:
    import psutil

    try:
        root = psutil.Process(pid)
        procs = [*root.children(recursive=True), root]
    except psutil.NoSuchProcess:
        return
    for proc in procs:
        try:
            proc.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(procs, timeout=5)


def kill_task_processes(task_id: str) -> int:
    """Kill every process on this machine started for ``task_id`` (they carry
    ZGM_TASK_ID in their environment). The last resort when Ray had to kill a
    worker outright, which leaves the worker's children running."""
    import psutil

    killed = 0
    for proc in psutil.process_iter():
        try:
            if proc.environ().get("ZGM_TASK_ID") == task_id:
                _kill_tree(proc.pid)
                killed += 1
        except (psutil.Error, OSError):
            continue
    return killed


def _run_command(args: list[str] | str, cwd: str, env: dict[str, str]) -> dict:
    """Runs in the Ray worker: its own environment, plus the task's overrides.

    A cancel arrives as KeyboardInterrupt (ray.cancel without force); the
    task's own processes are killed before the worker gives up, so nothing
    keeps holding the GPU."""
    import os
    import subprocess

    import ray

    gpu_ids = [int(g) for g in ray.get_gpu_ids()]
    full_env = {**os.environ, **env, "ZGM_GPU_IDS": ",".join(map(str, gpu_ids))}
    proc = subprocess.Popen(args, shell=isinstance(args, str), cwd=cwd, env=full_env,
                            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)
    try:
        stdout, stderr = proc.communicate()
    except BaseException:
        _kill_tree(proc.pid)
        raise
    return {
        "returncode": proc.returncode,
        "gpu_ids": gpu_ids,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "stdout_tail": stdout[-4000:],
        "stderr_tail": stderr[-4000:],
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
        spec = build_process(task, None)  # Ray sets CUDA_VISIBLE_DEVICES itself
        ref = remote.remote(spec.args, spec.cwd, spec.overrides())
        return RunHandle(assignment.id, ref, task.id)

    def cancel(self, handle: RunHandle, grace: float = 10.0) -> None:
        """Ask the task to stop (it kills its own processes); if it doesn't
        within ``grace`` seconds, kill the worker and then, on this machine,
        whatever it left behind. On Windows Ray can't interrupt a running task,
        so it goes straight to the second step."""
        ray = self._ensure()
        if os.name != "nt":
            ray.cancel(handle.ref, force=False)
            ready, _ = ray.wait([handle.ref], timeout=grace)
            if ready:
                return
        ray.cancel(handle.ref, force=True)
        ray.wait([handle.ref], timeout=5)
        if handle.task_id:
            kill_task_processes(handle.task_id)

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
