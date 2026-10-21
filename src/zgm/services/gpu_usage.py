"""Which GPUs are in use by programs outside zgm.

The source is NVML's list of compute processes per GPU (CUDA and MPS clients):
the same data ``nvidia-smi`` shows. Graphics-only processes (the desktop,
browsers drawing) are not counted; every display GPU has them.

A process belongs to zgm if it is a task's runner or child, any descendant of
those (torchrun workers, data loader processes, ...), or carries the
``ZGM_TASK_ID`` of a running task in its environment (which also catches
processes whose parent has exited). Everything else on a GPU is external, and
the scheduler doesn't start new work there until it is gone.

On another host (cluster mode) only the second test applies: ``zgm.probe`` reads
each process's ``ZGM_TASK_ID`` there and reports it with the process.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import psutil

from ..models import AssignmentState, GPUDevice, GPUProcess
from ..probe import task_of
from .task_store import TaskStore


def zgm_pids(store: TaskStore) -> set[int]:
    """Processes of the tasks this database has running, with their descendants."""
    from ..runtime.local import read_run

    roots = set()
    for a in store.assignments.values():
        if a.state != AssignmentState.RUNNING or not a.log_path:
            continue
        info = read_run(Path(a.log_path).with_suffix("")) or {}
        if info.get("remote"):
            continue  # its pids are another machine's
        roots.update(p for p in (info.get("runner_pid"), info.get("child_pid")) if p)
    pids = set(roots)
    for pid in roots:
        try:
            pids.update(c.pid for c in psutil.Process(pid).children(recursive=True))
        except psutil.Error:
            continue
    return pids


def process_name(pid: int) -> str:
    try:
        return psutil.Process(pid).name()
    except psutil.Error:
        return "?"


def external(processes: Iterable[GPUProcess], own_pids: set[int],
             running_tasks: set[str]) -> list[GPUProcess]:
    def ours(p: GPUProcess) -> bool:
        if p.host is not None:  # on another node: probed there
            return p.zgm_task in running_tasks
        return p.pid in own_pids or task_of(p.pid) in running_tasks

    return [p for p in processes if not ours(p)]


class GPUWatch:
    """Keeps each GPUDevice's ``external`` list current; ``busy`` is the set of
    GPUs other programs use. A GPU whose processes NVML can't list is reported
    (``processes_visible = False``) but not blocked: nothing is known about it."""

    def __init__(self, provider, gpus: list[GPUDevice]) -> None:
        self.provider = provider
        self.gpus = gpus

    def refresh(self, store: TaskStore) -> tuple[set[int], list[str]]:
        """Re-read the processes. Returns (busy GPU ids, what changed, as messages)."""
        own = zgm_pids(store)
        running = {a.task_id for a in store.assignments.values()
                   if a.state == AssignmentState.RUNNING}
        changes = []
        for g in self.gpus:
            before = {p.pid for p in g.external}
            try:
                procs = self.provider.processes(g.id)
            except Exception as e:
                if g.processes_visible:
                    changes.append(f"GPU {g.id}: can't list its processes ({e}); "
                                   "use by other programs can't be detected there")
                g.processes_visible, g.external = False, []
                continue
            g.processes_visible = True
            g.external = external(procs, own, running)
            after = {p.pid for p in g.external}
            if after and not before:
                changes.append(f"GPU {g.id} is in use outside zgm ({describe(g.external)}): "
                               "nothing new starts there until it is free")
            elif before and not after:
                changes.append(f"GPU {g.id} is free of other programs again")
        return {g.id for g in self.gpus if g.external}, changes


def describe(procs: list[GPUProcess]) -> str:
    parts = []
    for p in procs[:3]:
        mem = f", {p.used_memory / 1024**3:.1f} GiB" if p.used_memory is not None else ""
        parts.append(f"{p.name} pid {p.pid}{mem}")
    if len(procs) > 3:
        parts.append(f"+{len(procs) - 3} more")
    return "; ".join(parts)
