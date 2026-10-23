"""Which GPUs are in use by programs outside zgm.

The source is NVML's list of compute processes per GPU (CUDA and MPS clients):
the same data ``nvidia-smi`` shows. Graphics-only processes (the desktop,
browsers drawing) are not counted; every display GPU has them.

A process belongs to zgm if it is a task's runner or child, any descendant of
those (torchrun workers, data loader processes, ...), or carries the
``ZGM_TASK_ID`` of a running task in its environment (which also catches
processes whose parent has exited). Everything else on a GPU is external, and
the scheduler doesn't start new work there until it is gone. That includes the
processes of a zgm task whose run just ended and that are still exiting (a
vLLM engine takes a few seconds): they hold GPU memory, so the GPU waits for
them too, but they are not reported as another program's.

On another host (cluster mode) only the second test applies: ``zgm.probe`` reads
each process's ``ZGM_TASK_ID`` there and reports it with the process.
"""

from __future__ import annotations

import time
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
    if not roots:
        return set()
    kids: dict[int, list[int]] = {}  # one pass over the process table for every run
    for p in psutil.process_iter(["ppid"]):
        kids.setdefault(p.info["ppid"], []).append(p.pid)
    pids, todo = set(roots), list(roots)
    while todo:
        for c in kids.get(todo.pop(), []):
            if c not in pids:
                pids.add(c)
                todo.append(c)
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


def task_id(p: GPUProcess) -> str | None:
    """The zgm task a GPU process belongs to, by its ``ZGM_TASK_ID``, if any."""
    return p.zgm_task if p.host is not None else task_of(p.pid)


class GPUWatch:
    """Keeps each GPUDevice's ``external`` list current; ``busy`` is the set of
    GPUs other programs use. A GPU whose processes NVML can't list is reported
    (``processes_visible = False``) but not blocked: nothing is known about it.

    A GPU is blocked as soon as another program is seen on it, but reported only
    once that lasts ``report_after`` seconds (except on the first refresh, which
    reports what is there at once): a task's processes that are still exiting
    may no longer tell which task they belong to, and are gone within seconds."""

    REPORT_AFTER = 30.0

    def __init__(self, provider, gpus: list[GPUDevice], report_after: float = REPORT_AFTER,
                 clock=time.monotonic) -> None:
        self.provider = provider
        self.gpus = gpus
        self.report_after = report_after
        self.clock = clock
        self.since: dict[int, float] = {}  # GPU -> when other programs were first seen there
        self.reported: set[int] = set()
        self.refreshed = False

    def refresh(self, store: TaskStore) -> tuple[set[int], list[str]]:
        """Re-read the processes. Returns (busy GPU ids, what changed, as messages)."""
        own = zgm_pids(store)
        running = {a.task_id for a in store.assignments.values()
                   if a.state == AssignmentState.RUNNING}
        changes = []

        def foreign(procs):  # not a zgm task's leftovers
            return {p.pid for p in procs if task_id(p) not in store.tasks}

        now, first = self.clock(), not self.refreshed
        self.refreshed = True
        for g in self.gpus:
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
            after = foreign(g.external)
            if not after:
                self.since.pop(g.id, None)
                if g.id in self.reported:
                    self.reported.discard(g.id)
                    changes.append(f"GPU {g.id} is free of other programs again")
                continue
            seen = self.since.setdefault(g.id, now)
            if g.id not in self.reported and (first or now - seen >= self.report_after):
                self.reported.add(g.id)
                others = [p for p in g.external if p.pid in after]
                changes.append(f"GPU {g.id} is in use outside zgm ({describe(others)}): "
                               "nothing new starts there until it is free")
        return {g.id for g in self.gpus if g.external}, changes


def describe(procs: list[GPUProcess]) -> str:
    parts = []
    for p in procs[:3]:
        mem = f", {p.used_memory / 1024**3:.1f} GiB" if p.used_memory is not None else ""
        parts.append(f"{p.name} pid {p.pid}{mem}")
    if len(procs) > 3:
        parts.append(f"+{len(procs) - 3} more")
    return "; ".join(parts)
