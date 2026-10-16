"""Executor: starts queued work, follows runs to their end, and stops them.

Exactly one process per database ticks an Executor: the holder of the
ExecutorLease (``zgm daemon``, or a TUI with ``--runtime``). Other zgm
processes are clients. They only use ``stop``, ``request_run_now`` and
``autostart``, which act through the database (plus, for local runs, the run's
own files), so the executor sees the change on its next tick.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from ..models import Assignment, AssignmentState, Task, TaskStatus, config_fingerprint
from ..runtime.base import RunHandle, RunState, Runtime
from . import config_loader as cl
from .scheduler import AssignmentService
from .task_store import TaskStore

Notify = Callable[..., None]  # notify(message, severity="information")

END_STATES = {
    RunState.SUCCEEDED: (AssignmentState.COMPLETED, TaskStatus.COMPLETED),
    RunState.CANCELLED: (AssignmentState.CANCELLED, TaskStatus.CANCELLED),
    RunState.FAILED: (AssignmentState.FAILED, TaskStatus.FAILED),
}


class Executor:
    def __init__(self, store: TaskStore, svc: AssignmentService,
                 runtime: Runtime | None = None, notify: Notify | None = None) -> None:
        self.store = store
        self.svc = svc
        self.runtime = runtime
        self.notify: Notify = notify or (lambda message, severity="information": None)
        self.runs: dict[str, RunHandle] = {}  # assignment id -> run being followed
        self._warned: set[str] = set()

    # auto-start, shared by every zgm on the database ---------------------------
    @property
    def autostart(self) -> bool:
        return self.store.storage.get_setting("autostart", "1") == "1"

    @autostart.setter
    def autostart(self, on: bool) -> None:
        self.store.storage.set_setting("autostart", "1" if on else "0")

    # the executor's loop ----------------------------------------------------------
    def tick(self, hold: bool = False) -> list[Task]:
        """Follow running work; start what is ready unless ``hold``."""
        self.adopt()
        self.poll()
        return [] if hold else self.dispatch()

    def adopt(self) -> None:
        """Follow runs this executor didn't start: from an earlier session, or
        from the executor before it. A run that left no trace is marked lost."""
        attach = getattr(self.runtime, "attach", None)
        for a in self.svc.running():
            if a.id in self.runs:
                continue
            handle = attach(a) if attach else None
            if handle is not None:
                self.runs[a.id] = handle
            elif attach is not None:
                self.finish(a, RunState.FAILED, lost=True)
            elif a.id not in self._warned:  # in-process runtimes can't reattach
                self._warned.add(a.id)
                task = self.store.tasks.get(a.task_id)
                self.notify(f"{task.name if task else a.task_id} is still marked running "
                            "from an earlier session, but this runtime can't follow it; "
                            "stop it (s) to free its GPUs.", severity="warning")

    def poll(self) -> None:
        for aid, handle in list(self.runs.items()):
            a = self.store.assignments.get(aid)
            if a is None or a.state != AssignmentState.RUNNING:
                # stopped or deleted by another zgm: make sure it is really gone
                del self.runs[aid]
                if self.runtime is not None and self.runtime.status(handle) == RunState.RUNNING:
                    self.runtime.cancel(handle)
                continue
            state = self.runtime.status(handle)
            if state in (RunState.PENDING, RunState.RUNNING):
                continue
            del self.runs[aid]
            self.finish(a, state, handle)

    def finish(self, a: Assignment, state: RunState, handle: RunHandle | None = None,
               lost: bool = False) -> None:
        if self.store.storage.assignment_state(a.id) not in (AssignmentState.RUNNING, None):
            return  # another zgm ended it meanwhile (e.g. stopped it); keep theirs
        a.actual_end = datetime.now()
        result = {}
        if handle is not None and hasattr(self.runtime, "result"):
            result = self.runtime.result(handle) or {}
            if result.get("ended"):  # it may have ended while no executor was watching
                a.actual_end = datetime.fromisoformat(result["ended"])
            if state == RunState.SUCCEEDED and result.get("gpu_ids"):
                # Ray picks the physical devices; record what it actually used
                a.actual_gpu_ids = tuple(result["gpu_ids"])
        a.state, task_status = END_STATES.get(state, END_STATES[RunState.FAILED])
        task = self.store.tasks.get(a.task_id)
        if task:
            task.status = task_status
            self.store.update_task(task, quiet=True)
        self.store.save_assignment(a, "lost" if lost else a.state.value)
        self.svc.recompute()
        self.store.notify()
        name = task.name if task else a.task_id
        if lost:
            self.notify(f"{name}: its run left no trace (killed or machine restarted); "
                        "marked failed", severity="warning")
        elif state == RunState.SUCCEEDED:
            self.notify(f"{name} completed")
        elif state == RunState.FAILED:
            code = result.get("returncode")
            self.notify(f"{name} failed" + (f" (exit code {code})" if code is not None else ""),
                        severity="error")

    def dispatch(self) -> list[Task]:
        """Start what may start now: tasks asked to run now (r) as soon as their
        GPUs are free, then — unless paused — the queue in order."""
        if self.runtime is None:
            return []
        started = []
        for a in [x for x in self.svc.planned() if x.run_now]:
            if not set(a.gpu_ids) & (self.svc.busy_gpus() | self.svc.blocked):
                task = self.store.tasks.get(a.task_id)
                if task and self.start(task, a):
                    started.append(task)
        if self.autostart:
            for a in self.svc.ready_to_start():
                task = self.store.tasks[a.task_id]
                if self.start(task, a):
                    started.append(task)
        if started:
            self.notify(f"Started {', '.join(t.name for t in started)}")
        return started

    # actions ------------------------------------------------------------------------
    def start(self, task: Task, a: Assignment) -> bool:
        """Hand a planned assignment to the runtime. If it can't start (bad
        working dir, missing python/conda env, ...), the task goes back to the
        waiting queue as failed so it doesn't hold up the work behind it."""
        if self.store.storage.assignment_state(a.id) != AssignmentState.PLANNED:
            return False  # started, stopped or removed by another zgm meanwhile
        self.refresh_fingerprint(task)  # the run uses the config as it is *now*
        try:
            handle = self.runtime.submit(task, a)
        except Exception as e:
            self.launch_failed(task, a, e)
            return False
        a.state = AssignmentState.RUNNING
        a.run_now = False
        a.actual_start = a.actual_start or datetime.now()
        # remember the estimate the run started with (None = no history)
        a.planned_start = a.actual_start
        a.planned_end = (a.actual_start + task.estimated_duration
                         if task.estimated_duration else None)
        self.runs[a.id] = handle
        task.status = TaskStatus.RUNNING
        self.store.update_task(task, quiet=True)
        self.store.save_assignment(a, "started")
        self.svc.recompute()
        self.store.notify()
        return True

    def launch_failed(self, task: Task, a: Assignment, error: Exception) -> None:
        self.svc.unassign(a.id)
        task.status = TaskStatus.FAILED
        self.store.update_task(task, quiet=True)
        self.store.storage.log_event("launch_failed", task.id, a.id, {"error": str(error)})
        self.store.notify()
        self.notify(f"{task.name} could not start: {error}. It is back in the queue.",
                    severity="error")

    def request_run_now(self, task: Task) -> tuple[str, str]:
        """Mark a planned task to start ahead of the queue as soon as its GPUs are
        free. Returns (message, severity); the executor starts it on its tick."""
        a = self.store.active_assignment(task.id)
        if a is None or a.state != AssignmentState.PLANNED:
            return "Place the task on the timeline before running it.", "warning"
        a.run_now = True
        self.store.save_assignment(a, "run_now")
        busy = sorted(set(a.gpu_ids) & (self.svc.busy_gpus() | self.svc.blocked))
        if busy:
            outside = sorted(set(busy) & self.svc.blocked)
            why = f" (GPU {','.join(map(str, outside))} is in use outside zgm)" if outside else ""
            return (f"{task.name} starts as soon as GPU {','.join(map(str, busy))} is free, "
                    f"ahead of the queue{why}", "warning")
        return f"Starting {task.name}", "information"

    def stop(self, task: Task, requeue: bool) -> bool:
        """End a running task, keep the partial run in the history, then put the
        task back in the queue or delete it. Works from any zgm: the database is
        updated first, then the run is cancelled."""
        a = next((x for x in self.store.assignments_for_task(task.id)
                  if x.state == AssignmentState.RUNNING), None)
        if a is None:
            return False
        a.state = AssignmentState.CANCELLED
        a.actual_end = datetime.now()
        self.store.save_assignment(a, "stopped")
        handle = self.runs.pop(a.id, None)
        if handle is None and hasattr(self.runtime, "attach"):
            handle = self.runtime.attach(a)
        if handle is not None and self.runtime is not None:
            self.runtime.cancel(handle)
        if requeue:
            task.status = TaskStatus.PENDING
            self.store.update_task(task, quiet=True)
        else:
            self.store.remove_task(task.id)
        self.svc.recompute()
        self.store.notify()
        return True

    @staticmethod
    def refresh_fingerprint_of(task: Task) -> None:
        if not task.config_path:
            return
        node = cl.load_node(Path(task.config_path))
        if not node.error:
            params = {k: v for k, v in node.resolved.items() if k != cl.SCHEDULER_KEY}
            task.fingerprint = config_fingerprint(params, task.required_gpus)

    def refresh_fingerprint(self, task: Task) -> None:
        before = task.fingerprint
        self.refresh_fingerprint_of(task)
        if task.fingerprint != before:
            self.store.update_task(task, quiet=True)
