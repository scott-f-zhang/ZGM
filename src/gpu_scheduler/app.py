"""GPUSchedulerApp: wires services to the two views (Schedule, Configs)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from textual.app import App
from textual.binding import Binding

from .models import AssignmentState, Task, TaskStatus
from .runtime.base import RunHandle, RunState, Runtime
from .screens.config import ConfigScreen
from .screens.schedule import ScheduleScreen
from .services.config_repo import DEFAULT_COMMAND_TEMPLATE, ConfigRepository
from .services.gpu_discovery import MockGPUProvider, make_provider
from .services.scheduler import AssignmentService
from .services.task_store import TaskStore, seed_demo
from .storage.sqlite import SqliteStorage
from .widgets.timeline_state import DEFAULT_CARD_WIDTH, MIN_CARD_WIDTH


class GPUSchedulerApp(App):
    TITLE = "gpu-scheduler"
    CSS_PATH = "styles/app.tcss"
    MODES = {"schedule": ScheduleScreen, "configs": ConfigScreen}
    DEFAULT_MODE = "schedule"
    BINDINGS = [
        Binding("1,f1", "switch_mode('schedule')", "Schedule", show=False),
        Binding("2,f2", "switch_mode('configs')", "Configs", show=False),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(
        self,
        storage: SqliteStorage | None = None,
        mock_gpus: int | None = None,
        config_dir: str | Path = "configs",
        demo: bool = False,
        runtime: Runtime | None = None,
        command_template: str = DEFAULT_COMMAND_TEMPLATE,
        stats_interval: float = 2.0,
        card_width: int = DEFAULT_CARD_WIDTH,
    ) -> None:
        super().__init__()
        self.provider = make_provider(mock_gpus)
        self.gpu_source = "mock" if isinstance(self.provider, MockGPUProvider) else "NVML"
        self.gpus = self.provider.discover()
        self.store = TaskStore(storage or SqliteStorage(":memory:"))
        self.assignment_service = AssignmentService(self.store, [g.id for g in self.gpus])
        if demo:
            seed_demo(self.store)
        self.config_repo = ConfigRepository(config_dir)
        self.command_template = command_template
        self.runtime = runtime
        self.stats_interval = stats_interval
        self.card_width = max(MIN_CARD_WIDTH, card_width)
        self.assignment_service.recompute()  # "now" moved since the plan was saved
        self._runs: dict[str, RunHandle] = {}

    def on_mount(self) -> None:
        self.theme = "textual-dark"
        if self.stats_interval:
            self.set_interval(self.stats_interval, self.tick)

    def tick(self) -> None:
        self.provider.refresh(self.gpus)
        self.poll_runs()
        self.assignment_service.recompute(persist=False)  # queued starts follow "now"
        screen = self.screen
        if isinstance(screen, ScheduleScreen) and screen.is_mounted:
            screen.refresh_timeline()

    # execution (phase-2 hook; off unless --runtime is given) ---------------
    def run_selected(self, task: Task | None) -> None:
        if self.runtime is None or task is None:
            return
        a = self.store.active_assignment(task.id)
        if a is None or a.state != AssignmentState.PLANNED:
            self.notify("Place the task on the timeline before running it.", severity="warning")
            return
        handle = self.runtime.submit(task, a)
        a.state = AssignmentState.RUNNING
        a.actual_start = a.actual_start or datetime.now()
        self._runs[a.id] = handle
        task.status = TaskStatus.RUNNING
        self.store.update_task(task, quiet=True)
        self.store.save_assignment(a, "started")
        self.assignment_service.recompute()
        self.store.notify()
        self.notify(f"Started {task.name} via {self.runtime.name}")

    def poll_runs(self) -> None:
        if not self._runs or self.runtime is None:
            return
        for aid, handle in list(self._runs.items()):
            state = self.runtime.status(handle)
            if state in (RunState.PENDING, RunState.RUNNING):
                continue
            a = self.store.assignments.get(aid)
            del self._runs[aid]
            if a is None:
                continue
            a.actual_end = datetime.now()
            if state == RunState.SUCCEEDED and hasattr(self.runtime, "result"):
                # Ray picks the physical devices; record what it actually used
                gpu_ids = self.runtime.result(handle).get("gpu_ids")
                if gpu_ids:
                    a.actual_gpu_ids = tuple(gpu_ids)
            a.state = {
                RunState.SUCCEEDED: AssignmentState.COMPLETED,
                RunState.CANCELLED: AssignmentState.CANCELLED,
            }.get(state, AssignmentState.FAILED)
            task = self.store.tasks.get(a.task_id)
            if task:
                task.status = {
                    AssignmentState.COMPLETED: TaskStatus.COMPLETED,
                    AssignmentState.CANCELLED: TaskStatus.CANCELLED,
                }.get(a.state, TaskStatus.FAILED)
                self.store.update_task(task, quiet=True)
            self.store.save_assignment(a, a.state.value)
            self.assignment_service.recompute()
            self.store.notify()
