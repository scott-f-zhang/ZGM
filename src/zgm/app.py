"""ZGMApp: wires services to the two views (Schedule, Configs)."""

from __future__ import annotations

from pathlib import Path

from textual.app import App
from textual.binding import Binding

from .models import Task
from .runtime.base import Runtime
from .screens.config import ConfigScreen
from .screens.schedule import ScheduleScreen
from .services.config_repo import DEFAULT_COMMAND_TEMPLATE, ConfigRepository
from .services.executor import Executor
from .services.gpu_discovery import MockGPUProvider, make_provider
from .services.gpu_usage import GPUWatch
from .services.lease import ExecutorLease
from .services.scheduler import AssignmentService
from .services.task_store import TaskStore, seed_demo
from .storage.sqlite import SqliteStorage
from .widgets.timeline_state import DEFAULT_CARD_WIDTH, MIN_CARD_WIDTH


class ZGMApp(App):
    TITLE = "ZGM"
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
        autostart: bool | None = None,
    ) -> None:
        super().__init__()
        self.provider = make_provider(mock_gpus)
        self.gpu_source = "mock" if isinstance(self.provider, MockGPUProvider) else "NVML"
        self.gpus = self.provider.discover()
        self.store = TaskStore(storage or SqliteStorage(":memory:"))
        self.assignment_service = AssignmentService(self.store, [g.id for g in self.gpus])
        self.gpu_watch = GPUWatch(self.provider, self.gpus)
        self.assignment_service.blocked, self._gpu_changes = self.gpu_watch.refresh(self.store)
        if demo:
            seed_demo(self.store)
        self.config_repo = ConfigRepository(config_dir)
        self.command_template = command_template
        self.lease = ExecutorLease(self.store.storage.path)
        self._holder: dict | None = None
        self.executor = Executor(self.store, self.assignment_service, notify=self._notify)
        self.runtime = runtime  # runs the queue if no other zgm does (see lead())
        if autostart is not None:
            self.autostart = autostart
        self.stats_interval = stats_interval
        self.card_width = max(MIN_CARD_WIDTH, card_width)
        self.assignment_service.recompute()  # "now" moved since the plan was saved

    def _notify(self, message: str, severity: str = "information") -> None:
        self.notify(message, severity=severity, timeout=15 if severity == "error" else 5)

    def on_mount(self) -> None:
        self.theme = "textual-dark"
        self._holder = self.lease.holder()
        for message in self._gpu_changes:
            self.notify(message, severity="warning", timeout=10)
        if self.is_executor:
            self.executor.adopt()
        if self.stats_interval:
            self.set_interval(self.stats_interval, self.tick)

    def on_unmount(self) -> None:
        self.lease.release()  # tasks keep running; the next executor attaches to them

    # who runs the queue ---------------------------------------------------------
    @property
    def runtime(self) -> Runtime | None:
        """This TUI's own runtime (--runtime), whether or not it is the executor."""
        return self._runtime

    @runtime.setter
    def runtime(self, runtime: Runtime | None) -> None:
        self._runtime = runtime
        self.executor.runtime = runtime
        self.lead()

    @property
    def is_executor(self) -> bool:
        return self.lease.held

    def lead(self) -> bool:
        """Become the executor if this TUI has a runtime and nobody else is one."""
        if self._runtime is None or self.lease.held:
            return self.lease.held
        log_dir = getattr(self._runtime, "log_dir", None)
        led = self.lease.acquire("tui", self._runtime.name, str(log_dir) if log_dir else None)
        self._holder = self.lease.holder()
        return led

    def executor_info(self) -> dict | None:
        """The process running the queue (maybe this one), or None. Checked on
        every tick; cached in between, since the status line asks often."""
        if self.lease.held:
            return self.lease.holder()
        return self._holder

    @property
    def can_control_runs(self) -> bool:
        """r/p/s work: this TUI runs the queue, or another zgm does."""
        return self.is_executor or self.executor_info() is not None

    def _client_runtime(self) -> None:
        """While another zgm executes, stopping a local run still works from here."""
        if self.is_executor or self.executor.runtime is not None:
            return
        info = self.executor_info()
        if info and info.get("runtime") == "local":
            from .runtime.local import LocalRuntime

            self.executor.runtime = LocalRuntime(info.get("log_dir") or ".")

    def tick(self) -> None:
        self.provider.refresh(self.gpus)
        if self.store.sync():  # changed by another zgm (submit, daemon, another TUI)
            self.assignment_service.recompute()
        self.assignment_service.blocked, changes = self.gpu_watch.refresh(self.store)
        for message in changes:
            self.notify(message, severity="warning", timeout=10)
        self._holder = self.lease.holder()
        if not self.is_executor and self.lead():
            self.notify("This zgm now runs the queue.")
        if self.is_executor:
            self.executor.tick(hold=self._placing())
        else:
            self._client_runtime()
        self.assignment_service.recompute(persist=False)  # queued starts follow "now"
        screen = self.screen
        if isinstance(screen, ScheduleScreen) and screen.is_mounted:
            screen.refresh_timeline()

    def _placing(self) -> bool:
        """Nothing starts while the user is placing a card, so it can't start
        out from under them."""
        screen = self.screen
        return isinstance(screen, ScheduleScreen) and screen.placement is not None

    # actions (from the Schedule view) ---------------------------------------------
    @property
    def autostart(self) -> bool:
        return self.executor.autostart

    @autostart.setter
    def autostart(self, on: bool) -> None:
        self.executor.autostart = on

    def set_autostart(self, on: bool) -> None:
        self.autostart = on
        if on:
            self.dispatch()

    def dispatch(self) -> list[Task]:
        if not self.is_executor or self._placing():
            return []
        return self.executor.dispatch()

    def poll_runs(self) -> None:
        if self.is_executor:
            self.executor.poll()

    @property
    def _runs(self):
        return self.executor.runs

    def run_selected(self, task: Task | None) -> None:
        """``r``: start this task ahead of the queue as soon as its GPUs are free."""
        if task is None or not self.can_control_runs:
            return
        message, severity = self.executor.request_run_now(task)
        self.notify(message, severity=severity)
        self.dispatch()

    def stop_task(self, task: Task, requeue: bool) -> bool:
        self._client_runtime()
        return self.executor.stop(task, requeue)

    def log_path(self, task: Task) -> Path | None:
        """The log of the task's latest run, if it wrote one."""
        runs = [a for a in self.store.assignments_for_task(task.id) if a.actual_start]
        if not runs:
            return None
        latest = max(runs, key=lambda a: a.actual_start)
        if latest.log_path:
            path = Path(latest.log_path)
        elif self._runtime is not None and hasattr(self._runtime, "log_path"):
            path = self._runtime.log_path(latest.id)
        else:
            return None
        return path if path.exists() else None

    def config_renamed(self, old: Path, new: Path) -> int:
        """Point tasks created from a renamed config at its new file."""
        count = 0
        for t in self.store.tasks.values():
            if t.config_path and Path(t.config_path) == old:
                t.config_path = str(new)
                spec = t.launch
                if spec.command:  # made from --command-template: the path is in it
                    spec.command = spec.command.replace(str(old), str(new))
                spec.args = [arg.replace(str(old), str(new)) for arg in spec.args]
                t.command = spec.describe()
                self.store.update_task(t, quiet=True)
                count += 1
        if count:
            self.store.notify()
        return count
