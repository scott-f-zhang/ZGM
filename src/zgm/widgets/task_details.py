"""TaskDetails: full information about the selected task and its placement."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from rich.table import Table
from rich.text import Text
from textual.widgets import Static

from ..models import Assignment, AssignmentState, Task, format_estimate
from .palette import task_color
from .task_card import STATUS_STYLE


def _fmt(t: datetime | None) -> str:
    return t.strftime("%m-%d %H:%M") if t else "—"


def _gpus(ids) -> str:
    return ",".join(str(g) for g in ids) if ids else "—"


class TaskDetails(Static):
    def show(
        self,
        task: Task | None,
        assignments: list[Assignment] | None = None,
        note: Text | str | None = None,
        log: Path | None = None,
        blocked: set[int] | None = None,
    ) -> None:
        if task is None:
            self.update(Text("No task selected.", style="dim"))
            return
        label, style = STATUS_STYLE[task.status]
        grid = Table.grid(padding=(0, 2), expand=True)
        grid.add_column(style="dim", no_wrap=True)
        grid.add_column(ratio=2)
        grid.add_column(style="dim", no_wrap=True)
        grid.add_column(ratio=3)

        name = Text()
        name.append("█ ", style=task_color(task.id))
        name.append(task.name, style="bold")
        grid.add_row("Task", name, "Status", Text(label, style=style))
        grid.add_row(
            "GPUs",
            str(task.required_gpus),
            "Estimate",
            Text(format_estimate(task, long=True),
                 style="" if task.estimated_duration else "dim italic"),
        )
        spec = task.launch
        grid.add_row("Run", Text(spec.describe(), overflow="fold"), "ID", task.id)
        grid.add_row(
            "Environment",
            Text(spec.environment or "python on PATH", overflow="fold",
                 style="" if spec.environment else "dim"),
            "Working dir",
            Text(spec.cwd or "zgm's working dir", overflow="fold",
                 style="" if spec.cwd else "dim"),
        )
        if spec.env:
            grid.add_row("Env vars", Text(spec.env_text, overflow="fold"), "", "")
        if task.config_path:
            grid.add_row("Config", Text(task.config_path, overflow="fold"), "", "")

        active = [a for a in (assignments or []) if a.is_active]
        hist = [a for a in (assignments or []) if not a.is_active]
        if active:
            a = active[0]
            if a.state == AssignmentState.RUNNING:
                when = f"started {_fmt(a.actual_start)}, " + (
                    f"ends ~{_fmt(a.planned_end)}" if a.planned_end else "end unknown")
            elif a.planned_start is None and set(a.gpu_ids) & (blocked or set()):
                gpus = ",".join(map(str, sorted(set(a.gpu_ids) & blocked)))
                when = f"start unknown (GPU {gpus} is in use outside zgm)"
            elif a.planned_start is None:
                when = "start unknown (work ahead of it has no run history)"
            else:
                when = f"~{_fmt(a.planned_start)} → " + (
                    f"~{_fmt(a.planned_end)}" if a.planned_end else "end unknown")
            grid.add_row(
                "Running" if a.state == AssignmentState.RUNNING else "Planned",
                when,
                "GPU pref / actual",
                f"{_gpus(a.preferred_gpu_ids)} / {_gpus(a.actual_gpu_ids)}",
            )
        else:
            grid.add_row("Planned", Text("waiting in the queue — press a to place it", style="dim"),
                         "", "")
        for a in hist[-2:]:
            grid.add_row(
                "Ran",
                f"{_fmt(a.actual_start)} → {_fmt(a.actual_end)}",
                "on GPU",
                f"{_gpus(a.display_gpu_ids)} ({a.state.value})",
            )
        if log:
            grid.add_row("Log", Text(str(log), overflow="fold"), "", "")
        if note:
            grid.add_row("", note if isinstance(note, Text) else Text(note), "", "")
        self.update(grid)
