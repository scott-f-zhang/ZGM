"""TaskDetails: full information about the selected task and its placement."""

from __future__ import annotations

from datetime import datetime

from rich.table import Table
from rich.text import Text
from textual.widgets import Static

from ..models import Assignment, Task, format_duration
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
            format_duration(task.estimated_duration),
        )
        grid.add_row("Command", Text(task.command, overflow="fold"), "ID", task.id)
        if task.config_path:
            grid.add_row("Config", Text(task.config_path, overflow="fold"), "", "")

        active = [a for a in (assignments or []) if a.is_active]
        hist = [a for a in (assignments or []) if not a.is_active]
        if active:
            a = active[0]
            grid.add_row(
                "Planned" if not a.actual_start else "Expected",
                f"~{_fmt(a.planned_start)} → ~{_fmt(a.planned_end)} (estimate)",
                "GPU pref / actual",
                f"{_gpus(a.preferred_gpu_ids)} / {_gpus(a.actual_gpu_ids)}",
            )
            if a.actual_start:
                grid.add_row("Started", _fmt(a.actual_start), "", "")
        else:
            grid.add_row("Planned", Text("not on the timeline — press a to place", style="dim"),
                         "", "")
        for a in hist[-2:]:
            grid.add_row(
                "Ran",
                f"{_fmt(a.actual_start)} → {_fmt(a.actual_end)}",
                "on GPU",
                f"{_gpus(a.display_gpu_ids)} ({a.state.value})",
            )
        if note:
            grid.add_row("", note if isinstance(note, Text) else Text(note), "", "")
        self.update(grid)
