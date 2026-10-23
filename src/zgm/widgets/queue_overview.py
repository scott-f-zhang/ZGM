"""QueueOverview: the waiting tasks, shown when the queue is full screen. Two
sections: tasks marked for auto-fill (started on GPUs left idle) and the rest,
placed by hand. In each, tasks are grouped by where their config lives (its
folder under the configs directory); one line per task: name, state, GPUs, and
how its last run went."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from rich.text import Text
from textual.binding import Binding
from textual.message import Message
from textual.widgets import OptionList
from textual.widgets.option_list import Option

from ..models import Assignment, Task, format_duration
from . import theme
from .task_card import STATUS_STYLE, priority_label

SECTIONS = [(True, "auto-fill", "starts on GPUs left idle, highest priority first · space: move to manual"),
            (False, "manual", "place with a · space: move to auto-fill")]
NO_CONFIG = "no config"


def config_group(task: Task, root: Path) -> str:
    """The folder of the task's config, relative to ``root`` ("." for files at
    the top); a config elsewhere is listed under its own folder's path."""
    if not task.config_path:
        return NO_CONFIG
    folder = Path(task.config_path).parent
    try:
        rel = folder.resolve().relative_to(root.resolve())
    except ValueError:
        return str(folder)
    return str(rel) if rel.parts else "."


class QueueOverview(OptionList):
    BINDINGS = [
        Binding("space", "toggle", "auto-fill ⇄ manual", show=False),
        Binding("A", "toggle_group", "whole folder ⇄", show=False),
        Binding("plus,equals_sign", "priority(1)", "priority +1", show=False),
        Binding("minus", "priority(-1)", "priority -1", show=False),
    ]

    class SelectionChanged(Message):
        def __init__(self, task_id: str) -> None:
            super().__init__()
            self.task_id = task_id

    class Toggle(Message):
        """Move these tasks to the other section (``autofill`` = where they go)."""
        def __init__(self, task_ids: list[str], autofill: bool) -> None:
            super().__init__()
            self.task_ids, self.autofill = task_ids, autofill

    class Priority(Message):
        """Raise (``delta`` > 0) or lower a task's priority."""
        def __init__(self, task_id: str, delta: int) -> None:
            super().__init__()
            self.task_id, self.delta = task_id, delta

    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        self.tasks: dict[str, Task] = {}
        self.group_of: dict[str, str] = {}

    def set_tasks(self, tasks: list[Task], root: Path,
                  last_run: Callable[[Task], Assignment | None],
                  selected: str | None = None) -> None:
        self.tasks = {t.id: t for t in tasks}
        self.group_of = {t.id: config_group(t, root) for t in tasks}
        width = max((len(t.name) for t in tasks), default=0)
        options: list[Option] = []
        for auto, title, hint in SECTIONS:
            # in the order auto-fill takes them: higher priority first, then oldest
            members = sorted((t for t in tasks if t.autofill == auto), key=lambda t: -t.priority)
            head = Text(f"{title.upper()}  {len(members)}", style=f"bold {theme.HI_FG}")
            head.append(f"   {hint}", style=theme.SUBTLE)
            if options:
                options.append(Option(Text(""), disabled=True))
            options.append(Option(head, disabled=True))
            groups: dict[str, list[Task]] = {}
            for t in members:
                groups.setdefault(self.group_of[t.id], []).append(t)
            for group in sorted(groups, key=lambda g: (g == NO_CONFIG, g)):
                label = f"{group}/" if group not in (".", NO_CONFIG) else group
                head = Text(f" {label}", style=f"bold {theme.TITLE}")
                head.append(f"  {len(groups[group])}", style=theme.SUBTLE)
                options.append(Option(head, disabled=True))
                options += [Option(self.line(t, width, last_run(t)), id=t.id)
                            for t in groups[group]]
        highlighted = self.highlighted_option.id if self.highlighted_option else None
        self.clear_options()
        self.add_options(options)
        target = selected or highlighted
        ids = [o.id for o in options if o.id]
        if ids:
            self.highlighted = self.get_option_index(target if target in ids else ids[0])

    def action_toggle(self) -> None:
        t = self.tasks.get(self.highlighted_option.id) if self.highlighted_option else None
        if t is not None:
            self.post_message(self.Toggle([t.id], not t.autofill))

    def action_priority(self, delta: int) -> None:
        if self.highlighted_option and self.highlighted_option.id:
            self.post_message(self.Priority(self.highlighted_option.id, delta))

    def action_toggle_group(self) -> None:
        """Move every task of the highlighted one's folder, in its section."""
        t = self.tasks.get(self.highlighted_option.id) if self.highlighted_option else None
        if t is None:
            return
        group = self.group_of[t.id]
        ids = [x.id for x in self.tasks.values()
               if self.group_of[x.id] == group and x.autofill == t.autofill]
        self.post_message(self.Toggle(ids, not t.autofill))

    @staticmethod
    def line(t: Task, width: int, run: Assignment | None) -> Text:
        label, style = STATUS_STYLE[t.status]
        text = Text("   ")
        text.append(t.name.ljust(width + 2), style=theme.MAIN_FG)
        text.append(label.ljust(10), style=style)
        label, style = priority_label(t.priority)
        text.append(label.ljust(5), style=style)
        text.append(f"{t.required_gpus} GPU{'s' if t.required_gpus != 1 else ' '}  ",
                    style=theme.SUBTLE)
        if run is None:
            text.append("never run", style=theme.SUBTLE)
        else:
            took = (f", {format_duration(run.actual_end - run.actual_start)}"
                    if run.actual_end and run.actual_start else "")
            text.append(f"last run {run.actual_start:%m-%d %H:%M} · {run.state.value}{took}",
                        style=theme.SUBTLE)
        return text

    def on_option_list_option_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        event.stop()
        if event.option.id:
            self.post_message(self.SelectionChanged(event.option.id))
