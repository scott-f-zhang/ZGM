"""TaskCard: one task in the horizontal queue. Click selects, drag places."""

from __future__ import annotations

from rich.text import Text
from textual import events
from textual.message import Message
from textual.reactive import reactive
from textual.widget import Widget

from ..models import Task, TaskStatus, format_estimate
from . import theme
from .palette import task_color

STATUS_STYLE = {
    TaskStatus.PENDING: ("pending", f"bold {theme.WARN}"),
    TaskStatus.PLANNED: ("planned", f"bold {theme.CACHED[1]}"),
    TaskStatus.RUNNING: ("running", f"bold {theme.OK}"),
    TaskStatus.COMPLETED: ("done", theme.OK),
    TaskStatus.FAILED: ("failed", f"bold {theme.BAD}"),
    TaskStatus.CANCELLED: ("cancelled", theme.SUBTLE),
}

DRAG_THRESHOLD = 2  # cells of movement before a press becomes a drag


class TaskCard(Widget):
    selected: reactive[bool] = reactive(False)

    class Selected(Message):
        def __init__(self, task_id: str) -> None:
            super().__init__()
            self.task_id = task_id

    class DragMove(Message):
        def __init__(self, task_id: str, x: int, y: int) -> None:
            super().__init__()
            self.task_id, self.x, self.y = task_id, x, y

    class DragEnd(Message):
        def __init__(self, task_id: str, x: int, y: int, dragged: bool) -> None:
            super().__init__()
            self.task_id, self.x, self.y, self.dragged = task_id, x, y, dragged

    def __init__(self, task: Task, **kw) -> None:
        super().__init__(id=f"card-{task.id}", **kw)
        self.task_model = task
        self._press: tuple[int, int] | None = None
        self._dragging = False

    def update_task(self, task: Task) -> None:
        self.task_model = task
        self.set_class(task.status != TaskStatus.PENDING, "-scheduled")
        self.refresh()

    def on_mount(self) -> None:
        self.update_task(self.task_model)

    def watch_selected(self, value: bool) -> None:
        self.set_class(value, "-selected")

    def render(self) -> Text:
        t = self.task_model
        color = task_color(t.id)
        label, style = STATUS_STYLE[t.status]
        width = max(8, self.size.width)
        out = Text(no_wrap=True, overflow="ellipsis")
        out.append("█ ", style=color)
        name = t.name if len(t.name) <= width - 2 else t.name[: width - 3] + "…"
        out.append(name + "\n", style=f"bold {theme.TITLE}")
        out.append(f"gpu×{t.required_gpus}", style=theme.DOWNLOAD[2])
        out.append("  ")
        known = t.estimated_duration is not None
        out.append(f"⏱ {format_estimate(t)}\n", style=theme.MAIN_FG if known else theme.SUBTLE)
        out.append(label, style=style)
        if t.config_path:
            out.append("  ⚙", style=theme.SUBTLE)
        return out

    # mouse: press selects, press+move drags onto the timeline ---------------
    def on_mouse_down(self, event: events.MouseDown) -> None:
        if event.button != 1:
            return
        self._press = (event.screen_x, event.screen_y)
        self._dragging = False
        self.post_message(self.Selected(self.task_model.id))
        self.capture_mouse()
        event.stop()

    def on_mouse_move(self, event: events.MouseMove) -> None:
        if self._press is None:
            return
        dx = abs(event.screen_x - self._press[0])
        dy = abs(event.screen_y - self._press[1])
        if not self._dragging and max(dx, dy) >= DRAG_THRESHOLD:
            self._dragging = True
            self.add_class("-dragging")
        if self._dragging:
            self.post_message(self.DragMove(self.task_model.id, event.screen_x, event.screen_y))

    def on_mouse_up(self, event: events.MouseUp) -> None:
        if self._press is None:
            return
        dragged = self._dragging
        self._press = None
        self._dragging = False
        self.remove_class("-dragging")
        self.capture_mouse(False)
        if dragged:
            self.post_message(
                self.DragEnd(self.task_model.id, event.screen_x, event.screen_y, dragged)
            )
        event.stop()
