"""GPUTimeline: time axis plus one GPURow per GPU, with keyboard navigation."""

from __future__ import annotations

from datetime import datetime

from rich.segment import Segment
from rich.style import Style
from textual import events
from textual.binding import Binding
from textual.containers import Vertical
from textual.message import Message
from textual.strip import Strip
from textual.widget import Widget

from ..models import Assignment, GPUDevice, Task
from .gpu_row import LABEL_WIDTH, Ghost, GPURow
from .timeline_state import TimelineState


class TimeAxis(Widget):
    """Two-line header: date/hour labels and tick marks."""

    def __init__(self, state: TimelineState, **kw) -> None:
        super().__init__(**kw)
        self.state = state
        self.now = datetime.now()

    def render_line(self, y: int) -> Strip:
        width = max(0, self.size.width - LABEL_WIDTH)
        st = self.state
        dim = Style(color="#6b7280")
        head = (
            f" {st.minutes_per_cell}m/cell  {st.view_start:%m-%d %H:%M}"
            if y == 0
            else f" cursor {st.cursor_time:%a %H:%M}"
        )
        segs = [Segment(f"{head:<{LABEL_WIDTH - 1}.{LABEL_WIDTH - 1}}│", dim)]
        line = [" "] * width
        # label every N cells so labels don't collide
        step_min = next(
            m for m in (15, 30, 60, 120, 180, 360, 720, 1440, 2880)
            if m / st.minutes_per_cell >= 7 or m == 2880
        )
        for x in range(width):
            t = st.x_to_time(x)
            mins = t.hour * 60 + t.minute
            if mins % step_min < st.minutes_per_cell:  # first cell of a label interval
                if y == 0:
                    label = t.strftime("%m-%d") if mins < st.minutes_per_cell else t.strftime("%H:%M")
                    for i, ch in enumerate(label):
                        if x + i < width:
                            line[x + i] = ch
                else:
                    line[x] = "┴" if mins < st.minutes_per_cell else "╵"
            elif y == 1:
                line[x] = "─"
        now_x = int(st.time_to_x(self.now) // 1)
        text = "".join(line)
        if y == 1 and 0 <= now_x < width:
            segs.append(Segment(text[:now_x], dim))
            segs.append(Segment("▼", Style(color="#f43f5e", bold=True)))
            segs.append(Segment(text[now_x + 1:], dim))
        else:
            segs.append(Segment(text, Style(color="#9ca3af")))
        return Strip(segs)


class GPUTimeline(Vertical, can_focus=True):
    BINDINGS = [
        Binding("left", "cursor(-1)", "◀ time", show=False),
        Binding("right", "cursor(1)", "time ▶", show=False),
        Binding("up,k", "gpu(-1)", "GPU ↑", show=False),
        Binding("down,j", "gpu(1)", "GPU ↓", show=False),
        Binding("h,shift+left", "pan(-1)", "Pan ◀", show=False),
        Binding("l,shift+right", "pan(1)", "Pan ▶", show=False),
        Binding("plus,equals_sign,=", "zoom(-1)", "Zoom in"),
        Binding("minus,-", "zoom(1)", "Zoom out"),
        Binding("t", "now", "Now"),
    ]

    class CursorMoved(Message):
        pass

    def __init__(self, gpus: list[GPUDevice], **kw) -> None:
        super().__init__(**kw)
        self.gpus = gpus
        self.state = TimelineState()
        self.state.jump_to(datetime.now())
        self.axis = TimeAxis(self.state, id="time-axis")
        self.rows = [GPURow(g, self.state) for g in gpus]
        self._data: tuple = ([], {}, None)

    def compose(self):
        yield self.axis
        with Vertical(id="gpu-rows"):
            yield from self.rows

    def on_resize(self, event: events.Resize) -> None:
        self.state.width = max(1, self.size.width - LABEL_WIDTH)
        self.refresh_rows()

    def row_for_gpu(self, gpu_id: int) -> GPURow:
        return next(r for r in self.rows if r.gpu.id == gpu_id)

    def update_data(
        self,
        assignments: list[Assignment],
        tasks: dict[str, Task],
        ghost: Ghost | None = None,
    ) -> None:
        self._data = (assignments, tasks, ghost)
        self.refresh_rows()

    def refresh_rows(self) -> None:
        assignments, tasks, ghost = self._data
        now = datetime.now()
        self.axis.now = now
        self.axis.refresh()
        for row in self.rows:
            on_row = [a for a in assignments if row.gpu.id in a.display_gpu_ids]
            row.set_data(on_row, tasks, ghost, now)

    def assignment_under_cursor(self) -> Assignment | None:
        if not self.rows:
            return None
        return self.row_for_gpu(self.state.cursor_gpu).assignment_at_column(self.state.cursor_x)

    def locate(self, screen_x: int, screen_y: int) -> tuple[int, datetime] | None:
        """Screen coordinates -> (gpu id, snapped time), or None if outside the strip."""
        for row in self.rows:
            r = row.region
            if r.y <= screen_y < r.y + r.height:
                col = row.column_at(screen_x)
                if 0 <= col < self.state.width:
                    return row.gpu.id, self.state.x_to_time(col)
                return None
        return None

    # actions ---------------------------------------------------------------
    def _changed(self) -> None:
        self.refresh_rows()
        self.post_message(self.CursorMoved())

    def action_cursor(self, delta: int) -> None:
        self.state.move_cursor(delta)
        self._changed()

    def action_gpu(self, delta: int) -> None:
        if not self.gpus:
            return
        ids = [g.id for g in self.gpus]
        idx = ids.index(self.state.cursor_gpu) if self.state.cursor_gpu in ids else 0
        self.state.cursor_gpu = ids[max(0, min(len(ids) - 1, idx + delta))]
        self._changed()

    def action_pan(self, direction: int) -> None:
        step = max(1, self.state.width // 4)
        self.state.pan(direction * step)
        self.state.cursor_time += self.state.cell * direction * step
        self._changed()

    def action_zoom(self, step: int) -> None:
        self.state.zoom(step)
        self._changed()

    def action_now(self) -> None:
        self.state.jump_to(datetime.now())
        self._changed()

    def set_cursor(self, gpu_id: int, t: datetime) -> None:
        self.state.cursor_gpu = gpu_id
        self.state.set_cursor_time(t)
        self._changed()

    def on_mouse_scroll_down(self, event: events.MouseScrollDown) -> None:
        self.state.pan(2)
        self.state.cursor_time += self.state.cell * 2
        self._changed()
        event.stop()

    def on_mouse_scroll_up(self, event: events.MouseScrollUp) -> None:
        self.state.pan(-2)
        self.state.cursor_time -= self.state.cell * 2
        self._changed()
        event.stop()

    def on_gpurow_hover(self, event: GPURow.Hover) -> None:
        self.state.cursor_gpu = event.gpu_id
        self.state.cursor_time = self.state.x_to_time(event.column)
        self._changed()
