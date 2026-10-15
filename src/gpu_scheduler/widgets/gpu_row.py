"""GPURow: one GPU's label column plus its Gantt strip."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from rich.segment import Segment
from rich.style import Style
from textual import events
from textual.message import Message
from textual.strip import Strip
from textual.widget import Widget

from ..models import Assignment, GPUDevice, Task, TaskStatus
from ..models.gpu import format_bytes_gib
from .palette import GHOST_BAD, GHOST_OK, HISTORY_COLOR, task_color
from .timeline_state import TimelineState

LABEL_WIDTH = 24
DRAG_THRESHOLD = 1


@dataclass
class Ghost:
    """Preview of a placement in progress."""

    task_id: str
    name: str
    gpu_ids: tuple[int, ...]
    start: datetime
    end: datetime
    ok: bool
    assignment_id: str | None = None  # set when moving an existing block


@dataclass
class Block:
    x0: int
    x1: int
    label: str
    bg: str
    fg: str
    assignment_id: str | None
    bold: bool = False


def _bar(frac: float, width: int) -> str:
    full = int(round(frac * width))
    return "▮" * full + "▯" * (width - full)


class GPURow(Widget):
    """Rendered line-by-line; the timeline owns state and passes it in."""

    class BlockPressed(Message):
        def __init__(self, gpu_id: int, assignment_id: str | None, x: int, y: int) -> None:
            super().__init__()
            self.gpu_id, self.assignment_id, self.x, self.y = gpu_id, assignment_id, x, y

    class BlockDragMove(Message):
        def __init__(self, assignment_id: str, x: int, y: int) -> None:
            super().__init__()
            self.assignment_id, self.x, self.y = assignment_id, x, y

    class BlockDragEnd(Message):
        def __init__(self, assignment_id: str, x: int, y: int) -> None:
            super().__init__()
            self.assignment_id, self.x, self.y = assignment_id, x, y

    class Hover(Message):
        def __init__(self, gpu_id: int, column: int) -> None:
            super().__init__()
            self.gpu_id, self.column = gpu_id, column

    def __init__(self, gpu: GPUDevice, state: TimelineState, **kw) -> None:
        super().__init__(id=f"gpu-row-{gpu.id}", **kw)
        self.gpu = gpu
        self.state = state
        self.assignments: list[Assignment] = []
        self.tasks: dict[str, Task] = {}
        self.ghost: Ghost | None = None
        self.now = datetime.now()
        self._press: tuple[int, int, str] | None = None
        self._dragging = False

    # data ------------------------------------------------------------------
    def set_data(
        self,
        assignments: list[Assignment],
        tasks: dict[str, Task],
        ghost: Ghost | None,
        now: datetime,
    ) -> None:
        self.assignments = assignments
        self.tasks = tasks
        self.ghost = ghost if ghost and self.gpu.id in ghost.gpu_ids else None
        self.now = now
        self.refresh()

    def column_at(self, screen_x: int) -> int:
        return screen_x - self.region.x - LABEL_WIDTH

    def assignment_at_column(self, col: int) -> Assignment | None:
        t = self.state.x_to_time(col)
        t_end = t + self.state.cell
        for a in self.assignments:
            s, e = a.display_interval(self.now)
            if s < t_end and t < e:
                return a
        return None

    # rendering -------------------------------------------------------------
    def _blocks(self) -> list[Block]:
        blocks = []
        for a in sorted(self.assignments, key=lambda a: a.display_interval(self.now)[0]):
            s, e = a.display_interval(self.now)
            span = self.state.span(s, e)
            if span is None:
                continue
            task = self.tasks.get(a.task_id)
            name = task.name if task else a.task_id
            if a.is_historical:
                bg, fg = HISTORY_COLOR, "#e5e7eb"
            else:
                bg, fg = task_color(a.task_id), "#0b0f19"
            if task and task.status == TaskStatus.RUNNING:
                name = "▶ " + name
            if self.gpu.id != min(a.display_gpu_ids):  # label multi-GPU blocks once
                name = "┆"
            blocks.append(Block(span[0], span[1], name, bg, fg, a.id))
        g = self.ghost
        if g:
            span = self.state.span(g.start, g.end)
            if span:
                bg = GHOST_OK if g.ok else GHOST_BAD
                mark = "✓ " if g.ok else "✗ "
                label = mark + g.name if self.gpu.id == min(g.gpu_ids) else "┆"
                blocks.append(Block(span[0], span[1], label, bg, "#0b0f19", None, True))
        return blocks

    def _label_line(self, y: int, height: int) -> list[Segment]:
        g = self.gpu
        cursor_row = self.state.cursor_gpu == g.id
        base = Style(bgcolor="#1f2937" if cursor_row else None)
        if height == 1:
            txt = f"{'▶' if cursor_row else ' '}{g.id} {g.short_name[:9]:<9} {g.utilization:>3d}% "
            txt += f"{format_bytes_gib(g.memory_used)}/{format_bytes_gib(g.memory_total)}"
            return [Segment(f"{txt:<{LABEL_WIDTH - 1}.{LABEL_WIDTH - 1}}│", base)]
        lines = [
            [(f"{'▶' if cursor_row else ' '}GPU{g.id} ", base + Style(bold=True)),
             (g.short_name, base + Style(color="#9ca3af"))],
            [(" mem ", base + Style(dim=True)),
             (_bar(g.memory_fraction, 8), base + Style(color="#a78bfa")),
             (f" {format_bytes_gib(g.memory_used)}/{format_bytes_gib(g.memory_total)}", base)],
            [(" util", base + Style(dim=True)),
             (" " + _bar(g.utilization / 100, 8), base + Style(color="#34d399")),
             (f" {g.utilization}%", base)],
        ]
        parts = lines[y] if y < len(lines) else []
        segs = []
        used = 0
        for text, style in parts:
            text = text[: max(0, LABEL_WIDTH - 1 - used)]
            segs.append(Segment(text, style))
            used += len(text)
        segs.append(Segment(" " * (LABEL_WIDTH - 1 - used), base))
        segs.append(Segment("│", Style(color="#374151")))
        return segs

    def render_line(self, y: int) -> Strip:
        height = self.size.height
        width = max(0, self.size.width - LABEL_WIDTH)
        segs = self._label_line(y, height)

        cells: list[tuple[str, Style]] = []
        now_x = int(self.state.time_to_x(self.now) // 1)
        cell_min = self.state.minutes_per_cell
        for x in range(width):
            t = self.state.x_to_time(x)
            past = x < now_x
            hour_mark = (t.hour * 60 + t.minute) % max(60, cell_min * 6) == 0
            ch = "┊" if hour_mark and y == height - 1 else " "
            bg = "#111827" if past else "#0b0f19"
            if t.hour == 0 and t.minute < cell_min:
                ch = "┃"
            cells.append((ch, Style(bgcolor=bg, color="#374151")))

        mid = height // 2
        for b in self._blocks():
            for x in range(b.x0, b.x1):
                left_edge = x == b.x0
                cells[x] = ("▏" if left_edge and not b.bold else " ",
                            Style(bgcolor=b.bg, color=b.fg, bold=b.bold))
            if y == mid:
                label = b.label[: max(0, b.x1 - b.x0 - 1)]
                for i, ch in enumerate(label):
                    cells[b.x0 + 1 + i] = (ch, Style(bgcolor=b.bg, color=b.fg, bold=True))

        if 0 <= now_x < width:
            ch, st = cells[now_x]
            cells[now_x] = ("│" if ch == " " else ch, st + Style(color="#f43f5e", bold=True))

        cx = self.state.cursor_x
        if self.state.cursor_gpu == self.gpu.id and 0 <= cx < width:
            ch, st = cells[cx]
            cells[cx] = ("▎" if ch == " " else ch, st + Style(reverse=True))

        # merge runs of identical style to keep segment count small
        run_text, run_style = "", None
        for ch, st in cells:
            if st == run_style:
                run_text += ch
            else:
                if run_text:
                    segs.append(Segment(run_text, run_style))
                run_text, run_style = ch, st
        if run_text:
            segs.append(Segment(run_text, run_style))
        return Strip(segs)

    # mouse -----------------------------------------------------------------
    def on_mouse_move(self, event: events.MouseMove) -> None:
        if self._press is not None:
            dx = abs(event.screen_x - self._press[0])
            dy = abs(event.screen_y - self._press[1])
            if not self._dragging and max(dx, dy) >= DRAG_THRESHOLD:
                self._dragging = True
            if self._dragging:
                self.post_message(self.BlockDragMove(self._press[2], event.screen_x, event.screen_y))
            return
        col = self.column_at(event.screen_x)
        if col >= 0:
            self.post_message(self.Hover(self.gpu.id, col))

    def on_mouse_down(self, event: events.MouseDown) -> None:
        if event.button != 1:
            return
        col = self.column_at(event.screen_x)
        a = self.assignment_at_column(col) if col >= 0 else None
        self.post_message(self.BlockPressed(self.gpu.id, a.id if a else None,
                                            event.screen_x, event.screen_y))
        if a is not None and a.is_active and not a.actual_start:
            self._press = (event.screen_x, event.screen_y, a.id)
            self._dragging = False
            self.capture_mouse()
        event.stop()

    def on_mouse_up(self, event: events.MouseUp) -> None:
        if self._press is None:
            return
        aid = self._press[2]
        dragged = self._dragging
        self._press = None
        self._dragging = False
        self.capture_mouse(False)
        if dragged:
            self.post_message(self.BlockDragEnd(aid, event.screen_x, event.screen_y))
        event.stop()
