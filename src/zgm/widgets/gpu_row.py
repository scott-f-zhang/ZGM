"""GPURow: one GPU's label column plus its timeline strip.

Left of "now" the strip is to scale (history); right of "now" running work
shares the first column (its GPUs never overlap) and every queued assignment is
a fixed-width card in its own column after it. The timeline builds the
cards once and hands the same list to every row, so a multi-GPU card lines up
in the same column on each of its rows.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rich.segment import Segment
from rich.style import Style
from textual import events
from textual.message import Message
from textual.strip import Strip
from textual.widget import Widget

from ..models import GPUDevice
from ..models.gpu import format_bytes_gib
from . import theme
from .timeline_state import Hit, TimelineState, wrap_center

LABEL_WIDTH = 24
LABEL_LINES = 3
DRAG_THRESHOLD = 1
METER_WIDTH = 8

PAST_BG = theme.PAST_BG
FUTURE_BG = theme.MAIN_BG
NOW_COLOR = theme.NOW
GRID = theme.DIV_LINE


@dataclass
class Card:
    """A fixed-width block in the future zone (one per column)."""

    col: int
    key: str  # assignment id or the ghost key
    task_id: str
    gpu_ids: tuple[int, ...]
    lines: list[str]  # already centred to the card width
    bg: str
    fg: str = theme.MAIN_BG
    kind: str = "planned"  # running | planned | ghost
    selected: bool = False
    dimmed: bool = False  # original of a block that is being moved
    title_lines: int = 1  # leading lines that hold the task name (drawn bold)


@dataclass
class PastBlock:
    """A to-scale block in the past zone (actual times)."""

    key: str
    x0: int
    x1: int
    label: str
    bg: str
    fg: str = theme.MAIN_FG
    gpu_ids: tuple[int, ...] = field(default_factory=tuple)


def card_rows_needed(card: Card) -> int:
    return len(card.lines)


class GPURow(Widget):
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
        def __init__(self, gpu_id: int, hit: Hit) -> None:
            super().__init__()
            self.gpu_id, self.hit = gpu_id, hit

    def __init__(self, gpu: GPUDevice, state: TimelineState, **kw) -> None:
        super().__init__(id=f"gpu-row-{gpu.id}", **kw)
        self.gpu = gpu
        self.state = state
        self.cards: list[Card] = []
        self.past: list[PastBlock] = []
        self.prev_gpu: int | None = None
        self.next_gpu: int | None = None
        self.end_slot_col: int | None = None  # draw the "end of queue" placeholder here
        self._press: tuple[int, int, str] | None = None
        self._dragging = False

    # data ------------------------------------------------------------------
    def set_data(
        self,
        cards: list[Card],
        past: list[PastBlock],
        prev_gpu: int | None,
        next_gpu: int | None,
        end_slot_col: int | None,
    ) -> None:
        self.cards = [c for c in cards if self.gpu.id in c.gpu_ids]
        self.past = [b for b in past if self.gpu.id in b.gpu_ids]
        self.prev_gpu, self.next_gpu = prev_gpu, next_gpu
        self.end_slot_col = end_slot_col
        self.styles.height = self.needed_height()
        self.refresh()

    def shows_content(self, card: Card) -> bool:
        """Text goes on the top row of each contiguous run of a card's rows."""
        return self.prev_gpu is None or self.prev_gpu not in card.gpu_ids

    def needed_height(self) -> int:
        content = [card_rows_needed(c) for c in self.cards if self.shows_content(c)]
        return max([self.label_lines, *content]) + 1  # +1 gutter line between rows

    @property
    def label_lines(self) -> int:
        g = self.gpu
        return LABEL_LINES + (1 if g.external or not g.processes_visible else 0)

    def column_at(self, screen_x: int) -> int:
        return screen_x - self.region.x - LABEL_WIDTH

    def card_at(self, col: int | None) -> Card | None:
        """The card in column ``col``; in a column of running work, this GPU's."""
        here = [c for c in self.cards if c.col == col]
        return next((c for c in here if self.gpu.id in c.gpu_ids),
                    here[0] if len(here) == 1 else None)

    # rendering -------------------------------------------------------------
    def _label_line(self, y: int) -> list[Segment]:
        g = self.gpu
        cursor_row = self.state.cursor_gpu == g.id
        base = Style(bgcolor=theme.SELECTED_BG if cursor_row else None)
        dim = base + Style(color=theme.SUBTLE)
        text = base + Style(color=theme.MAIN_FG)
        lines = [
            [(f"{'▶' if cursor_row else ' '}", base + Style(color=theme.TITLE, bold=True)),
             (f"gpu{g.id} ", base + Style(color=theme.TITLE, bold=True)),
             *([(f"{g.host} ", text)] if g.host else []),
             (g.short_name, dim)],
            [(" mem ", dim),
             theme.meter(g.memory_fraction, METER_WIDTH, theme.TEMP, base),
             (f" {format_bytes_gib(g.memory_used)}/{format_bytes_gib(g.memory_total)}", text)],
            [(" util", dim), (" ", base),
             theme.meter(g.utilization / 100, METER_WIDTH, theme.CPU, base),
             (f" {g.utilization}%", text)],
        ]
        if g.external:
            known = [p.used_memory for p in g.external if p.used_memory is not None]
            mem = f" {sum(known) / 1024**3:.1f}G" if known else ""
            n = len(g.external)
            lines.append([(f" ! other: {n} proc{'s' if n > 1 else ''}{mem}",
                           base + Style(color=theme.BAD, bold=True))])
        elif not g.processes_visible:
            lines.append([(" ? procs unknown", base + Style(color=theme.WARN))])
        parts = lines[y] if y < len(lines) else []
        segs, used = [], 0
        for part in parts:
            if isinstance(part, list):  # a meter: already segments, one cell each
                part = part[: max(0, LABEL_WIDTH - 1 - used)]
                segs.extend(part)
                used += sum(len(s.text) for s in part)
                continue
            text, style = part
            text = text[: max(0, LABEL_WIDTH - 1 - used)]
            segs.append(Segment(text, style))
            used += len(text)
        segs.append(Segment(" " * (LABEL_WIDTH - 1 - used), base))
        segs.append(Segment("│", Style(color=GRID)))
        return segs

    def render_line(self, y: int) -> Strip:
        st = self.state
        height = self.size.height or self.needed_height()
        body = height - 1  # last line is the gutter between rows
        gutter = y == body
        width = max(0, self.size.width - LABEL_WIDTH)
        cells: list[tuple[str, Style]] = [(" ", Style(bgcolor=FUTURE_BG))] * width

        # past zone: to scale ---------------------------------------------------
        pw = min(st.past_width, width)
        mark_every = max(60, st.minutes_per_cell * 6)
        for x in range(pw):
            t = st.x_to_time(x)
            tick = gutter and (t.hour * 60 + t.minute) % mark_every < st.minutes_per_cell
            cells[x] = ("┊" if tick else " ", Style(bgcolor=PAST_BG, color=GRID))
        for b in self.past:
            x0, x1 = max(0, b.x0), min(pw, b.x1)
            if x1 <= x0 or gutter:
                continue
            style = Style(bgcolor=b.bg, color=b.fg)
            for x in range(x0, x1):
                cells[x] = (" ", style)
            lines = wrap_center(b.label, x1 - x0) if x1 - x0 >= 3 else []
            if len(lines) > body:  # never taller than the row: last line gets an ellipsis
                lines = lines[:body]
                lines[-1] = (lines[-1].rstrip()[:-1] + "…").center(x1 - x0)
            top = (body - len(lines)) // 2
            if top <= y < top + len(lines):
                for i, ch in enumerate(lines[y - top]):
                    cells[x0 + i] = (ch, style + Style(bold=True))

        # divider at now ------------------------------------------------------
        if st.divider_x < width:
            cells[st.divider_x] = ("┃", Style(color=NOW_COLOR, bold=True, bgcolor=FUTURE_BG))

        # future zone: fixed-width cards --------------------------------------
        w = st.card_width
        for card in self.cards:
            x = st.col_x(card.col)
            if x is None or x + w > width:
                continue
            if gutter and not (self.next_gpu is not None and self.next_gpu in card.gpu_ids):
                continue
            style = Style(bgcolor=card.bg, color=card.fg, bold=card.kind == "ghost",
                          dim=card.dimmed)
            for i in range(w):
                cells[x + i] = (" ", style)
            if card.selected:
                edge = Style(bgcolor=card.bg, color=theme.TITLE, bold=True)
                cells[x] = ("▌", edge)
                cells[x + w - 1] = ("▐", edge)
            if gutter or not self.shows_content(card):
                continue
            top = (body - len(card.lines)) // 2
            if top <= y < top + len(card.lines):
                line = card.lines[y - top]
                bold = y - top < card.title_lines
                for i, ch in enumerate(line[:w]):
                    if ch != " ":
                        cells[x + i] = (ch, style + Style(bold=bold or card.kind == "ghost"))

        if self.end_slot_col is not None and not gutter:
            x = st.col_x(self.end_slot_col)
            if x is not None and x + w <= width:
                style = Style(bgcolor="#141414", color=theme.SUBTLE)
                for i in range(w):
                    cells[x + i] = (" ", style)
                if y == body // 2:
                    for i, ch in enumerate("+ end of queue".center(w)):
                        cells[x + i] = (ch, style)

        segs = self._label_line(y)
        run_text, run_style = "", None
        for ch, style in cells:
            if style == run_style:
                run_text += ch
            else:
                if run_text:
                    segs.append(Segment(run_text, run_style))
                run_text, run_style = ch, style
        if run_text:
            segs.append(Segment(run_text, run_style))
        # cells without a background of their own take the widget's (not the terminal's)
        return Strip(segs).apply_style(self.rich_style)

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
        hit = self.state.hit(self.column_at(event.screen_x))
        if hit.zone == "future":
            self.post_message(self.Hover(self.gpu.id, hit))

    def on_mouse_down(self, event: events.MouseDown) -> None:
        if event.button != 1:
            return
        hit = self.state.hit(self.column_at(event.screen_x))
        card = self.card_at(hit.col) if hit.zone == "future" else None
        key = card.key if card else None
        if hit.zone == "past":
            block = next((b for b in self.past
                          if b.x0 <= self.column_at(event.screen_x) < b.x1), None)
            key = block.key if block else None
        self.post_message(self.BlockPressed(self.gpu.id, key, event.screen_x, event.screen_y))
        if card is not None and card.kind == "planned":
            self._press = (event.screen_x, event.screen_y, card.key)
            self._dragging = False
            self.capture_mouse()
        event.stop()

    def on_mouse_up(self, event: events.MouseUp) -> None:
        if self._press is None:
            return
        key = self._press[2]
        dragged = self._dragging
        self._press = None
        self._dragging = False
        self.capture_mouse(False)
        if dragged:
            self.post_message(self.BlockDragEnd(key, event.screen_x, event.screen_y))
        event.stop()
