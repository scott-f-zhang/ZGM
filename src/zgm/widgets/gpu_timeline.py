"""GPUTimeline: time axis plus one GPURow per GPU, with keyboard navigation."""

from __future__ import annotations

from datetime import datetime

from rich.segment import Segment
from rich.style import Style
from textual import events
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.message import Message
from textual.strip import Strip
from textual.widget import Widget

from ..models import Assignment, AssignmentState, GPUDevice, Task
from ..services.scheduler import Slot
from .gpu_row import LABEL_WIDTH, NOW_COLOR, Card, GPURow, PastBlock
from . import theme
from .palette import GHOST_BAD, GHOST_OK, HISTORY_COLOR, task_color
from .timeline_state import DEFAULT_CARD_WIDTH, Hit, TimelineState, wrap_center

FAILED_COLOR = theme.USED[0]  # dark red
STOPPED_COLOR = theme.AVAILABLE[0]  # dark amber


def pack_columns(gpu_sets: list[tuple[int, ...]]) -> list[int]:
    """A column for each item: the first one where it shares no GPU with an item
    already there. Running work never shares a GPU, so it all lines up in column 0."""
    used: list[set[int]] = []
    cols = []
    for gpus in gpu_sets:
        col = next((i for i, u in enumerate(used) if not u & set(gpus)), len(used))
        if col == len(used):
            used.append(set())
        used[col] |= set(gpus)
        cols.append(col)
    return cols


def slot_columns(slots: list[Slot]) -> tuple[list[int], int]:
    """The column of each slot, and how many columns running work takes: running
    work shares columns (see pack_columns); queued work follows, packed per GPU:
    each item takes the first column after the items queued before it on its
    GPUs, so a GPU's row has no gaps for other GPUs' work, and a multi-GPU item
    lines up across its rows."""
    running = [s.gpu_ids for s in slots if s.kind == "running"]
    run_cols = pack_columns(running)
    n_run = max(run_cols, default=-1) + 1
    cols, it, next_free = [], iter(run_cols), {}
    for s in slots:
        if s.kind == "running":
            cols.append(next(it))
        else:
            col = max(next_free.get(g, n_run) for g in s.gpu_ids)
            for g in s.gpu_ids:
                next_free[g] = col + 1
            cols.append(col)
    return cols, n_run


def queue_position(slots: list[Slot], gpu: int, col: int) -> tuple[int, int]:
    """Where something put on ``gpu`` at column ``col`` goes: (its index among the
    queued slots, the column it lands in). Before the first item queued on that
    GPU at or after ``col``; past the end of the row, after the last one there; on
    a GPU with nothing queued, at the end of the queue."""
    cols, n_run = slot_columns(slots)
    queued = [(s, c) for s, c in zip(slots, cols) if s.kind != "running"]
    here = [(i, c) for i, (s, c) in enumerate(queued) if gpu in s.gpu_ids]
    for i, c in here:
        if c >= col:
            return i, c
    if here:
        return here[-1][0] + 1, here[-1][1] + 1
    return len(queued), n_run


def card_lines(task: Task | None, slot: Slot, width: int, now: datetime,
               note: str | None = None) -> list[str]:
    """Full task name (wrapped, centred) + GPUs for multi-GPU work.

    The queue zone deliberately shows no times: they are only estimates.
    """
    inner = width - 2
    name = task.name if task else slot.task_id
    lines = wrap_center(name, inner)
    if slot.kind == "running":  # the marker goes here, so it never splits the name
        lines.append("▶ running".center(inner))
    if len(slot.gpu_ids) > 1:
        lines.append(f"GPU {','.join(map(str, slot.gpu_ids))}".center(inner))
    if note:
        lines.extend(wrap_center(note, inner))
    return [f" {line} " for line in lines]


class TimeAxis(Widget):
    """Two header lines: past-zone clock ticks | per-column start times."""

    def __init__(self, state: TimelineState, **kw) -> None:
        super().__init__(**kw)
        self.state = state
        self.columns: dict[int, Slot] = {}

    def render_line(self, y: int) -> Strip:
        st = self.state
        width = max(0, self.size.width - LABEL_WIDTH)
        dim = Style(color=theme.SUBTLE)
        if y == 0:
            head = ""
        else:
            left, right = st.hidden_cols(len(self.columns))
            head = " queue" + (f" ◀{left}" if left else "") + (f" {right}▶" if right else "")
        segs = [Segment(f"{head:<{LABEL_WIDTH - 1}.{LABEL_WIDTH - 1}}│", dim)]
        line = [" "] * width
        pw = min(st.past_width, width)
        step = next(m for m in (15, 30, 60, 120, 180, 360, 720, 1440, 2880)
                    if m / st.minutes_per_cell >= 7 or m == 2880)
        for x in range(pw):
            t = st.x_to_time(x)
            mins = t.hour * 60 + t.minute
            if mins % step < st.minutes_per_cell:
                if y == 0:
                    label = t.strftime("%m-%d") if mins < st.minutes_per_cell else t.strftime("%H:%M")
                    if x + len(label) <= pw:  # never cut a label at the divider
                        line[x: x + len(label)] = label
                else:
                    line[x] = "╵"
            elif y == 1:
                line[x] = "─"
        w = st.card_width
        # queued columns are numbered per GPU (1 = next to start on its GPUs);
        # running work is labelled as such
        n_run = min((c for c, s in self.columns.items() if s.kind != "running"), default=0)
        number = {col: col - n_run + 1 for col, s in self.columns.items() if s.kind != "running"}
        for col, slot in self.columns.items():
            x = st.col_x(col)
            if x is None or x + w > width:
                continue
            label = f"#{number[col]}" if col in number else "running"
            text = label.center(w) if y == 0 else "─" * w
            for i, ch in enumerate(text):
                line[x + i] = ch
        segs.append(Segment("".join(line[:st.divider_x]), Style(color=theme.SUBTLE)))
        if st.divider_x < width:
            now_label = "▼" if y == 1 else "┃"
            segs.append(Segment(now_label, Style(color=NOW_COLOR, bold=True)))
            segs.append(Segment("".join(line[st.divider_x + 1:]), Style(color=theme.SUBTLE)))
        # cells without a background of their own take the widget's (not the terminal's)
        return Strip(segs).apply_style(self.rich_style)


class GPUTimeline(Vertical, can_focus=True):
    BINDINGS = [
        Binding("left", "cursor(-1)", "earlier in the queue", show=False),
        Binding("right", "cursor(1)", "later in the queue", show=False),
        Binding("up,k", "gpu(-1)", "GPU above", show=False),
        Binding("down,j", "gpu(1)", "GPU below", show=False),
        Binding("h", "pan(1)", "history: back", show=False),
        Binding("l", "pan(-1)", "history: forward", show=False),
        Binding("plus,equals_sign,=", "zoom(-1)", "history: zoom in", show=False),
        Binding("minus,-", "zoom(1)", "history: zoom out", show=False),
        Binding("g", "now", "back to now", show=False),
    ]

    class CursorMoved(Message):
        pass

    def __init__(self, gpus: list[GPUDevice], card_width: int = DEFAULT_CARD_WIDTH,
                 **kw) -> None:
        super().__init__(**kw)
        self.gpus = gpus
        self.state = TimelineState(card_width=card_width)
        self.axis = TimeAxis(self.state, id="time-axis")
        self.rows = [GPURow(g, self.state) for g in gpus]
        self.slots: list[Slot] = []  # the future zone's items, in order
        self.cols: list[int] = []  # the column of each (running work shares columns)
        self.show_end_slot = False

    def compose(self):
        yield self.axis
        with VerticalScroll(id="gpu-rows"):
            yield from self.rows

    def on_resize(self, event: events.Resize) -> None:
        # rows sit in a scroll container: keep one cell for its scrollbar
        self.state.width = max(1, self.size.width - LABEL_WIDTH - 1)
        self.state.ensure_cursor_visible()
        self._changed()

    def set_gpus(self, gpus: list[GPUDevice]) -> None:
        """The GPUs changed (cluster mode: a host came or went): new rows."""
        self.gpus = gpus
        ids = [g.id for g in gpus]
        if ids and self.state.cursor_gpu not in ids:
            self.state.cursor_gpu = ids[0]
        self.call_later(self._swap_rows, [GPURow(g, self.state) for g in gpus])

    async def _swap_rows(self, rows: list[GPURow]) -> None:
        container = self.query_one("#gpu-rows")
        await container.remove_children()  # first: the new rows reuse the ids
        self.rows = rows
        await container.mount_all(rows)
        self._changed()

    def row_for_gpu(self, gpu_id: int) -> GPURow:
        return next(r for r in self.rows if r.gpu.id == gpu_id)

    @property
    def n_items(self) -> int:
        """Columns in the future zone (the cursor can also sit one past them)."""
        return max(self.cols, default=-1) + 1

    # data ------------------------------------------------------------------
    def update_data(
        self,
        slots: list[Slot],
        assignments: list[Assignment],
        tasks: dict[str, Task],
        now: datetime | None = None,
        moving: str | None = None,
        ghost_ok: bool = True,
        ghost_note: str | None = None,
        show_cursor: bool = False,
    ) -> None:
        now = now or datetime.now()
        st = self.state
        st.now = now
        self.slots = slots
        self.cols, _ = slot_columns(slots)
        st.cursor_col = min(st.cursor_col, self.n_items)
        st.clamp_scroll(self.n_items)
        w = st.card_width

        cards = []
        for col, slot in zip(self.cols, slots):
            task = tasks.get(slot.task_id)
            if slot.kind == "ghost":
                bg = GHOST_OK if ghost_ok else GHOST_BAD
                lines = card_lines(task, slot, w, now, None if ghost_ok else ghost_note)
            else:
                bg = task_color(slot.task_id)
                lines = card_lines(task, slot, w, now)
            name = task.name if task else slot.task_id
            title = len(wrap_center(("▶ " if slot.kind == "running" else "") + name, w - 2))
            cards.append(Card(col, slot.key, slot.task_id, slot.gpu_ids, lines, bg,
                              kind=slot.kind, selected=show_cursor and col == st.cursor_col,
                              dimmed=slot.key == moving, title_lines=title))

        past = []
        for a in assignments:
            if a.actual_start is None or a.state == AssignmentState.CANCELLED and not a.actual_end:
                continue
            start, end = a.display_interval(now)
            end = min(end, now)
            span = st.span(start, end)
            if span is None:
                continue
            task = tasks.get(a.task_id)
            if a.state == AssignmentState.RUNNING:
                bg, fg = task_color(a.task_id), theme.MAIN_BG
            elif a.state == AssignmentState.FAILED:
                bg, fg = FAILED_COLOR, "#fecaca"
            elif a.state == AssignmentState.CANCELLED:
                bg, fg = STOPPED_COLOR, "#fde68a"
            else:
                bg, fg = HISTORY_COLOR, theme.MAIN_FG
            past.append(PastBlock(a.id, span[0], span[1], task.name if task else a.task_id,
                                  bg, fg, a.display_gpu_ids))

        self.axis.columns = dict(zip(self.cols, slots))
        shown = (repr(self.axis.columns), st.zoom_index, st.past_offset, st.future_scroll,
                 st.width, int(now.timestamp() // 60))
        if shown != getattr(self.axis, "_shown", None):  # update_data runs on every tick
            self.axis._shown = shown
            self.axis.refresh()
        end_col = self.n_items if show_cursor and st.cursor_col == self.n_items else None
        ids = [r.gpu.id for r in self.rows]
        for i, row in enumerate(self.rows):
            row.set_data(
                cards,
                past,
                ids[i - 1] if i > 0 else None,
                ids[i + 1] if i + 1 < len(ids) else None,
                end_col if row.gpu.id == st.cursor_gpu else None,
            )

    def slot_under_cursor(self) -> Slot | None:
        """The slot in the cursor's column; in a column of running work, the one
        on the cursor's GPU."""
        st = self.state
        here = [s for s, c in zip(self.slots, self.cols) if c == st.cursor_col]
        return next((s for s in here if st.cursor_gpu in s.gpu_ids),
                    here[0] if len(here) == 1 else None)

    def locate(self, screen_x: int, screen_y: int) -> tuple[int, Hit] | None:
        """Screen coordinates -> (gpu id, hit in the strip), or None outside the rows."""
        for row in self.rows:
            r = row.region
            if r.height and r.y <= screen_y < r.y + r.height:
                return row.gpu.id, self.state.hit(row.column_at(screen_x))
        return None

    # actions ---------------------------------------------------------------
    def _changed(self) -> None:
        self.post_message(self.CursorMoved())

    def action_cursor(self, delta: int) -> None:
        self.state.move_cursor(delta, self.n_items)
        self._changed()

    def action_gpu(self, delta: int) -> None:
        if not self.gpus:
            return
        ids = [g.id for g in self.gpus]
        idx = ids.index(self.state.cursor_gpu) if self.state.cursor_gpu in ids else 0
        self.state.cursor_gpu = ids[max(0, min(len(ids) - 1, idx + delta))]
        self.call_after_refresh(self.query_one("#gpu-rows").scroll_to_widget,
                                self.row_for_gpu(self.state.cursor_gpu), animate=False)
        self._changed()

    def action_pan(self, cells: int) -> None:
        self.state.pan_past(cells * max(1, self.state.past_width // 3))
        self._changed()

    def action_zoom(self, step: int) -> None:
        self.state.zoom(step)
        self._changed()

    def action_now(self) -> None:
        self.state.reset()
        self._changed()

    def set_cursor(self, gpu_id: int, col: int | None = None) -> None:
        self.state.cursor_gpu = gpu_id
        if col is not None:
            self.state.set_cursor_col(col, self.n_items)
        self._changed()

    def on_mouse_scroll_down(self, event: events.MouseScrollDown) -> None:
        self._wheel(event, 1)

    def on_mouse_scroll_up(self, event: events.MouseScrollUp) -> None:
        self._wheel(event, -1)

    def _wheel(self, event, direction: int) -> None:
        loc = self.locate(event.screen_x, event.screen_y)
        if loc is None:
            return  # let the rows container scroll vertically
        _, hit = loc
        if hit.zone == "past":
            self.state.pan_past(-2 * direction)
        else:
            self.state.scroll_future(direction, self.n_items)
        self._changed()
        event.stop()

    def on_gpurow_hover(self, event: GPURow.Hover) -> None:
        if event.hit.col is None:
            return
        self.state.cursor_gpu = event.gpu_id
        self.state.cursor_col = max(0, min(self.n_items, event.hit.col))
        self._changed()
