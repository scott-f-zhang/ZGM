"""GPUTimeline: two views of the same work, with keyboard navigation.

- **time**: a continuous time axis. History uses actual times, running and
  queued work use their estimates; queued work with no estimate goes to a
  fixed-width "start unknown" tail at the right.
- **tasks**: fixed-width cards for the last 200 finished runs, then running and
  queued work, so a run of a few seconds is as readable as a long one.
"""

from __future__ import annotations

import math

from datetime import datetime, timedelta

from rich.segment import Segment
from rich.style import Style
from textual import events
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.message import Message
from textual.strip import Strip
from textual.widget import Widget

from ..models import Assignment, AssignmentState, GPUDevice, Task
from ..services.scheduler import GHOST, Slot
from .gpu_row import LABEL_WIDTH, NOW_COLOR, Card, GPURow, PastBlock
from . import theme
from .palette import GHOST_BAD, GHOST_OK, HISTORY_COLOR, task_color
from .timeline_state import (
    DEFAULT_CARD_WIDTH, TASK_VIEW, TIME_VIEW, Hit, TimelineState, wrap_center,
)

FAILED_COLOR = theme.USED[0]  # dark red
STOPPED_COLOR = theme.AVAILABLE[0]  # dark amber
HISTORY_LIMIT = 200  # finished runs shown in the task view


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


def history_columns(slots: list[Slot]) -> list[int]:
    """Columns of finished runs (oldest first) packed towards "now": each GPU's
    latest run sits in the column just before now, the one before it next to
    that, and so on, so every row's recent history is next to the divider. A
    multi-GPU run takes the same column on all its rows."""
    rev = sequential_columns(slots[::-1])
    depth = max(rev, default=-1) + 1
    return [depth - 1 - c for c in rev[::-1]]


def sequential_columns(slots: list[Slot]) -> list[int]:
    """One column after another per GPU: the fixed-width card order."""
    next_free: dict[int, int] = {}
    cols = []
    for s in slots:
        col = max((next_free.get(g, 0) for g in s.gpu_ids), default=0)
        for g in s.gpu_ids:
            next_free[g] = col + 1
        cols.append(col)
    return cols


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


def short_duration(d: timedelta) -> str:
    total = d.total_seconds()
    seconds = max(0, int(round(total)))
    if total > 0:
        seconds = max(1, seconds)  # a sub-second run is still "1s", not "0s"
    if seconds < 60:
        return f"{seconds}s"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m" if minutes else f"{hours}h"


def history_slots(assignments: list[Assignment]) -> list[Slot]:
    """Finished runs, oldest first, as timeline slots."""
    out = []
    for a in assignments:
        if a.actual_start is None or a.actual_end is None:
            continue
        if a.state not in (AssignmentState.COMPLETED, AssignmentState.FAILED,
                           AssignmentState.CANCELLED):
            continue
        out.append(Slot(a.id, a.task_id, a.display_gpu_ids, a.actual_start,
                        a.actual_end, "history", a.state))
    out.sort(key=lambda s: s.start)
    return out


def card_lines(task: Task | None, slot: Slot, width: int, now: datetime,
               note: str | None = None) -> list[str]:
    """Full task name (wrapped, centred), then its state/GPUs.

    History cards carry their duration and start time. Future cards carry no
    times: those are estimates, drawn by the time view.
    """
    inner = width - 2
    name = task.name if task else slot.task_id
    lines = wrap_center(name, inner)
    if slot.kind == "running":  # the marker goes here, so it never splits the name
        lines.append("▶ running".center(inner))
    elif slot.kind == "history":
        marks = {AssignmentState.COMPLETED: "✓", AssignmentState.FAILED: "✗",
                 AssignmentState.CANCELLED: "■"}
        mark = marks.get(slot.state, "·")
        dur = short_duration(slot.end - slot.start) if slot.start and slot.end else "?"
        when = slot.start.strftime("%m-%d %H:%M") if (
            slot.start and slot.start.date() != now.date()) else (
            slot.start.strftime("%H:%M") if slot.start else "?")
        lines.extend(wrap_center(f"{mark} {dur} · {when}", inner))
    if len(slot.gpu_ids) > 1:
        lines.append(f"GPU {','.join(map(str, slot.gpu_ids))}".center(inner))
    if note:
        lines.extend(wrap_center(note, inner))
    return [f" {line} " for line in lines]


class TimeAxis(Widget):
    """Two header lines: clock ticks (time view) and per-card labels."""

    def __init__(self, state: TimelineState, **kw) -> None:
        super().__init__(**kw)
        self.state = state
        self.columns: dict[int, Slot] = {}
        self.labels: dict[int, str] = {}
        self.n_cols = 0

    def render_line(self, y: int) -> Strip:
        st = self.state
        width = max(0, self.size.width - LABEL_WIDTH)
        dim = Style(color=theme.SUBTLE)
        if y == 0:
            head = ""
        else:
            left, right = st.hidden_cols(self.n_cols)
            head = (" " + st.view) + (f" ◀{left}" if left else "") + (
                f" {right}▶" if right else "")
        segs = [Segment(f"{head:<{LABEL_WIDTH - 1}.{LABEL_WIDTH - 1}}│", dim)]
        line = [" "] * width
        if st.view == TIME_VIEW:
            pw = min(st.past_width, width)
            step = next(m for m in (15, 30, 60, 120, 180, 360, 720, 1440, 2880)
                        if m / st.minutes_per_cell >= 7 or m == 2880)
            for x in range(pw):
                t = st.x_to_time(x)
                mins = t.hour * 60 + t.minute
                if mins % step < st.minutes_per_cell:
                    if y == 0:
                        label = (t.strftime("%m-%d") if mins < st.minutes_per_cell
                                 else t.strftime("%H:%M"))
                        crosses_now = x < st.divider_x < x + len(label)
                        if x + len(label) <= pw and not crosses_now:
                            line[x: x + len(label)] = label
                    else:
                        line[x] = "╵"
                elif y == 1:
                    line[x] = "─"
        w = st.card_width
        for col in self.columns:
            x = st.col_x(col)
            if x is None or x + w > width:
                continue
            label = self.labels.get(col, "?")
            text = label.center(w) if y == 0 else "─" * w
            for i, ch in enumerate(text):
                line[x + i] = ch
        dx = st.divider_x
        now_limit = min(st.past_width, width) if st.view == TIME_VIEW else width
        if 0 <= dx < now_limit:
            now_label = "▼" if y == 1 else "┃"
            segs.append(Segment("".join(line[:dx]), Style(color=theme.SUBTLE)))
            segs.append(Segment(now_label, Style(color=NOW_COLOR, bold=True)))
            segs.append(Segment("".join(line[dx + 1:]), Style(color=theme.SUBTLE)))
        else:
            segs.append(Segment("".join(line), Style(color=theme.SUBTLE)))
        # cells without a background of their own take the widget's (not the terminal's)
        return Strip(segs).apply_style(self.rich_style)


class GPUTimeline(Vertical, can_focus=True):
    BINDINGS = [
        Binding("left", "cursor(-1)", "previous run", show=False),
        Binding("right", "cursor(1)", "next run", show=False),
        Binding("up,k", "gpu(-1)", "GPU above", show=False),
        Binding("down,j", "gpu(1)", "GPU below", show=False),
        Binding("h", "pan(1)", "time: back / cards: left", show=False),
        Binding("l", "pan(-1)", "time: forward / cards: right", show=False),
        Binding("ctrl+plus,ctrl+equals_sign,plus,equals_sign", "zoom(-1)", "zoom in", show=False),
        Binding("ctrl+minus,ctrl+underscore,minus", "zoom(1)", "zoom out", show=False),
        Binding("g", "now", "back to now", show=False),
        Binding("t", "toggle_view", "time/task view"),
    ]

    class CursorMoved(Message):
        pass

    class ViewChanged(Message):
        def __init__(self, view: str) -> None:
            super().__init__()
            self.view = view

    def __init__(self, gpus: list[GPUDevice], card_width: int = DEFAULT_CARD_WIDTH,
                 **kw) -> None:
        super().__init__(**kw)
        self.gpus = gpus
        self.state = TimelineState(card_width=card_width)
        self.axis = TimeAxis(self.state, id="time-axis")
        self.rows = [GPURow(g, self.state) for g in gpus]
        self.slots: list[Slot] = []  # running + queued work
        self.items: list[Slot] = []  # history + slots, for the time cursor
        self.cols: list[int] = []  # card columns (all in tasks, tail in time)
        self.tail_slots: list[Slot] = []
        self.tail_cols: list[int] = []

    def compose(self):
        yield self.axis
        with VerticalScroll(id="gpu-rows"):
            yield from self.rows

    def on_resize(self, event: events.Resize) -> None:
        # rows sit in a scroll container: keep one cell for its scrollbar
        self.state.width = max(1, self.size.width - LABEL_WIDTH - 1)
        if not self.state.is_tasks and self.state.cursor_zone == "time":
            if self.state.cursor_at_now or self.state.cursor_x is None:
                self.state.cursor_x = max(0, min(max(0, self.state.past_width - 1),
                                                self.state.now_x))
            else:
                self.state.cursor_x = max(0, min(max(0, self.state.past_width - 1),
                                                self.state.cursor_x))
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
        """Card columns (the cursor can also sit one past them)."""
        return max(self.cols, default=-1) + 1

    def column_of(self, key: str) -> int | None:
        slots, cols = ((self.slots, self.cols) if self.state.is_tasks
                       else (self.tail_slots, self.tail_cols))
        return next((c for s, c in zip(slots, cols) if s.key == key), None)

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
        live = {g.id for g in self.gpus}
        hist = [s for s in history_slots(assignments) if set(s.gpu_ids) & live]
        future_cols, n_run = slot_columns(slots)
        w = st.card_width

        if st.is_tasks:
            hist = hist[-HISTORY_LIMIT:]
            hist_cols = history_columns(hist)
            now_col = max(hist_cols, default=-1) + 1
            cols = hist_cols + [now_col + c for c in future_cols]
            items = hist + slots
            self.slots = items
            self.items, self.cols, self.tail_slots, self.tail_cols = items, cols, [], []
            st.now_col = now_col
            st.cursor_zone = "cards"
            st.cursor_col = min(st.cursor_col, self.n_items)
            if st.scroll_to_now:
                st.scroll_to_now_col()
                st.cursor_col = min(now_col, max(0, self.n_items - 1))
                st.scroll_to_now = False
            st.clamp_scroll(self.n_items)

            cards = []
            for col, slot in zip(cols, items):
                task = tasks.get(slot.task_id)
                bg, fg = self._slot_colors(slot, task, ghost_ok)
                lines = card_lines(task, slot, w, now)
                name = task.name if task else slot.task_id
                title = len(wrap_center(("▶ " if slot.kind == "running" else "") + name,
                                        w - 2))
                cards.append(Card(col, slot.key, slot.task_id, slot.gpu_ids, lines, bg,
                                  fg=fg, kind=slot.kind,
                                  selected=(show_cursor and col == st.cursor_col
                                            and st.cursor_gpu in slot.gpu_ids),
                                  dimmed=slot.key == moving, title_lines=title))
            self.axis.columns = dict(zip(cols, items))
            self.axis.labels = self._task_labels(items, cols, slots, future_cols, n_run,
                                                 now_col, st.cursor_gpu, now)
            self.axis.n_cols = self.n_items
            past = []
            end_col = (self.n_items if show_cursor and st.cursor_col == self.n_items
                       else None)
        else:
            tail = [s for s in slots if s.start is None]
            tail_cols = sequential_columns(tail)
            n_tail = max(tail_cols, default=-1) + 1
            need = n_tail * st.pitch if tail else 0
            # the tail takes room right of now only, so now stays where it is, and
            # starts where the known work ends (now, or the last estimated end): a
            # tail at the far right would read as starting hours from now
            known = [math.ceil(st.time_to_x(s.end)) for s in [*hist, *slots]
                     if s.start is not None and s.end is not None]
            edge = max([st.now_x + 1, *known])
            room = max(0, st.width - st.now_anchor - 2)
            st.tail_width = min(room, max(need, st.width - edge - 1)) if tail else 0
            if st.width - st.tail_width < 8:
                st.tail_width = 0
            self.tail_slots, self.tail_cols = tail, tail_cols
            self.items = hist + slots
            self.cols = tail_cols
            st.cursor_col = min(st.cursor_col, self.n_items)
            picked = (next((s for s in self.items if s.key == st.cursor_key), None)
                      if st.cursor_key else None)
            if picked is None or picked.start is None or st.cursor_gpu not in picked.gpu_ids:
                st.cursor_key = None
            elif st.cursor_zone == "time":
                x0 = st.time_to_x(picked.start)
                if 0 <= x0 < st.past_width:
                    st.cursor_x = int(x0)
            if st.cursor_zone == "time":
                if st.cursor_at_now or st.cursor_x is None:
                    st.cursor_x = max(0, min(max(0, st.past_width - 1), st.now_x))
                else:
                    st.cursor_x = max(0, min(max(0, st.past_width - 1), st.cursor_x))
            st.clamp_scroll(self.n_items)

            blocks = []
            for slot in [*hist, *[s for s in slots if s.start is not None]]:
                start = slot.start
                end = slot.end or now
                if end < start:
                    end = start
                span = st.span(start, end)
                if span is None:
                    continue
                task = tasks.get(slot.task_id)
                bg, fg = self._slot_colors(slot, task, ghost_ok)
                name = task.name if task else slot.task_id
                label = ("▶ " if slot.kind == "running" else "") + name
                selected = (show_cursor and st.cursor_zone == "time"
                            and st.cursor_gpu in slot.gpu_ids
                            and (slot.key == st.cursor_key if st.cursor_key
                                 else span[0] <= st.cursor_x < span[1]))
                blocks.append(PastBlock(slot.key, span[0], span[1], label, bg, fg,
                                        slot.gpu_ids, selected, slot.kind))

            cards = []
            for col, slot in zip(tail_cols, tail):
                task = tasks.get(slot.task_id)
                bg, fg = self._slot_colors(slot, task, ghost_ok)
                note = None if slot.kind == "ghost" and not ghost_ok else "start unknown"
                lines = card_lines(task, slot, w, now,
                                   ghost_note if slot.kind == "ghost" and not ghost_ok
                                   else note)
                name = task.name if task else slot.task_id
                title = len(wrap_center(name, w - 2))
                cards.append(Card(col, slot.key, slot.task_id, slot.gpu_ids, lines, bg,
                                  fg=fg, kind=slot.kind,
                                  selected=(show_cursor and st.cursor_zone == "tail"
                                            and col == st.cursor_col
                                            and st.cursor_gpu in slot.gpu_ids),
                                  dimmed=slot.key == moving, title_lines=title))
            self.axis.columns = dict(zip(tail_cols, tail))
            self.axis.labels = {col: "?" for col in tail_cols}
            self.axis.n_cols = self.n_items
            past = blocks
            end_col = None

        shown = (st.view, repr(self.axis.columns), repr(self.axis.labels),
                 st.zoom_index, st.time_pan, st.future_scroll, st.width,
                 st.tail_width, st.now_col, self.axis.n_cols,
                 int(now.timestamp() // 60))
        if shown != getattr(self.axis, "_shown", None):
            self.axis._shown = shown
            self.axis.refresh()
        ids = [r.gpu.id for r in self.rows]
        for i, row in enumerate(self.rows):
            row.set_data(
                cards,
                past,
                ids[i - 1] if i > 0 else None,
                ids[i + 1] if i + 1 < len(ids) else None,
                end_col if row.gpu.id == st.cursor_gpu else None,
                show_cursor=show_cursor,
            )

    @staticmethod
    def _slot_colors(slot: Slot, task: Task | None, ghost_ok: bool) -> tuple[str, str]:
        if slot.kind == "ghost":
            return (GHOST_OK if ghost_ok else GHOST_BAD), theme.MAIN_BG
        if slot.kind == "history":
            if slot.state == AssignmentState.FAILED:
                return FAILED_COLOR, "#fecaca"
            if slot.state == AssignmentState.CANCELLED:
                return STOPPED_COLOR, "#fde68a"
            return HISTORY_COLOR, theme.MAIN_FG
        return task_color(slot.task_id), theme.MAIN_BG

    @staticmethod
    def _task_labels(items: list[Slot], cols: list[int], slots: list[Slot],
                     future_cols: list[int], n_run: int, now_col: int,
                     cursor_gpu: int, now: datetime) -> dict[int, str]:
        """Per-column axis text in the task view: start time for history, the
        queue number for queued work, "running" for running work."""
        by_key = {s.key: c for s, c in zip(slots, future_cols)}
        out: dict[int, str] = {}
        grouped: dict[int, list[Slot]] = {}
        for slot, col in zip(items, cols):
            grouped.setdefault(col, []).append(slot)
        for col, group in grouped.items():
            slot = next((s for s in group if cursor_gpu in s.gpu_ids), group[0])
            if slot.kind == "history":
                out[col] = (slot.start.strftime("%m-%d") if slot.start.date() != now.date()
                            else slot.start.strftime("%H:%M"))
            elif slot.kind == "running":
                out[col] = "running"
            else:
                future = by_key.get(slot.key)
                out[col] = f"#{future - n_run + 1}" if future is not None else "#?"
        return out

    def slot_under_cursor(self) -> Slot | None:
        """The slot under the cursor: a time block, a task card, or a tail card."""
        st = self.state
        if st.is_tasks:
            here = [s for s, c in zip(self.slots, self.cols) if c == st.cursor_col]
            return next((s for s in here if st.cursor_gpu in s.gpu_ids),
                        here[0] if len(here) == 1 else None)
        if st.cursor_zone == "tail":
            here = [s for s, c in zip(self.tail_slots, self.tail_cols)
                    if c == st.cursor_col]
            return next((s for s in here if st.cursor_gpu in s.gpu_ids),
                        here[0] if len(here) == 1 else None)
        if st.cursor_key:
            picked = next((s for s in self.items if s.key == st.cursor_key), None)
            if picked is not None:
                return picked
        x = st.cursor_x if st.cursor_x is not None else st.now_x
        here = []
        for probe in (x, x - 1):  # a block that just ended at now owns the cell left of it
            here = []
            for s in self.items:
                if s.start is None:
                    continue
                span = st.span(s.start, s.end or st.now)
                if span is not None and span[0] <= probe < span[1]:
                    here.append(s)
            if here:
                break
        here.sort(key=lambda s: s.kind != "history")  # running/queued last, so they win
        on_gpu = [s for s in here if st.cursor_gpu in s.gpu_ids]
        if on_gpu:
            return on_gpu[-1]
        return here[0] if len(here) == 1 else None

    def locate(self, screen_x: int, screen_y: int) -> tuple[int, Hit] | None:
        """Screen coordinates -> (gpu id, hit in the strip), or None outside the rows."""
        for row in self.rows:
            r = row.region
            if r.height and r.y <= screen_y < r.y + r.height:
                return row.gpu.id, self.state.hit(row.column_at(screen_x))
        return None

    # views -----------------------------------------------------------------
    def set_view(self, view: str) -> None:
        if view not in (TIME_VIEW, TASK_VIEW) or view == self.state.view:
            return
        self.state.view = view
        self.state.cursor_key = None
        if view == TASK_VIEW:
            self.state.cursor_zone = "cards"
            self.state.scroll_to_now = True
        else:
            self.state.cursor_zone = "time"
            self.state.cursor_at_now = True
            self.state.cursor_x = max(0, min(max(0, self.state.past_width - 1),
                                            self.state.now_x))
        self.post_message(self.ViewChanged(view))

    def action_toggle_view(self) -> None:
        self.set_view(TASK_VIEW if self.state.view == TIME_VIEW else TIME_VIEW)

    # actions ---------------------------------------------------------------
    def _changed(self) -> None:
        self.post_message(self.CursorMoved())

    def action_cursor(self, delta: int) -> None:
        st = self.state
        # time view: left/right go from run to run on the cursor's GPU (one cell at
        # a time would skip short runs or take many presses over long ones and gaps);
        # placing a task keeps cell steps, to pick the drop time
        if (not st.is_tasks and st.cursor_zone == "time"
                and getattr(self.screen, "placement", None) is None):
            if not self.step_run(delta) and delta > 0 and st.tail_width > 0:
                st.move_cursor(st.past_width, self.n_items)  # past the axis: the tail
            self._changed()
            return
        if st.is_tasks and getattr(self.screen, "placement", None) is None:
            # task view: card to card on this GPU's row; nothing further: stay
            cols = self.card_cols(st.cursor_gpu)
            nxt = ([c for c in cols if c > st.cursor_col] if delta > 0
                   else [c for c in cols if c < st.cursor_col])
            if nxt:
                st.cursor_col = nxt[0] if delta > 0 else nxt[-1]
                st.ensure_cursor_visible()
            self._changed()
            return
        st.move_cursor(delta, self.n_items)
        self._changed()

    def card_cols(self, gpu_id: int) -> list[int]:
        """Task view: the columns holding a card on ``gpu_id``'s row, in order."""
        return sorted({c for s, c in zip(self.slots, self.cols) if gpu_id in s.gpu_ids})

    def runs_on_cursor_gpu(self) -> list[Slot]:
        """Time view: the runs on the cursor's GPU that have a time, in time order."""
        st = self.state
        row = [s for s in self.items if s.start is not None and s.kind != "ghost"
               and st.cursor_gpu in s.gpu_ids]
        return sorted(row, key=lambda s: (s.start, s.end or st.now, s.key))

    def step_run(self, delta: int) -> bool:
        """Select the run ``delta`` runs away on the cursor's GPU; False if none."""
        st = self.state
        row = self.runs_on_cursor_gpu()
        keys = [s.key for s in row]
        cur = st.cursor_key if st.cursor_key in keys else None
        if cur is None:
            under = self.slot_under_cursor()
            cur = under.key if under is not None and under.key in keys else None
        if cur is not None:
            j = keys.index(cur) + delta
        else:  # between runs: the first one after / the last one before the cursor
            t = st.cursor_time
            after = [i for i, s in enumerate(row) if s.start > t]
            before = [i for i, s in enumerate(row) if s.start <= t]
            j = (after[0] if after else len(row)) if delta > 0 else (before[-1] if before else -1)
        if not 0 <= j < len(row):
            return False
        s = row[j]
        st.select_run(s.key, s.start, s.end or st.now)
        return True

    def action_gpu(self, delta: int) -> None:
        if not self.gpus:
            return
        st = self.state
        ids = [g.id for g in self.gpus]
        idx = ids.index(st.cursor_gpu) if st.cursor_gpu in ids else 0
        if st.is_tasks and getattr(self.screen, "placement", None) is None:
            # task view: the nearest row with a card that way, on the card closest
            # to the cursor's column; empty rows are skipped, none left: stay
            step = 1 if delta > 0 else -1
            for gpu in ids[idx + step::step] if step > 0 else ids[:idx][::-1]:
                cols = self.card_cols(gpu)
                if cols:
                    st.cursor_gpu = gpu
                    st.cursor_col = min(cols, key=lambda c: (abs(c - st.cursor_col), c))
                    st.ensure_cursor_visible()
                    break
            else:
                self._changed()
                return
        else:
            st.cursor_gpu = ids[max(0, min(len(ids) - 1, idx + delta))]
        self.call_after_refresh(self.query_one("#gpu-rows").scroll_to_widget,
                                self.row_for_gpu(self.state.cursor_gpu), animate=False)
        self._changed()

    def action_pan(self, cells: int) -> None:
        self.state.pan(cells, self.n_items)
        self._changed()

    def action_zoom(self, step: int) -> None:
        """Zoom around the selected run (it stays where it is on screen); with
        nothing selected, around now. Task view: around the selected card."""
        st = self.state
        if st.is_tasks:
            x = st.col_x(st.cursor_col) if self.slot_under_cursor() is not None else None
            st.zoom(step)
            if x is not None:  # keep the selected card where it was
                st.scroll_to_now = False
                st.future_scroll = st.cursor_col - max(0, x - st.lead) // st.pitch
            self._changed()
            return
        picked = None
        if st.cursor_zone == "time" and (st.cursor_key or not st.cursor_at_now):
            picked = self.slot_under_cursor()
        before = st.zoom_index
        if picked is not None and st.cursor_x is not None:
            x = st.cursor_x
            t = picked.start if st.cursor_key == picked.key and picked.start else st.x_to_time(x)
            if picked.start and st.time_to_x(picked.start) < 0:
                t = st.x_to_time(x)  # starts off the left edge: hold the cursor's time
            st.zoom(step)
            if st.zoom_index != before:
                # the smallest whole-cell pan that puts t in cell x again (rounding
                # could land it a hair left of x, and the cursor would creep left)
                st.time_pan = math.ceil((st.now - t) / st.cell + x - st.now_anchor - 1e-9)
        else:
            st.zoom(step)  # pan in cells is unchanged, so now stays put
        self._changed()
        self._changed()

    def action_now(self) -> None:
        self.state.reset()
        self._changed()

    def set_cursor(self, gpu_id: int, col: int | None = None,
                   x: int | None = None) -> None:
        self.state.cursor_gpu = gpu_id
        if x is not None:
            self.state.set_cursor_x(x)
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
        if not self.state.is_tasks and hit.zone == "past":
            self.state.pan_time(-2 * direction)
        else:
            self.state.scroll_future(direction, self.n_items)
        self._changed()
        event.stop()

    def on_gpurow_hover(self, event: GPURow.Hover) -> None:
        hit = event.hit
        if self.state.is_tasks:
            if hit.col is None:
                return
            self.state.cursor_gpu = event.gpu_id
            self.state.cursor_col = max(0, min(self.n_items, hit.col))
        elif hit.col is not None:
            self.state.cursor_gpu = event.gpu_id
            self.state.cursor_zone = "tail"
            self.state.cursor_col = max(0, min(self.n_items, hit.col))
        else:
            return
        self._changed()
