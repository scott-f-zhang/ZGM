"""Pure view-state for the GPU timeline.

Two views share one widget:

- **time**: the whole strip is a continuous time axis. History uses actual
  times, running and queued work use their estimate, all to scale. Queued work
  with no estimate (``start is None``) cannot be placed on the axis and goes to
  a fixed-width "start unknown" tail at the right.
- **tasks**: the whole strip is fixed-width cards, one per run in order:
  history, running, then the queue. A run of a few seconds is as readable as
  one of several hours.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from datetime import datetime, timedelta

ZOOM_LEVELS = [5, 15, 30, 60, 120, 360]  # minutes per cell in the time view
DEFAULT_CARD_WIDTH = 24
MIN_CARD_WIDTH = 12
TIME_VIEW = "time"
TASK_VIEW = "tasks"
NOW_FRACTION = 0.7  # both views: "now" sits this far from the strip's left edge
MAX_CARD_WIDTH = 48


def floor_time(t: datetime, minutes: int) -> datetime:
    t = t.replace(second=0, microsecond=0)
    total = t.hour * 60 + t.minute
    return t.replace(hour=0, minute=0) + timedelta(minutes=total - total % minutes)


def ceil_time(t: datetime, minutes: int) -> datetime:
    f = floor_time(t, minutes)
    return f if f == t else f + timedelta(minutes=minutes)


def wrap_center(text: str, width: int) -> list[str]:
    """Wrap ``text`` to ``width`` (breaking long words) and centre every line."""
    if width <= 0:
        return []
    lines = textwrap.wrap(text, width, break_long_words=True, break_on_hyphens=True) or [""]
    return [line.center(width) for line in lines]


@dataclass
class Hit:
    zone: str  # "past" | "divider" | "future" | "label"
    time: datetime | None = None  # time view: the time under the pointer
    x: int | None = None  # time view: the strip cell under the pointer
    col: int | None = None  # cards/tail: column index
    after: bool = False  # cards/tail: pointer is on the right half of the column


@dataclass
class TimelineState:
    now: datetime = field(default_factory=datetime.now)
    zoom_index: int = 1
    width: int = 80  # cells in the strip (time area + gutter + unknown tail)
    card_width: int = DEFAULT_CARD_WIDTH
    view: str = TIME_VIEW
    time_pan: int = 0  # cells the time window is panned back (negative: forward)
    future_scroll: int = 0  # first visible card column (tail or task view)
    cursor_gpu: int = 0
    cursor_col: int = 0  # task view / tail column (== n items: end of queue)
    cursor_x: int | None = None  # time view: the strip cell under the cursor
    cursor_zone: str = "time"  # "time" | "tail" | "cards"
    cursor_after: bool = False  # cards/tail: place after the item under the cursor
    cursor_at_now: bool = True  # time view: the cursor follows now until moved
    cursor_key: str | None = None  # time view: the run left/right selected (follows it)
    tail_width: int = 0  # cells reserved for unknown-start cards (time view)
    now_col: int = 0  # task view: first future column
    scroll_to_now: bool = False  # task view: bring the history/queue boundary into view

    def __post_init__(self) -> None:
        self.card_width = max(MIN_CARD_WIDTH, self.card_width)

    # geometry --------------------------------------------------------------
    @property
    def minutes_per_cell(self) -> int:
        return ZOOM_LEVELS[self.zoom_index]

    @property
    def cell(self) -> timedelta:
        return timedelta(minutes=self.minutes_per_cell)

    @property
    def is_tasks(self) -> bool:
        return self.view == TASK_VIEW

    @property
    def past_width(self) -> int:
        """Cells used by the time axis (time view) or by nothing (task view)."""
        if self.is_tasks:
            return 0
        return max(0, min(self.width, self.width - self.tail_width))

    @property
    def now_x(self) -> int:
        """Cell of "now" in the time area."""
        if self.is_tasks:
            return 0
        return self.now_anchor + self.time_pan

    @property
    def now_anchor(self) -> int:
        """Time view: where "now" sits unpanned, from the whole strip (not the axis
        left of the start-unknown tail, so a tail appearing doesn't move it)."""
        return max(1, int(self.width * NOW_FRACTION))

    @property
    def divider_x(self) -> int:
        if self.is_tasks:
            x = self.col_x(self.now_col)
            return -1 if x is None else x - 1
        return self.now_x

    @property
    def future_x0(self) -> int:
        return self.lead if self.is_tasks else self.past_width + 1

    @property
    def now_target(self) -> int:
        """Task view: x where the first future card starts once scrolled to now,
        so the divider (one cell before it) is at NOW_FRACTION of the strip."""
        return min(self.width - 1, int(self.width * NOW_FRACTION) + 1)

    @property
    def lead(self) -> int:
        """Task view: blank cells left of the first card column, so that card
        columns line up with ``now_target``."""
        return self.now_target % self.pitch

    @property
    def now_cols(self) -> int:
        """Task view: whole card columns left of ``now_target``."""
        return self.now_target // self.pitch

    @property
    def pitch(self) -> int:
        return self.card_width + 1  # card plus one-cell gutter

    @property
    def visible_cols(self) -> int:
        if not self.is_tasks and self.tail_width <= 0:
            return 0
        return max(1, (self.width - self.future_x0 + 1) // self.pitch)

    # time view: one continuous axis ----------------------------------------
    @property
    def past_start(self) -> datetime:
        if self.is_tasks:
            return self.now
        return self.now - self.cell * self.now_x

    @property
    def past_end(self) -> datetime:
        if self.is_tasks:
            return self.now
        return self.past_start + self.cell * self.past_width

    def time_to_x(self, t: datetime) -> float:
        return (t - self.past_start) / self.cell

    def time_cell(self, t: datetime) -> int:
        """The strip cell a time falls in (nearest cell, so a minute of rounding
        does not move the cursor into the previous one)."""
        return int(self.time_to_x(t) + 0.5)

    def x_to_time(self, x: int) -> datetime:
        return self.past_start + self.cell * int(x)

    @property
    def cursor_time(self) -> datetime:
        return self.x_to_time(self.cursor_x if self.cursor_x is not None else self.now_x)

    def span(self, start: datetime, end: datetime) -> tuple[int, int] | None:
        """Visible [x0, x1) of a time interval, or None (time view only)."""
        if self.is_tasks:
            return None
        x0, x1 = self.time_to_x(start), self.time_to_x(end)
        pw = self.past_width
        if x1 <= 0 or x0 >= pw:
            return None
        a = max(0, int(x0 // 1))
        b = min(pw, max(a + 1, int(-(-x1 // 1))))
        return a, b

    # fixed-width columns: the task view, and the time view's unknown tail ---
    def col_x(self, col: int) -> int | None:
        """Left x of card column ``col``, or None if scrolled out of view."""
        if not self.is_tasks and self.tail_width <= 0:
            return None
        rel = col - self.future_scroll
        if rel < 0 or rel >= self.visible_cols:
            return None
        x = self.future_x0 + rel * self.pitch
        return x if x + self.card_width <= self.width else None

    def hit(self, x: int) -> Hit:
        if x < 0:
            return Hit("label")
        if self.is_tasks:
            rel = x - self.future_x0
            if rel < 0:
                return Hit("future", x=x, col=None)
            col = self.future_scroll + rel // self.pitch
            if col < 0:
                return Hit("future", x=x, col=None)
            return Hit("future", x=x, col=col, after=(rel % self.pitch) * 2 >= self.pitch)
        if x < self.past_width:
            return Hit("past", time=self.x_to_time(x), x=x)
        if self.tail_width <= 0 or x < self.future_x0:
            return Hit("divider", x=x)
        rel = x - self.future_x0
        col = self.future_scroll + rel // self.pitch
        return Hit("future", x=x, col=col, after=(rel % self.pitch) * 2 >= self.pitch)

    # navigation ------------------------------------------------------------
    def move_cursor(self, delta: int, n_items: int) -> None:
        if not self.is_tasks:
            if self.cursor_zone == "time":
                self.cursor_at_now = False
                self.cursor_key = None
                x = (self.cursor_x if self.cursor_x is not None else self.now_x) + delta
                if x >= self.past_width and self.tail_width > 0:
                    self.cursor_zone, self.cursor_col = "tail", 0
                    self.ensure_cursor_visible()
                else:
                    self.cursor_x = max(0, min(max(0, self.past_width - 1), x))
                return
            col = self.cursor_col + delta
            if col < 0:
                self.cursor_zone = "time"
                self.cursor_x = max(0, self.past_width - 1)
                return
            self.cursor_col = min(n_items, col)
            self.ensure_cursor_visible()
            return
        self.cursor_col = max(0, min(n_items, self.cursor_col + delta))
        self.ensure_cursor_visible()

    def set_cursor_col(self, col: int, n_items: int) -> None:
        self.cursor_zone = "tail" if not self.is_tasks else "cards"
        self.cursor_col = max(0, min(n_items, col))
        self.ensure_cursor_visible()

    def set_cursor_x(self, x: int) -> None:
        self.cursor_zone = "time"
        self.cursor_at_now = False
        self.cursor_key = None
        self.cursor_x = max(0, min(max(0, self.past_width - 1), x))

    def scroll_to_now_col(self) -> None:
        """Task view: put the history/queue boundary at NOW_FRACTION; with little
        history the columns left of it stay empty (negative scroll)."""
        self.future_scroll = self.now_col - self.now_cols

    def select_run(self, key: str, start: datetime, end: datetime) -> None:
        """Time view: put the cursor on a run, panning so that it is in view."""
        self.cursor_zone = "time"
        self.cursor_at_now = False
        self.cursor_key = key
        x0, x1 = self.time_to_x(start), self.time_to_x(max(start, end))
        pw = self.past_width
        if x1 <= 0 or x0 >= pw:  # off screen: bring its start to the middle
            self.time_pan += int(round(pw / 2 - x0))
            x0 = self.time_to_x(start)
        self.cursor_x = max(0, min(max(0, pw - 1), int(x0)))

    def ensure_cursor_visible(self) -> None:
        if self.cursor_col < self.future_scroll:
            self.future_scroll = self.cursor_col
        elif self.cursor_col >= self.future_scroll + self.visible_cols:
            self.future_scroll = self.cursor_col - self.visible_cols + 1

    def hidden_cols(self, n_items: int) -> tuple[int, int]:
        """Card columns (incl. the end-of-queue slot) scrolled out left / right."""
        if self.visible_cols <= 0:
            return 0, 0
        total = n_items + 1
        return self.future_scroll, max(0, total - self.future_scroll - self.visible_cols)

    def scroll_future(self, delta: int, n_items: int) -> None:
        self.future_scroll += delta
        self.clamp_scroll(n_items)

    def clamp_scroll(self, n_items: int) -> None:
        """Keep the scroll inside the cards, also when the queue got shorter."""
        visible = self.visible_cols
        if visible <= 0:
            self.future_scroll = 0
            return
        low = min(0, self.now_col - self.now_cols) if self.is_tasks else 0
        self.future_scroll = max(low, min(max(low, n_items + 1 - visible),
                                          self.future_scroll))

    def pan_past(self, cells: int) -> None:
        """Positive = further back into history (time view)."""
        self.pan_time(cells)

    def pan_time(self, cells: int) -> None:
        self.time_pan += cells

    def pan(self, cells: int, n_items: int) -> None:
        if self.is_tasks:
            self.scroll_future(cells * max(1, self.visible_cols // 2), n_items)
        else:
            self.pan_time(cells * max(1, self.past_width // 3))

    def zoom(self, step: int) -> None:
        """Positive = zoom out. Time view: minutes per cell; task view: card width."""
        if self.is_tasks:
            self.card_width = max(MIN_CARD_WIDTH, min(MAX_CARD_WIDTH, self.card_width - 4 * step))
            self.scroll_to_now = True
            return
        self.zoom_index = max(0, min(len(ZOOM_LEVELS) - 1, self.zoom_index + step))

    def reset(self) -> None:
        """Back to now: the time window centred on now, or the task view's cards
        around the history/queue boundary."""
        if self.is_tasks:
            self.scroll_to_now = True
            self.cursor_zone = "cards"
            self.future_scroll = 0
            self.cursor_col = 0
            return
        self.time_pan = 0
        self.cursor_zone = "time"
        self.cursor_at_now = True
        self.cursor_key = None
        self.cursor_x = self.now_x
