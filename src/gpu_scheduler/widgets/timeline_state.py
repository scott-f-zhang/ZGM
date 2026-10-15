"""Pure view-state for the GPU timeline.

The strip is split at "now":

- the **past zone** (left) is drawn to scale: work there really ran, so its
  blocks use actual start/end times (``minutes_per_cell`` sets the scale);
- the **future zone** (right) is a row of fixed-width columns, one per queued
  assignment, in queue order. Its times are estimates, so they are printed in
  the cards rather than implied by width.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from datetime import datetime, timedelta

ZOOM_LEVELS = [5, 15, 30, 60, 120, 360]  # minutes per cell in the past zone
DEFAULT_CARD_WIDTH = 24
MIN_CARD_WIDTH = 12
PAST_FRACTION = 0.25
MIN_PAST_WIDTH = 8


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
    time: datetime | None = None  # past zone: start of the cell
    col: int | None = None  # future zone: column index
    after: bool = False  # future zone: pointer is on the right half of the column


@dataclass
class TimelineState:
    now: datetime = field(default_factory=datetime.now)
    zoom_index: int = 1
    width: int = 80  # cells in the strip (past + divider + future)
    card_width: int = DEFAULT_CARD_WIDTH
    past_offset: int = 0  # how many cells the past zone is panned back from now
    future_scroll: int = 0  # first visible future column
    cursor_gpu: int = 0
    cursor_col: int = 0  # future column index (== n items means "end of queue")

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
    def past_width(self) -> int:
        pw = max(MIN_PAST_WIDTH, int(self.width * PAST_FRACTION))
        return max(0, min(pw, self.width - self.card_width - 1))

    @property
    def divider_x(self) -> int:
        return self.past_width

    @property
    def future_x0(self) -> int:
        return self.past_width + 1

    @property
    def pitch(self) -> int:
        return self.card_width + 1  # card plus one-cell gutter

    @property
    def visible_cols(self) -> int:
        return max(1, (self.width - self.future_x0 + 1) // self.pitch)

    # past zone (to scale) --------------------------------------------------
    @property
    def past_end(self) -> datetime:
        return self.now - self.cell * self.past_offset

    @property
    def past_start(self) -> datetime:
        return self.past_end - self.cell * self.past_width

    def time_to_x(self, t: datetime) -> float:
        return (t - self.past_start) / self.cell

    def x_to_time(self, x: int) -> datetime:
        return self.past_start + self.cell * int(x)

    def span(self, start: datetime, end: datetime) -> tuple[int, int] | None:
        """Visible [x0, x1) of an interval inside the past zone, or None."""
        x0, x1 = self.time_to_x(start), self.time_to_x(end)
        if x1 <= 0 or x0 >= self.past_width:
            return None
        a = max(0, int(x0 // 1))
        b = min(self.past_width, max(a + 1, int(-(-x1 // 1))))
        return a, b

    # future zone (fixed-width columns) -------------------------------------
    def col_x(self, col: int) -> int | None:
        """Left x of future column ``col``, or None if scrolled out of view."""
        rel = col - self.future_scroll
        if rel < 0 or rel >= self.visible_cols:
            return None
        x = self.future_x0 + rel * self.pitch
        return x if x + self.card_width <= self.width else None

    def hit(self, x: int) -> Hit:
        if x < 0:
            return Hit("label")
        if x < self.past_width:
            return Hit("past", time=self.x_to_time(x))
        if x == self.divider_x:
            return Hit("divider")
        rel = x - self.future_x0
        col = self.future_scroll + rel // self.pitch
        return Hit("future", col=col, after=(rel % self.pitch) * 2 >= self.pitch)

    # navigation ------------------------------------------------------------
    def move_cursor(self, delta: int, n_items: int) -> None:
        self.cursor_col = max(0, min(n_items, self.cursor_col + delta))
        self.ensure_cursor_visible()

    def set_cursor_col(self, col: int, n_items: int) -> None:
        self.cursor_col = max(0, min(n_items, col))
        self.ensure_cursor_visible()

    def ensure_cursor_visible(self) -> None:
        if self.cursor_col < self.future_scroll:
            self.future_scroll = self.cursor_col
        elif self.cursor_col >= self.future_scroll + self.visible_cols:
            self.future_scroll = self.cursor_col - self.visible_cols + 1

    def hidden_cols(self, n_items: int) -> tuple[int, int]:
        """Columns (incl. the end-of-queue slot) scrolled out left / right."""
        total = n_items + 1
        return self.future_scroll, max(0, total - self.future_scroll - self.visible_cols)

    def scroll_future(self, delta: int, n_items: int) -> None:
        self.future_scroll = max(0, min(max(0, n_items + 1 - self.visible_cols),
                                        self.future_scroll + delta))

    def pan_past(self, cells: int) -> None:
        """Positive = further back into history."""
        self.past_offset = max(0, self.past_offset + cells)

    def zoom(self, step: int) -> None:
        self.zoom_index = max(0, min(len(ZOOM_LEVELS) - 1, self.zoom_index + step))

    def reset(self) -> None:
        self.past_offset = 0
        self.future_scroll = 0
        self.cursor_col = 0
