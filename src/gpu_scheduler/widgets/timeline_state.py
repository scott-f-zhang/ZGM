"""Pure view-state for the GPU timeline: time <-> column mapping, pan, zoom."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

ZOOM_LEVELS = [5, 15, 30, 60, 120, 360]  # minutes per terminal cell


def floor_time(t: datetime, minutes: int) -> datetime:
    t = t.replace(second=0, microsecond=0)
    total = t.hour * 60 + t.minute
    floored = total - total % minutes
    return t.replace(hour=0, minute=0) + timedelta(minutes=floored)


def ceil_time(t: datetime, minutes: int) -> datetime:
    f = floor_time(t, minutes)
    return f if f == t else f + timedelta(minutes=minutes)


@dataclass
class TimelineState:
    view_start: datetime = field(default_factory=datetime.now)
    zoom_index: int = 1
    width: int = 80  # cells in the time strip
    cursor_gpu: int = 0
    cursor_time: datetime = field(default_factory=datetime.now)

    def __post_init__(self) -> None:
        self.view_start = floor_time(self.view_start, self.minutes_per_cell)
        self.cursor_time = floor_time(self.cursor_time, self.minutes_per_cell)

    @property
    def minutes_per_cell(self) -> int:
        return ZOOM_LEVELS[self.zoom_index]

    @property
    def cell(self) -> timedelta:
        return timedelta(minutes=self.minutes_per_cell)

    @property
    def view_end(self) -> datetime:
        return self.view_start + self.cell * self.width

    # mapping ---------------------------------------------------------------
    def time_to_x(self, t: datetime) -> float:
        return (t - self.view_start) / self.cell

    def x_to_time(self, x: int) -> datetime:
        """Start time of column ``x`` (snapped to the cell grid)."""
        return self.view_start + self.cell * int(x)

    def span(self, start: datetime, end: datetime) -> tuple[int, int] | None:
        """Visible [x0, x1) columns covered by an interval, or None if off-screen."""
        x0 = self.time_to_x(start)
        x1 = self.time_to_x(end)
        if x1 <= 0 or x0 >= self.width:
            return None
        a = max(0, int(x0 // 1))
        b = min(self.width, max(a + 1, int(-(-x1 // 1))))
        return a, b

    @property
    def cursor_x(self) -> int:
        return int(self.time_to_x(self.cursor_time) // 1)

    # navigation ------------------------------------------------------------
    def pan(self, cells: int) -> None:
        self.view_start += self.cell * cells

    def move_cursor(self, cells: int) -> None:
        self.cursor_time += self.cell * cells
        self.ensure_cursor_visible()

    def set_cursor_time(self, t: datetime, round_up: bool = False) -> None:
        snap = ceil_time if round_up else floor_time
        self.cursor_time = snap(t, self.minutes_per_cell)
        self.ensure_cursor_visible()

    def ensure_cursor_visible(self, margin: int = 2) -> None:
        x = self.cursor_x
        m = min(margin, max(0, self.width // 4))
        if x < m:
            self.pan(x - m)
        elif x >= self.width - m:
            self.pan(x - (self.width - m) + 1)

    def zoom(self, step: int, anchor: datetime | None = None) -> None:
        idx = max(0, min(len(ZOOM_LEVELS) - 1, self.zoom_index + step))
        if idx == self.zoom_index:
            return
        anchor = anchor or self.cursor_time
        rel = self.time_to_x(anchor) / max(1, self.width)
        self.zoom_index = idx
        self.cursor_time = floor_time(self.cursor_time, self.minutes_per_cell)
        span = self.cell * self.width
        self.view_start = floor_time(anchor - span * rel, self.minutes_per_cell)
        self.ensure_cursor_visible()

    def jump_to(self, t: datetime, lead_fraction: float = 0.2) -> None:
        self.view_start = floor_time(
            t - self.cell * int(self.width * lead_fraction), self.minutes_per_cell
        )
        self.set_cursor_time(t)
