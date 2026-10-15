from __future__ import annotations

from datetime import datetime, timedelta

from conftest import T0
from zgm.models import Assignment, AssignmentState
from zgm.widgets.timeline_state import (
    MIN_CARD_WIDTH, TimelineState, ceil_time, floor_time, wrap_center,
)


def state(width: int = 100, card_width: int = 24, **kw) -> TimelineState:
    return TimelineState(now=T0, width=width, card_width=card_width, **kw)


def test_zones_split_at_now():
    s = state()
    assert s.past_width == 25 and s.divider_x == 25 and s.future_x0 == 26
    assert s.past_end == T0
    assert s.visible_cols == 3  # (100 - 26 + 1) // 25


def test_past_zone_is_to_scale():
    s = state()  # 15 min/cell
    assert s.x_to_time(s.past_width) == T0
    assert s.time_to_x(T0 - timedelta(hours=1)) == s.past_width - 4
    assert s.span(T0 - timedelta(hours=1), T0) == (21, 25)
    assert s.span(T0, T0 + timedelta(hours=1)) is None  # future is not drawn to scale
    assert s.span(T0 - timedelta(days=3), T0 - timedelta(days=2)) is None
    s.pan_past(4)
    assert s.past_end == T0 - timedelta(hours=1)
    s.zoom(+1)
    assert s.minutes_per_cell == 30


def test_future_columns_have_fixed_width():
    s = state()
    assert [s.col_x(c) for c in range(4)] == [26, 51, 76, None]
    s.future_scroll = 1
    assert s.col_x(0) is None and s.col_x(1) == 26


def test_hit_testing():
    s = state()
    assert s.hit(-1).zone == "label"
    assert s.hit(3).zone == "past" and s.hit(3).time == s.x_to_time(3)
    assert s.hit(25).zone == "divider"
    h = s.hit(26)
    assert (h.zone, h.col, h.after) == ("future", 0, False)
    h = s.hit(26 + 20)
    assert (h.col, h.after) == (0, True)
    assert s.hit(51).col == 1


def test_cursor_scrolls_columns():
    s = state()
    s.move_cursor(5, n_items=5)
    assert s.cursor_col == 5 and s.future_scroll == 3
    s.move_cursor(-10, n_items=5)
    assert s.cursor_col == 0 and s.future_scroll == 0
    assert s.hidden_cols(5) == (0, 3)


def test_card_width_minimum():
    assert state(card_width=4).card_width == MIN_CARD_WIDTH


def test_wrap_center_keeps_full_text():
    name = "a-very-long-experiment-name-for-ablation-of-attention-heads"
    lines = wrap_center(name, 22)
    assert all(len(line) == 22 for line in lines)
    assert "".join(line.strip() for line in lines) == name
    assert wrap_center("bert", 10) == ["   bert   "]
    two = wrap_center("resnet50 baseline run", 10)
    assert [line.strip() for line in two] == ["resnet50", "baseline", "run"]


def test_snapping():
    t = datetime(2030, 1, 1, 9, 7, 30)
    assert floor_time(t, 15) == datetime(2030, 1, 1, 9, 0)
    assert ceil_time(t, 15) == datetime(2030, 1, 1, 9, 15)
    assert ceil_time(datetime(2030, 1, 1, 9, 15), 15) == datetime(2030, 1, 1, 9, 15)


def test_display_interval_planned_vs_actual():
    a = Assignment("t", (0,), T0, T0 + timedelta(hours=1))
    assert a.display_interval(T0) == (T0, T0 + timedelta(hours=1))
    assert not a.is_historical

    a.actual_start = T0 + timedelta(minutes=10)
    a.state = AssignmentState.RUNNING
    assert a.display_interval(T0 + timedelta(minutes=20)) == (
        T0 + timedelta(minutes=10), T0 + timedelta(minutes=70))
    late = T0 + timedelta(hours=3)
    assert a.display_interval(late) == (T0 + timedelta(minutes=10), late)

    a.actual_end = T0 + timedelta(minutes=40)
    a.state = AssignmentState.COMPLETED
    assert a.display_interval(late) == (T0 + timedelta(minutes=10), T0 + timedelta(minutes=40))
    assert a.is_historical


def test_actual_gpus_win_for_display():
    a = Assignment("t", (0,), T0, T0 + timedelta(hours=1))
    assert a.preferred_gpu_ids == (0,) and a.display_gpu_ids == (0,)
    a.actual_gpu_ids = (2,)
    assert a.display_gpu_ids == (2,) and a.preferred_gpu_ids == (0,)
