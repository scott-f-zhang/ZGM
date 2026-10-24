from __future__ import annotations

from datetime import datetime, timedelta

from conftest import T0
from zgm.models import Assignment, AssignmentState
from zgm.widgets.timeline_state import (
    MIN_CARD_WIDTH, TASK_VIEW, TIME_VIEW, TimelineState, ceil_time, floor_time,
    wrap_center,
)


def state(width: int = 100, card_width: int = 24, **kw) -> TimelineState:
    return TimelineState(now=T0, width=width, card_width=card_width, **kw)


def tail_state(width: int = 100, card_width: int = 24, **kw) -> TimelineState:
    """A time view with one unknown-start card column at the right."""
    s = state(width=width, card_width=card_width, **kw)
    s.tail_width = card_width + 1
    return s


def test_time_view_is_one_axis():
    s = state()
    assert s.view == TIME_VIEW
    assert s.past_width == 100 and s.future_x0 == 101
    assert s.now_x == 33 and s.divider_x == 33
    assert s.x_to_time(s.now_x) == T0
    assert s.time_to_x(T0) == s.now_x
    assert s.span(T0 - timedelta(hours=1), T0) == (29, 33)
    assert s.span(T0, T0 + timedelta(hours=1)) == (33, 37)  # future is to scale too
    assert s.span(T0 - timedelta(days=3), T0 - timedelta(days=2)) is None


def test_time_view_pans_and_zooms_around_now():
    s = state()
    s.pan_past(4)
    assert s.now_x == 37 and s.x_to_time(37) == T0
    s.zoom(+1)
    assert s.minutes_per_cell == 30
    assert s.x_to_time(s.now_x) == T0
    s.reset()
    assert s.now_x == 33 and s.cursor_x == 33


def test_unknown_tail_is_a_fixed_width_column():
    s = tail_state()
    assert s.past_width == 75 and s.future_x0 == 76
    assert s.visible_cols == 1
    assert s.col_x(0) == 76
    s.future_scroll = 1
    assert s.col_x(0) is None and s.col_x(1) == 76
    assert s.hit(76).zone == "future" and s.hit(76).col == 1
    assert s.hit(89).after is True


def test_time_view_hit_testing():
    s = state()
    assert s.hit(-1).zone == "label"
    hit = s.hit(3)
    assert hit.zone == "past" and hit.time == s.x_to_time(3) and hit.x == 3
    assert s.hit(s.now_x).zone == "past" and s.hit(s.now_x).time == T0
    assert s.hit(99).zone == "past"
    s = tail_state()
    assert s.hit(s.past_width).zone == "divider"
    assert s.hit(s.past_width + 1).zone == "future"


def test_task_view_is_all_fixed_width_cards():
    s = state(view=TASK_VIEW, now_col=3)
    assert s.past_width == 0 and s.future_x0 == 0
    assert [s.col_x(c) for c in range(4)] == [0, 25, 50, 75]
    assert s.col_x(4) is None
    s.future_scroll = 1
    assert s.col_x(0) is None and s.col_x(1) == 0
    assert s.hit(0).zone == "future" and s.hit(0).col == 1
    assert s.hit(20).after is True
    s.now_col = 1
    assert s.divider_x == s.col_x(1) - 1


def test_cursor_moves_between_time_and_tail():
    s = tail_state()
    s.set_cursor_x(74)
    s.move_cursor(1, n_items=1)
    assert s.cursor_zone == "tail" and s.cursor_col == 0
    s.move_cursor(-1, n_items=1)
    assert s.cursor_zone == "time" and s.cursor_x == 74


def test_cursor_scrolls_card_columns():
    s = state(view=TASK_VIEW)
    s.move_cursor(5, n_items=5)
    assert s.cursor_col == 5 and s.future_scroll == 2
    s.move_cursor(-10, n_items=5)
    assert s.cursor_col == 0 and s.future_scroll == 0
    assert s.hidden_cols(5) == (0, 2)


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
