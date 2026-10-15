from __future__ import annotations

from datetime import datetime, timedelta

from conftest import T0
from gpu_scheduler.models import Assignment, AssignmentState
from gpu_scheduler.widgets.timeline_state import TimelineState, ceil_time, floor_time


def state(**kw) -> TimelineState:
    s = TimelineState(view_start=T0, cursor_time=T0, **kw)
    s.width = 40
    return s


def test_mapping_round_trip():
    s = state()  # 15 min/cell
    assert s.minutes_per_cell == 15
    assert s.time_to_x(T0 + timedelta(hours=1)) == 4
    assert s.x_to_time(4) == T0 + timedelta(hours=1)
    assert s.view_end == T0 + timedelta(minutes=15 * 40)


def test_span_clips_to_view():
    s = state()
    assert s.span(T0, T0 + timedelta(hours=1)) == (0, 4)
    assert s.span(T0 - timedelta(hours=1), T0 + timedelta(minutes=30)) == (0, 2)
    assert s.span(T0 + timedelta(minutes=5), T0 + timedelta(minutes=10)) == (0, 1)  # min width 1
    assert s.span(T0 - timedelta(hours=2), T0 - timedelta(hours=1)) is None
    assert s.span(s.view_end, s.view_end + timedelta(hours=1)) is None
    assert s.span(T0, T0 + timedelta(days=2)) == (0, 40)


def test_snapping():
    t = datetime(2030, 1, 1, 9, 7, 30)
    assert floor_time(t, 15) == datetime(2030, 1, 1, 9, 0)
    assert ceil_time(t, 15) == datetime(2030, 1, 1, 9, 15)
    assert ceil_time(datetime(2030, 1, 1, 9, 15), 15) == datetime(2030, 1, 1, 9, 15)
    s = state()
    s.set_cursor_time(t)
    assert s.cursor_time == datetime(2030, 1, 1, 9, 0)
    s.set_cursor_time(t, round_up=True)
    assert s.cursor_time == datetime(2030, 1, 1, 9, 15)


def test_cursor_auto_pans():
    s = state()
    s.move_cursor(45)
    assert 0 <= s.cursor_x < s.width
    assert s.view_start > T0
    s.move_cursor(-100)
    assert 0 <= s.cursor_x < s.width


def test_zoom_keeps_anchor_in_view():
    s = state()
    s.move_cursor(10)
    s.zoom(+1)  # zoom out
    assert s.minutes_per_cell == 30
    assert 0 <= s.cursor_x < s.width
    s.zoom(-5)  # clamps at the finest level
    assert s.minutes_per_cell == 5
    s.zoom(+99)
    assert s.minutes_per_cell == 360


def test_pan():
    s = state()
    s.pan(-4)
    assert s.view_start == T0 - timedelta(hours=1)


def test_display_interval_planned_vs_actual():
    a = Assignment("t", (0,), T0, T0 + timedelta(hours=1))
    assert a.display_interval(T0) == (T0, T0 + timedelta(hours=1))
    assert not a.is_historical

    # running: starts at actual start, extends to "now" if overrunning
    a.actual_start = T0 + timedelta(minutes=10)
    a.state = AssignmentState.RUNNING
    assert a.display_interval(T0 + timedelta(minutes=20)) == (
        T0 + timedelta(minutes=10), T0 + timedelta(minutes=70))
    late = T0 + timedelta(hours=3)
    assert a.display_interval(late) == (T0 + timedelta(minutes=10), late)

    # completed: real times only, the estimate is ignored
    a.actual_end = T0 + timedelta(minutes=40)
    a.state = AssignmentState.COMPLETED
    assert a.display_interval(late) == (T0 + timedelta(minutes=10), T0 + timedelta(minutes=40))
    assert a.is_historical


def test_actual_gpus_win_for_display():
    a = Assignment("t", (0,), T0, T0 + timedelta(hours=1))
    assert a.preferred_gpu_ids == (0,) and a.display_gpu_ids == (0,)
    a.actual_gpu_ids = (2,)
    assert a.display_gpu_ids == (2,) and a.preferred_gpu_ids == (0,)
