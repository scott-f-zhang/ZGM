from __future__ import annotations

import shutil
from datetime import timedelta
from pathlib import Path

import pytest

from gpu_scheduler.app import GPUSchedulerApp
from gpu_scheduler.models import Task, TaskStatus
from gpu_scheduler.screens.config import ConfigScreen
from gpu_scheduler.screens.schedule import ScheduleScreen
from gpu_scheduler.widgets.gpu_row import LABEL_WIDTH, GPURow
from gpu_scheduler.widgets.gpu_timeline import GPUTimeline
from gpu_scheduler.widgets.task_card import TaskCard

EXAMPLES = Path(__file__).parent.parent / "examples" / "configs"
SIZE = (140, 42)


@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    d = tmp_path / "configs"
    shutil.copytree(EXAMPLES, d)
    return d


def make_app(config_dir: Path, tasks: list[Task] | None = None) -> GPUSchedulerApp:
    app = GPUSchedulerApp(mock_gpus=4, config_dir=config_dir, stats_interval=0)
    for t in tasks or []:
        app.store.add_task(t)
    return app


def task(name: str, gpus: int = 1, hours: float = 1) -> Task:
    return Task(name, f"echo {name}", gpus, timedelta(hours=hours))


async def test_boots_with_one_row_per_gpu(config_dir):
    app = make_app(config_dir, [task("a"), task("b")])
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause()
        assert isinstance(app.screen, ScheduleScreen)
        assert len(app.screen.query(GPURow)) == 4
        assert len(app.screen.query(TaskCard)) == 2


async def test_keyboard_assign(config_dir):
    a = task("alpha")
    app = make_app(config_dir, [a])
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause()
        await pilot.press("a")  # placement mode on the selected card
        assert app.screen.placement is not None
        await pilot.press("down", "right", "right", "enter")
        await pilot.pause()
        assert app.screen.placement is None
        (asg,) = app.store.assignments.values()
        assert asg.task_id == a.id and asg.gpu_ids == (1,)
        assert app.store.tasks[a.id].status == TaskStatus.PLANNED


async def test_conflicting_placement_is_refused(config_dir):
    a, b = task("alpha", hours=3), task("beta")
    app = make_app(config_dir, [a, b])
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause()
        screen: ScheduleScreen = app.screen
        tl = screen.query_one(GPUTimeline)
        start = tl.state.cursor_time + tl.state.cell * 4
        assert app.assignment_service.assign(a.id, [0], start).ok
        await pilot.pause()
        screen.queue.select(b.id)
        await pilot.press("a")
        # force the ghost on top of alpha
        tl.set_cursor(0, start + tl.state.cell)
        await pilot.pause()
        assert not screen.current_ghost().ok
        await pilot.press("enter")
        await pilot.pause()
        assert len(app.store.assignments) == 1
        assert screen.placement is not None  # still placing; user can move on
        assert "conflicts with alpha" in screen.last_error
        await pilot.press("escape")
        assert screen.placement is None


async def test_keyboard_auto_slot_avoids_existing_block(config_dir):
    a, b = task("alpha", hours=2), task("beta")
    app = make_app(config_dir, [a, b])
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause()
        screen: ScheduleScreen = app.screen
        tl = screen.query_one(GPUTimeline)
        assert app.assignment_service.assign(a.id, [0], tl.state.cursor_time).ok
        await pilot.pause()
        screen.queue.select(b.id)
        await pilot.press("a")
        assert screen.current_ghost().ok  # default start is the next free slot
        await pilot.press("enter")
        await pilot.pause()
        assert len(app.store.assignments) == 2


async def test_mouse_drag_card_onto_timeline(config_dir):
    a = task("alpha")
    app = make_app(config_dir, [a])
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause()
        screen: ScheduleScreen = app.screen
        card = screen.query_one(TaskCard)
        row = screen.query_one("#gpu-row-2", GPURow)
        tl = screen.query_one(GPUTimeline)
        col = 30
        await pilot.mouse_down(card, offset=(3, 1))
        await pilot.hover(card, offset=(6, 2))  # start dragging
        await pilot.hover(row, offset=(LABEL_WIDTH + col, 0))
        assert screen.placement is not None and screen.placement.via_mouse
        await pilot.mouse_up(row, offset=(LABEL_WIDTH + col, 0))
        await pilot.pause()
        (asg,) = app.store.assignments.values()
        assert asg.gpu_ids == (2,)
        assert asg.planned_start == tl.state.x_to_time(col)
        assert screen.placement is None


async def test_mouse_drag_existing_block(config_dir):
    a = task("alpha")
    app = make_app(config_dir, [a])
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause()
        screen: ScheduleScreen = app.screen
        tl = screen.query_one(GPUTimeline)
        start = tl.state.x_to_time(10)
        asg = app.assignment_service.assign(a.id, [0], start).assignment
        await pilot.pause()
        row0 = screen.query_one("#gpu-row-0", GPURow)
        row3 = screen.query_one("#gpu-row-3", GPURow)
        await pilot.mouse_down(row0, offset=(LABEL_WIDTH + 11, 0))
        await pilot.hover(row3, offset=(LABEL_WIDTH + 20, 0))
        await pilot.mouse_up(row3, offset=(LABEL_WIDTH + 20, 0))
        await pilot.pause()
        moved = app.store.assignments[asg.id]
        assert moved.gpu_ids == (3,)
        assert moved.planned_start == tl.state.x_to_time(20)
        assert len(app.store.assignments) == 1


async def test_unassign_from_timeline(config_dir):
    a = task("alpha")
    app = make_app(config_dir, [a])
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause()
        tl = app.screen.query_one(GPUTimeline)
        app.assignment_service.assign(a.id, [0], tl.state.cursor_time)
        await pilot.press("tab")  # focus timeline; cursor sits on the block
        await pilot.press("u")
        await pilot.pause()
        assert not app.store.assignments
        assert app.store.tasks[a.id].status == TaskStatus.PENDING


async def test_new_task_dialog(config_dir):
    app = make_app(config_dir)
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause()
        await pilot.press("n")
        await pilot.press(*"sweep", "enter")
        await pilot.press("enter")  # keep default command
        await pilot.press("backspace", "2", "enter")
        await pilot.press(*["backspace"] * 2, *"90m", "enter")
        await pilot.pause()
        (t,) = app.store.tasks.values()
        assert (t.name, t.required_gpus, t.estimated_duration) == (
            "sweep", 2, timedelta(minutes=90))
        assert len(app.screen.query(TaskCard)) == 1


async def test_view_switching_keeps_state(config_dir):
    app = make_app(config_dir, [task("a"), task("b")])
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause()
        await pilot.press("right")
        selected = app.screen.queue.selected_id
        await pilot.press("2")
        await pilot.pause()
        assert isinstance(app.screen, ConfigScreen)
        await pilot.press("1")
        await pilot.pause()
        assert isinstance(app.screen, ScheduleScreen)
        assert app.screen.queue.selected_id == selected


async def test_config_edit_derive_and_task(config_dir):
    app = make_app(config_dir)
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("2")
        await pilot.pause()
        screen: ConfigScreen = app.screen
        repo = app.config_repo
        screen.current = repo.get("resnet50_base").path
        screen.refresh_params()

        # derive a child with one override
        await pilot.press("D")
        await pilot.pause()
        await pilot.press(*["backspace"] * 40, *"rn50_lr05", "enter")
        await pilot.press(*"optimizer.lr=0.05", "enter")
        await pilot.pause()
        child_file = config_dir / "rn50_lr05.yaml"
        assert child_file.exists()
        assert screen.current == child_file.resolve()

        # edit an inherited key in the child -> override written to the child
        table = screen.query_one("#config-table")
        table.focus()
        table.move_cursor(row=table.get_row_index("data.batch_size"))
        await pilot.press("e")
        await pilot.pause()
        await pilot.press(*["backspace"] * 10, *"512", "enter")
        await pilot.pause()
        assert repo.get("rn50_lr05").dirty
        await pilot.press("ctrl+s")
        await pilot.pause()
        text = child_file.read_text()
        assert "_base_: resnet50_base.yaml" in text
        assert "batch_size: 512" in text and "lr: 0.05" in text
        assert "momentum" not in text  # only the diff is stored
        assert "batch_size: 256" in (config_dir / "resnet50_base.yaml").read_text()

        # create a task from it and see it in the schedule view
        await pilot.press("T")
        await pilot.pause()
        (t,) = app.store.tasks.values()
        assert t.config_path == str(child_file.resolve())
        assert t.estimated_duration == timedelta(hours=3)
        await pilot.press("1")
        await pilot.pause()
        assert [c.task_model.id for c in app.screen.query(TaskCard)] == [t.id]


async def test_runtime_disabled_by_default(config_dir):
    app = make_app(config_dir, [task("a")])
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause()
        await pilot.press("r")
        assert app.runtime is None
        assert all(a.actual_start is None for a in app.store.assignments.values())


async def test_mock_runtime_records_actual_times(config_dir):
    from gpu_scheduler.runtime.mock import MockRuntime

    a = task("alpha", hours=1)
    app = make_app(config_dir, [a])
    app.runtime = MockRuntime(speedup=1e9)
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause()
        tl = app.screen.query_one(GPUTimeline)
        asg = app.assignment_service.assign(a.id, [1], tl.state.cursor_time).assignment
        await pilot.press("r")
        assert app.store.tasks[a.id].status == TaskStatus.RUNNING
        app.poll_runs()
        assert asg.actual_end is not None and asg.actual_start is not None
        assert asg.actual_gpu_ids == (1,)
        assert app.store.tasks[a.id].status == TaskStatus.COMPLETED
