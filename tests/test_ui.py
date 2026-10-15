from __future__ import annotations

import shutil
from datetime import timedelta
from pathlib import Path

import pytest

from zgm.app import ZGMApp
from zgm.models import Task, TaskStatus
from zgm.screens.config import ConfigScreen
from zgm.screens.schedule import ScheduleScreen
from zgm.widgets.gpu_row import LABEL_WIDTH, GPURow
from zgm.widgets.gpu_timeline import GPUTimeline
from zgm.widgets.task_card import TaskCard

EXAMPLES = Path(__file__).parent.parent / "examples" / "configs"
SIZE = (140, 42)


@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    d = tmp_path / "configs"
    shutil.copytree(EXAMPLES, d)
    return d


def make_app(config_dir: Path, tasks: list[Task] | None = None) -> ZGMApp:
    app = ZGMApp(mock_gpus=4, config_dir=config_dir, stats_interval=0)
    for t in tasks or []:
        app.store.add_task(t)
    return app


def task(name: str, gpus: int = 1) -> Task:
    return Task(name, f"echo {name}", gpus)


async def test_boots_with_one_row_per_gpu(config_dir):
    app = make_app(config_dir, [task("a"), task("b")])
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause()
        assert isinstance(app.screen, ScheduleScreen)
        assert len(app.screen.query(GPURow)) == 4
        assert len(app.screen.query(TaskCard)) == 2


def planned_names(app) -> list[str]:
    return [app.store.tasks[a.task_id].name for a in app.assignment_service.planned()]


def strip_text(row: GPURow) -> str:
    """Everything a row currently renders, as plain text (one line per screen line)."""
    return "\n".join(row.render_line(y).text for y in range(row.size.height))


def card_offset(tl: GPUTimeline, col: int, right_half: bool = False) -> tuple[int, int]:
    x = tl.state.col_x(col)
    return LABEL_WIDTH + x + (tl.state.card_width - 2 if right_half else 1), 0


async def test_keyboard_assign_appends_to_queue(config_dir):
    a, b = task("alpha"), task("beta")
    app = make_app(config_dir, [a, b])
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause()
        await pilot.press("a")  # placement mode on the selected card (alpha)
        assert app.screen.placement is not None
        await pilot.press("down", "enter")
        await pilot.pause()
        assert app.screen.placement is None
        (asg,) = app.store.assignments.values()
        assert asg.task_id == a.id and asg.gpu_ids == (1,)
        assert app.store.tasks[a.id].status == TaskStatus.PLANNED
        # second task goes behind the first on the same GPU
        app.screen.queue.select(b.id)
        await pilot.press("a", "enter")
        await pilot.pause()
        ab = app.store.active_assignment(b.id)
        assert ab.gpu_ids == (1,) and ab.planned_start == asg.planned_end


async def test_keyboard_insert_before_existing(config_dir):
    a, b = task("alpha"), task("beta")
    app = make_app(config_dir, [a, b])
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause()
        app.assignment_service.assign(a.id, [0])
        await pilot.pause()
        app.screen.queue.select(b.id)
        await pilot.press("a")
        assert app.screen.timeline.state.cursor_col == 1  # end of the queue
        await pilot.press("left", "enter")
        await pilot.pause()
        assert planned_names(app) == ["beta", "alpha"]


async def test_invalid_placement_is_refused(config_dir):
    big = task("huge", gpus=8)
    app = make_app(config_dir, [big])
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause()
        screen: ScheduleScreen = app.screen
        await pilot.press("a")
        assert not screen.preview.result.ok
        await pilot.press("enter")
        await pilot.pause()
        assert not app.store.assignments
        assert screen.placement is not None and "needs 8" in screen.last_error
        await pilot.press("escape")
        assert screen.placement is None


async def test_reorder_with_brackets(config_dir):
    tasks = [task(n) for n in ("one", "two", "three")]
    app = make_app(config_dir, tasks)
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause()
        for t in tasks:
            app.assignment_service.assign(t.id, [0])
        await pilot.press("tab")  # timeline; cursor on column 0
        await pilot.press("right", "right")
        await pilot.press("left_square_bracket")
        await pilot.pause()
        assert planned_names(app) == ["one", "three", "two"]
        assert app.screen.timeline.state.cursor_col == 1  # cursor follows the card


async def test_full_name_centred_in_fixed_width_card(config_dir):
    long = "a-very-long-experiment-name-for-attention-head-ablation"
    t = task(long)
    app = make_app(config_dir, [t])
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause()
        app.assignment_service.assign(t.id, [0])
        await pilot.pause()
        tl = app.screen.timeline
        row = app.screen.query_one("#gpu-row-0", GPURow)
        x = LABEL_WIDTH + tl.state.col_x(0)
        w = tl.state.card_width
        lines = [line[x: x + w] for line in strip_text(row).splitlines()]
        text = "".join(line.strip() for line in lines if line.strip())
        assert text == long
        assert row.size.height >= 4  # grew to fit the wrapped name
        # every name line is centred inside the card
        for line in lines:
            if line.strip():
                left = len(line) - len(line.lstrip())
                right = len(line) - len(line.rstrip())
                assert abs(left - right) <= 1


async def test_queue_zone_shows_no_times(config_dir):
    import re

    a, b = task("alpha"), task("beta", gpus=2)
    app = make_app(config_dir, [a, b])
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause()
        app.assignment_service.assign(a.id, [0])
        app.assignment_service.assign(b.id, [1, 2])
        await pilot.pause()
        tl = app.screen.timeline
        x0 = LABEL_WIDTH + tl.state.future_x0
        clock = re.compile(r"\d{1,2}:\d{2}")
        queue_text = [line[x0:] for g in range(4)
                      for line in strip_text(app.screen.query_one(f"#gpu-row-{g}", GPURow)).splitlines()]
        axis = [tl.axis.render_line(y).text for y in range(2)]
        queue_text += [line[x0:] for line in axis]
        assert not any(clock.search(t) or "→" in t for t in queue_text)
        assert "#1" in axis[0] and "#2" in axis[0]
        assert clock.search(axis[0][LABEL_WIDTH: x0])  # history keeps its clock


async def test_multi_gpu_card_aligns_across_rows(config_dir):
    a, b = task("solo"), task("pair", gpus=2)
    app = make_app(config_dir, [a, b])
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause()
        app.assignment_service.assign(a.id, [0])
        app.assignment_service.assign(b.id, [1, 2])
        await pilot.pause()
        rows = {g: app.screen.query_one(f"#gpu-row-{g}", GPURow) for g in range(4)}
        assert [c.col for c in rows[1].cards] == [1] and [c.col for c in rows[2].cards] == [1]
        assert rows[0].cards[0].col == 0 and not rows[3].cards
        assert "pair" in strip_text(rows[1]) and "pair" not in strip_text(rows[2])


async def test_history_left_of_now_uses_actual_times(config_dir):
    from datetime import datetime

    from zgm.models import Assignment, AssignmentState

    t = task("finished")
    t.status = TaskStatus.COMPLETED
    app = make_app(config_dir, [t])
    now = datetime.now()
    app.store.save_assignment(Assignment(
        t.id, (3,), now - timedelta(hours=3), now - timedelta(hours=2),
        actual_start=now - timedelta(hours=2), actual_end=now - timedelta(hours=1),
        state=AssignmentState.COMPLETED), "completed")
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause()
        tl = app.screen.timeline
        (block,) = app.screen.query_one("#gpu-row-3", GPURow).past
        x0, x1 = tl.state.span(now - timedelta(hours=2), now - timedelta(hours=1))
        assert abs(block.x0 - x0) <= 1 and abs(block.x1 - x1) <= 1
        assert block.x1 <= tl.state.divider_x
        assert not app.screen.timeline.slots  # nothing in the queue zone


async def test_mouse_drag_card_onto_timeline(config_dir):
    a, b = task("alpha"), task("beta")
    app = make_app(config_dir, [a, b])
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause()
        app.assignment_service.assign(a.id, [0])
        await pilot.pause()
        screen: ScheduleScreen = app.screen
        card = screen.query_one(f"#card-{b.id}", TaskCard)
        row = screen.query_one("#gpu-row-2", GPURow)
        tl = screen.timeline
        await pilot.mouse_down(card, offset=(3, 1))
        await pilot.hover(card, offset=(6, 2))  # start dragging
        await pilot.hover(row, offset=card_offset(tl, 0))  # left half of alpha's column
        assert screen.placement is not None and screen.placement.via_mouse
        assert screen.preview.position == 0
        await pilot.mouse_up(row, offset=card_offset(tl, 0))
        await pilot.pause()
        assert planned_names(app) == ["beta", "alpha"]
        assert app.store.active_assignment(b.id).gpu_ids == (2,)
        assert screen.placement is None


async def test_mouse_drag_card_after_existing(config_dir):
    a, b = task("alpha"), task("beta")
    app = make_app(config_dir, [a, b])
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause()
        app.assignment_service.assign(a.id, [0])
        await pilot.pause()
        screen = app.screen
        card = screen.query_one(f"#card-{b.id}", TaskCard)
        row = screen.query_one("#gpu-row-0", GPURow)
        await pilot.mouse_down(card, offset=(3, 1))
        await pilot.hover(card, offset=(6, 2))
        await pilot.hover(row, offset=card_offset(screen.timeline, 0, right_half=True))
        await pilot.mouse_up(row, offset=card_offset(screen.timeline, 0, right_half=True))
        await pilot.pause()
        assert planned_names(app) == ["alpha", "beta"]


async def test_mouse_drag_existing_block(config_dir):
    a, b = task("alpha"), task("beta")
    app = make_app(config_dir, [a, b])
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause()
        app.assignment_service.assign(a.id, [0])
        app.assignment_service.assign(b.id, [0])
        await pilot.pause()
        screen = app.screen
        tl = screen.timeline
        row0 = screen.query_one("#gpu-row-0", GPURow)
        row3 = screen.query_one("#gpu-row-3", GPURow)
        await pilot.mouse_down(row0, offset=card_offset(tl, 1))  # beta
        await pilot.hover(row3, offset=card_offset(tl, 0))
        await pilot.mouse_up(row3, offset=card_offset(tl, 0))
        await pilot.pause()
        assert planned_names(app) == ["beta", "alpha"]
        assert app.store.active_assignment(b.id).gpu_ids == (3,)
        assert len(app.store.assignments) == 2


async def test_drop_on_history_cancels(config_dir):
    a = task("alpha")
    app = make_app(config_dir, [a])
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause()
        screen = app.screen
        card = screen.query_one(TaskCard)
        row = screen.query_one("#gpu-row-1", GPURow)
        await pilot.mouse_down(card, offset=(3, 1))
        await pilot.hover(card, offset=(6, 2))
        await pilot.mouse_up(row, offset=(LABEL_WIDTH + 3, 0))
        await pilot.pause()
        assert not app.store.assignments and screen.placement is None


async def test_unassign_from_timeline(config_dir):
    a = task("alpha")
    app = make_app(config_dir, [a])
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause()
        app.assignment_service.assign(a.id, [0])
        await pilot.press("tab")  # focus timeline; cursor sits on the first card
        await pilot.press("u")
        await pilot.pause()
        assert not app.store.assignments
        assert app.store.tasks[a.id].status == TaskStatus.PENDING


async def test_card_width_option(config_dir):
    app = ZGMApp(mock_gpus=2, config_dir=config_dir, stats_interval=0, card_width=30)
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause()
        assert app.screen.timeline.state.card_width == 30


async def test_ui_is_english_only(config_dir):
    import re

    app = make_app(config_dir, [task("a")])
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause()
        assert not re.search(r"[\u4e00-\u9fff]", str(app.screen.query_one("#view-tabs").render()))
    src = Path(__file__).parent.parent / "src"
    offenders = [p for p in src.rglob("*.py") if re.search(r"[\u4e00-\u9fff]", p.read_text(encoding="utf-8"))]
    assert offenders == []


async def test_new_task_dialog(config_dir):
    app = make_app(config_dir)
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause()
        await pilot.press("n")
        await pilot.press(*"sweep", "enter")
        await pilot.press("backspace", "2", "enter")
        await pilot.press(*["enter"] * 4)  # defaults: run, environment, working dir, env vars
        await pilot.pause()
        (t,) = app.store.tasks.values()
        assert (t.name, t.required_gpus, t.estimated_duration) == ("sweep", 2, None)
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
        text = child_file.read_text(encoding="utf-8")
        assert "_base_: resnet50_base.yaml" in text
        assert "batch_size: 512" in text and "lr: 0.05" in text
        assert "momentum" not in text  # only the diff is stored
        assert "batch_size: 256" in (config_dir / "resnet50_base.yaml").read_text(encoding="utf-8")

        # create a task from it and see it in the schedule view
        await pilot.press("T")
        await pilot.pause()
        (t,) = app.store.tasks.values()
        assert t.config_path == str(child_file.resolve())
        assert t.estimated_duration is None  # this exact config never ran
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
    from zgm.runtime.mock import MockRuntime

    a = task("alpha")
    app = make_app(config_dir, [a])
    app.runtime = MockRuntime(speedup=1e9)
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause()
        asg = app.assignment_service.assign(a.id, [1]).assignment
        await pilot.press("tab", "r")  # timeline cursor on the planned card
        assert app.store.tasks[a.id].status == TaskStatus.RUNNING
        app.poll_runs()
        assert asg.actual_end is not None and asg.actual_start is not None
        assert asg.actual_gpu_ids == (1,)
        assert app.store.tasks[a.id].status == TaskStatus.COMPLETED
        assert app.store.tasks[a.id] not in app.store.waiting_tasks()


async def test_queue_hides_planned_and_running_tasks(config_dir):
    a, b = task("alpha"), task("beta")
    app = make_app(config_dir, [a, b])
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause()
        await pilot.press("a", "enter")  # plan alpha (the selected card)
        await pilot.pause()
        ids = [c.task_model.id for c in app.screen.query(TaskCard)]
        assert ids == [b.id]
        assert app.screen.queue.selected_id == b.id  # selection moved on
        assert app.screen.focused is app.screen.queue  # back to the queue
        app.assignment_service.unassign(app.store.active_assignment(a.id).id)
        await pilot.pause()
        assert {c.task_model.id for c in app.screen.query(TaskCard)} == {a.id, b.id}


async def test_rename_task_from_queue_and_timeline(config_dir):
    a, b = task("alpha"), task("beta")
    app = make_app(config_dir, [a, b])
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause()
        app.assignment_service.assign(b.id, [0])
        await pilot.pause()
        await pilot.press("e")  # queue: alpha
        await pilot.pause()
        await pilot.press(*["backspace"] * 10, *"first", *["enter"] * 6)
        await pilot.pause()
        assert app.store.tasks[a.id].name == "first"
        await pilot.press("tab", "e")  # timeline: beta's card
        await pilot.pause()
        await pilot.press(*["backspace"] * 10, *"second", *["enter"] * 6)
        await pilot.pause()
        assert app.store.tasks[b.id].name == "second"
        row = app.screen.query_one("#gpu-row-0", GPURow)
        assert "second" in strip_text(row)


async def test_edit_gpu_count_of_planned_task(config_dir):
    a = task("alpha")
    app = make_app(config_dir, [a])
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause()
        app.assignment_service.assign(a.id, [1])
        await pilot.press("tab", "e")
        await pilot.pause()
        await pilot.press("enter", "backspace", "2", *["enter"] * 5)
        await pilot.pause()
        assert app.store.active_assignment(a.id).gpu_ids == (1, 2)
        assert app.store.tasks[a.id].required_gpus == 2


async def _start_running(app, pilot, t):
    from zgm.runtime.mock import MockRuntime

    app.runtime = MockRuntime(speedup=1)  # stays running
    app.assignment_service.assign(t.id, [0])
    await pilot.pause()
    await pilot.press("tab", "r")
    await pilot.pause()
    assert app.store.tasks[t.id].status == TaskStatus.RUNNING


@pytest.mark.parametrize("choice,kept", [("requeue", True), ("delete", False)])
async def test_stop_running_task(config_dir, choice, kept):
    from zgm.models import AssignmentState
    from zgm.widgets.dialogs import ChoiceScreen

    a = task("alpha")
    app = make_app(config_dir, [a])
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause()
        await _start_running(app, pilot, a)
        await pilot.press("s")
        await pilot.pause()
        assert isinstance(app.screen, ChoiceScreen)
        await pilot.click(f"#choice-{choice}")
        await pilot.pause()
        assert (a.id in app.store.tasks) is kept
        if kept:
            assert app.store.tasks[a.id].status == TaskStatus.PENDING
            assert [c.task_model.id for c in app.screen.query(TaskCard)] == [a.id]
            (run,) = app.store.assignments_for_task(a.id)
            assert run.state == AssignmentState.CANCELLED and run.actual_end is not None
            assert a.estimated_duration is None  # a stopped run is not a sample
        assert not app._runs


async def test_stop_can_be_called_off(config_dir):
    a = task("alpha")
    app = make_app(config_dir, [a])
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause()
        await _start_running(app, pilot, a)
        await pilot.press("s")
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
        assert app.store.tasks[a.id].status == TaskStatus.RUNNING


async def test_rename_config_repoints_tasks(config_dir):
    app = make_app(config_dir)
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("2")
        await pilot.pause()
        screen: ConfigScreen = app.screen
        repo = app.config_repo
        screen.current = repo.get("resnet50_adamw").path
        screen.refresh_params()
        await pilot.press("T")  # a task from the config, before the rename
        await pilot.pause()
        await pilot.press("r")
        await pilot.pause()
        await pilot.press(*["backspace"] * 30, *"rn50_adamw", "enter")
        await pilot.pause()
        assert (config_dir / "rn50_adamw.yaml").exists()
        assert "_base_: rn50_adamw.yaml" in (config_dir / "resnet50_adamw_bs512.yaml").read_text(encoding="utf-8")
        (t,) = app.store.tasks.values()
        assert t.config_path.endswith("rn50_adamw.yaml")
        assert t.launch.args == ["--config", "{config}"]  # resolved from config_path at start
        assert screen.current.name == "rn50_adamw.yaml"


async def test_run_user_script_with_local_runtime(config_dir, tmp_path):
    import sys

    from zgm.models import AssignmentState, LaunchSpec
    from zgm.runtime.local import LocalRuntime

    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "train.py").write_text(
        "import os\nprint('on', os.environ['CUDA_VISIBLE_DEVICES'])\n", encoding="utf-8")
    t = Task("job", launch=LaunchSpec(cwd=str(proj), python=sys.executable, script="train.py"))
    app = make_app(config_dir, [t])
    app.runtime = LocalRuntime(tmp_path / "logs")
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause()
        app.assignment_service.assign(t.id, [2])
        await pilot.pause()
        await pilot.press("tab", "r")
        await pilot.pause()
        assert app.store.tasks[t.id].status == TaskStatus.RUNNING
        for _ in range(100):
            app.poll_runs()
            if app.store.tasks[t.id].status != TaskStatus.RUNNING:
                break
            await pilot.pause(0.1)
        assert app.store.tasks[t.id].status == TaskStatus.COMPLETED
        (a,) = app.store.assignments_for_task(t.id)
        assert (a.state, a.actual_gpu_ids) == (AssignmentState.COMPLETED, (2,))
        log = app.log_path(t)
        assert log is not None and "on 2" in log.read_text(encoding="utf-8")


async def test_run_reports_launch_errors(config_dir, tmp_path):
    from zgm.models import LaunchSpec
    from zgm.runtime.local import LocalRuntime

    t = Task("job", launch=LaunchSpec(cwd=str(tmp_path), script="missing.py"))
    app = make_app(config_dir, [t])
    app.runtime = LocalRuntime(tmp_path / "logs")
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause()
        app.assignment_service.assign(t.id, [0])
        await pilot.pause()
        await pilot.press("tab", "r")
        await pilot.pause()
        # nothing started: back in the waiting queue as failed, off the timeline
        assert app.store.tasks[t.id].status == TaskStatus.FAILED
        assert app.store.active_assignment(t.id) is None
        assert t in app.store.waiting_tasks()
        events = [e["event"] for e in app.store.storage.history(t.id)]
        assert "launch_failed" in events
