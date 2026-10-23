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
        await pilot.press("t")  # timeline; cursor on column 0
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
        assert "#1" in axis[0] and "#2" not in axis[0]  # a (GPU 0) and b (GPU 1-2): both next
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
        # each GPU's row is packed: the pair is next on GPUs 1 and 2, beside solo
        assert [c.col for c in rows[1].cards] == [0] and [c.col for c in rows[2].cards] == [0]
        assert rows[0].cards[0].col == 0 and not rows[3].cards
        assert "pair" in strip_text(rows[1]) and "pair" not in strip_text(rows[2])


async def test_running_work_shares_the_first_column(config_dir):
    from zgm.runtime.mock import MockRuntime

    a, b, c = task("alpha"), task("beta"), task("gamma")
    app = make_app(config_dir, [a, b, c])
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause()
        app.runtime = MockRuntime(speedup=1)  # stays running
        app.assignment_service.assign(a.id, [0])
        app.assignment_service.assign(b.id, [1])
        app.assignment_service.assign(c.id, [0])
        for t in (a, b):
            app.executor.request_run_now(t)
        app.dispatch()
        await pilot.pause()
        assert app.store.tasks[b.id].status == TaskStatus.RUNNING
        rows = {g: app.screen.query_one(f"#gpu-row-{g}", GPURow) for g in range(2)}
        cols = {app.store.tasks[k.task_id].name: k.col for r in rows.values() for k in r.cards}
        assert cols == {"alpha": 0, "beta": 0, "gamma": 1}
        tl = app.screen.query_one(GPUTimeline)
        assert tl.n_items == 2
        tl.set_cursor(1, 0)
        assert tl.slot_under_cursor().task_id == b.id
        tl.set_cursor(0, 1)
        assert tl.slot_under_cursor().task_id == c.id


async def test_cards_stay_in_view_when_the_queue_shrinks(config_dir):
    tasks = [task(f"t{i}") for i in range(12)]
    app = make_app(config_dir, tasks)
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause()
        for t in tasks:
            app.assignment_service.assign(t.id, [0])
        await pilot.pause()
        tl = app.screen.query_one(GPUTimeline)
        tl.state.scroll_future(99, tl.n_items)  # scrolled to the end of a long queue
        tl._changed()
        await pilot.pause()
        assert tl.state.future_scroll > 3
        for t in tasks[3:]:  # most of the queue goes away elsewhere (done, moved)
            app.assignment_service.unassign(app.store.active_assignment(t.id).id)
        app.screen.refresh_timeline()
        await pilot.pause()
        row = app.screen.query_one("#gpu-row-0", GPURow)
        assert [c.col for c in row.cards] == [0, 1, 2]
        # the scroll came back inside the queue: its last card and the end slot are in view
        assert tl.state.future_scroll <= max(0, tl.n_items + 1 - tl.state.visible_cols)
        assert tl.state.col_x(2) is not None


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
        row = screen.query_one("#gpu-row-0", GPURow)
        tl = screen.timeline
        await pilot.mouse_down(card, offset=(3, 1))
        await pilot.hover(card, offset=(6, 2))  # start dragging
        await pilot.hover(row, offset=card_offset(tl, 0))  # left half of alpha's card
        assert screen.placement is not None and screen.placement.via_mouse
        assert screen.preview.position == 0
        await pilot.mouse_up(row, offset=card_offset(tl, 0))
        await pilot.pause()
        assert planned_names(app) == ["beta", "alpha"]
        assert app.store.active_assignment(b.id).gpu_ids == (0,)
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
        assert app.store.active_assignment(b.id).gpu_ids == (3,)
        assert len(app.store.assignments) == 2
        # nothing else is queued on GPU 3: beta is next there, beside alpha
        assert [c.col for c in row3.cards] == [0] and [c.col for c in row0.cards] == [0]


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
        await pilot.press("t")  # focus timeline; cursor sits on the first card
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



async def test_config_shows_when_it_last_ran(config_dir):
    from datetime import datetime

    from zgm.services.task_store import add_completed_run

    app = make_app(config_dir)
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("2")
        await pilot.pause()
        screen: ConfigScreen = app.screen
        repo = app.config_repo
        table = screen.query_one("#config-table")
        screen.current = repo.get("resnet50_base").path
        screen.refresh_params()
        assert table.border_subtitle == "never run"

        start = datetime(2026, 10, 15, 14, 5)
        add_completed_run(app.store, repo.make_task(repo.get("resnet50_base")), (0,),
                          start, timedelta(hours=2, minutes=30))
        app.store.notify()  # as the executor does when a run ends
        await pilot.pause()  # the change reaches the screen without reselecting
        assert table.border_subtitle == "last run 10-15 14:05 · completed, 2h30m"
        # a child with other parameters has not run
        screen.current = repo.get("resnet50_adamw").path
        screen.refresh_params()
        assert table.border_subtitle == "never run"


async def test_config_tree_widens_for_deep_names(config_dir):
    deep = "resnet50_base"
    for i in range(4):  # a chain of five levels with long names
        name = f"level_{i}_with_a_rather_long_config_name"
        (config_dir / f"{name}.yaml").write_text(f"_base_: {deep}.yaml\nx{i}: 1\n", encoding="utf-8")
        deep = name
    app = make_app(config_dir)
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("2")
        await pilot.pause()
        tree = app.screen.query_one("#config-tree")
        assert 36 < tree.size.width <= SIZE[0] // 2


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
        await pilot.press("t", "r")  # timeline cursor on the planned card
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
        await pilot.press("t", "e")  # timeline: beta's card
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
        await pilot.press("t", "e")
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
    await pilot.press("t", "r")
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
        await pilot.press("K")
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
        await pilot.press("K")
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
        await pilot.press("t", "r")
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
        await pilot.press("t", "r")
        await pilot.pause()
        # nothing started: back in the waiting queue as failed, off the timeline
        assert app.store.tasks[t.id].status == TaskStatus.FAILED
        assert app.store.active_assignment(t.id) is None
        assert t in app.store.waiting_tasks()
        events = [e["event"] for e in app.store.storage.history(t.id)]
        assert "launch_failed" in events


async def test_settings_view_switches_auto_fill(config_dir):
    from zgm.widgets.text_toggle import TextToggle
    from zgm.screens.settings import SettingsScreen

    app = make_app(config_dir)
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("3")
        await pilot.pause()
        assert isinstance(app.screen, SettingsScreen)
        autostart = app.screen.query_one("#setting-autostart", TextToggle)
        autofill = app.screen.query_one("#setting-autofill", TextToggle)
        assert autostart.value and autofill.value
        autofill.focus()
        await pilot.press("enter")
        await pilot.pause()
        assert app.store.storage.get_setting("autofill") == "0"
        # changed elsewhere (e.g. another TUI) while this view was not shown
        await pilot.press("1")
        app.executor.autostart = False
        await pilot.press("3")
        await pilot.pause()
        assert not app.screen.query_one("#setting-autostart", TextToggle).value
        assert app.store.storage.get_setting("autostart") == "0"


async def test_panels_focus_by_letter_and_go_full_screen(config_dir, tmp_path):
    from zgm.widgets.dialogs import ChoiceScreen
    from zgm.widgets.queue_overview import QueueOverview

    a = task("alpha")
    a.config_path = str(config_dir / "table3" / "a.yaml")
    b = task("bravo")
    b.config_path = str(config_dir / "ablation" / "b.yaml")
    c = task("charlie")
    c.autofill = True
    app = make_app(config_dir, [a, b, c])
    app.executor.autofill = False  # so nothing would start c on its own
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause()
        screen = app.screen
        await pilot.press("t")
        assert app.focused is screen.timeline
        await pilot.press("d")
        assert app.focused.id == "task-details"
        await pilot.press("f")  # details alone
        await pilot.pause()
        assert screen.zoomed == "details" and not screen.timeline.display
        await pilot.press("q")  # full screen follows the focus
        await pilot.pause()
        overview = screen.query_one(QueueOverview)
        assert screen.zoomed == "queue" and overview.display and app.focused is overview
        lines = [str(overview.get_option_at_index(i).prompt).strip()
                 for i in range(overview.option_count)]
        heads = [x.split()[0] for x in lines if x and not x.startswith(("alpha", "bravo", "charlie"))]
        assert heads == ["AUTO-FILL", "no", "MANUAL", "ablation/", "table3/"]
        # space moves the highlighted task (bravo, the first manual one) to auto-fill
        overview.highlighted = overview.get_option_index(b.id)
        await pilot.press("space")
        await pilot.pause()
        assert app.store.tasks[b.id].autofill
        overview.highlighted = overview.get_option_index(b.id)
        await pilot.press("plus", "plus", "minus")
        await pilot.pause()
        assert app.store.tasks[b.id].priority == 1
        await pilot.press("escape")  # leaves full screen
        await pilot.pause()
        assert screen.zoomed is None and screen.timeline.display and screen.queue.display
        await pilot.press("escape")  # nothing left to cancel: the menu
        await pilot.pause()
        from zgm.screens.menu import MenuScreen
        assert isinstance(app.screen, MenuScreen)
        assert "v0.1" in str(app.screen.query_one("#menu-version").render())
        await pilot.press("down", "up", "enter")  # Options is first
        await pilot.pause()
        from zgm.screens.settings import SettingsScreen
        assert isinstance(app.screen, SettingsScreen)


async def test_each_gpu_row_is_packed_from_the_first_queue_column(config_dir):
    # queued first on GPUs 0 and 1, then on GPU 3: GPU 3's card must not wait two columns
    a, b, c, d = task("on-0"), task("on-1"), task("on-3"), task("on-0-again")
    app = make_app(config_dir, [a, b, c, d])
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause()
        svc = app.assignment_service
        for t, g in [(a, 0), (b, 1), (c, 3), (d, 0)]:
            svc.assign(t.id, [g])
        await pilot.pause()
        rows = {g: app.screen.query_one(f"#gpu-row-{g}", GPURow) for g in range(4)}
        assert [x.col for x in rows[3].cards] == [0]
        assert [x.col for x in rows[0].cards] == [0, 1]
        # keyboard placement at a GPU's first column goes before the work queued there
        e = app.store.add_task(task("new"))
        screen = app.screen
        screen.queue.select(e.id)
        screen.start_placement(e.id)
        screen.timeline.state.cursor_gpu = 0
        screen.timeline.state.cursor_col = 0
        screen.refresh_timeline()
        assert screen.preview.position == 0 and screen.timeline.state.cursor_col == 0
        await pilot.press("enter")
        await pilot.pause()
        assert planned_names(app)[0] == "new"
        assert [x.col for x in rows[0].cards] == [0, 1, 2] and [x.col for x in rows[3].cards] == [0]


def test_reordering_steps_over_work_on_the_same_gpus_only():
    from zgm.services.scheduler import AssignmentService
    from zgm.services.task_store import TaskStore

    store = TaskStore()
    svc = AssignmentService(store, [0, 1])
    ts = {n: store.add_task(task(n)) for n in ("a0", "b1", "c0", "d1")}
    ids = {n: svc.assign(t.id, [int(n[1])]).assignment.id for n, t in ts.items()}
    names = lambda: [store.tasks[x.task_id].name for x in svc.planned()]
    assert svc.shift(ids["c0"], -1).ok
    assert names() == ["c0", "a0", "b1", "d1"]  # before a0; b1 and d1 keep their order
    assert svc.shift(ids["c0"], -1).ok and names()[0] == "c0"  # already first on GPU 0
    assert svc.shift(ids["b1"], 1).ok
    assert names() == ["c0", "a0", "d1", "b1"]


async def test_host_panel_shows_cpu_memory_and_disk(config_dir):
    from zgm.widgets.host_panel import HostPanel

    app = make_app(config_dir)
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause()
        panel = app.screen.query_one(HostPanel)
        text = str(panel.render())
        assert panel.display and "host" in str(panel.border_title)
        assert "CPU" in text and "MEM" in text and "DISK" in text and "GiB free" in text
        await pilot.press("d", "f")  # another panel full screen: the row goes
        assert not app.screen.query_one("#top-row").display
        await pilot.press("escape")
        assert app.screen.query_one("#top-row").display and panel.display


async def test_queue_splits_autofill_and_manual_by_priority(config_dir):
    tasks = [task("m-low"), task("a-low"), task("m-high"), task("a-high"), task("a-mid")]
    for t in tasks:
        t.autofill = t.name.startswith("a-")
        t.priority = {"low": 0, "mid": 1, "high": 2}[t.name.split("-")[1]]
    app = make_app(config_dir, tasks)
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause()
        names = lambda kind: [c.task_model.name for c in
                              app.screen.query(f"#queue-{kind}-strip TaskCard")]
        assert names("auto") == ["a-high", "a-mid", "a-low"]
        assert names("manual") == ["m-high", "m-low"]
        queue = app.screen.queue
        assert [app.store.tasks[i].name for i in queue.order] == \
            ["a-high", "a-mid", "a-low", "m-high", "m-low"]
        queue.focus()
        await pilot.press("end")
        assert app.store.tasks[queue.selected_id].name == "m-low"

        tasks[0].autofill = True  # m-low goes to the auto-fill half
        app.store.update_task(tasks[0])
        await pilot.pause()
        assert names("auto") == ["a-high", "a-mid", "m-low", "a-low"]  # oldest first
        assert names("manual") == ["m-high"]


async def test_details_show_the_config_and_only_its_parents(config_dir, tmp_path):
    from zgm.widgets.config_chain import ConfigChain

    (tmp_path / "base.yaml").write_text("lr: 1\n")
    (tmp_path / "mid.yaml").write_text("_base_: base.yaml\nlr: 2\n")
    (tmp_path / "sibling.yaml").write_text("_base_: base.yaml\nlr: 3\n")
    (tmp_path / "other.yaml").write_text("x: 1\n")
    (tmp_path / "leaf.yaml").write_text("_base_: [mid.yaml, other.yaml]\nlr: 4\n")
    t = task("a")
    t.config_path = str(tmp_path / "leaf.yaml")
    app = make_app(config_dir, [t])
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause()
        tree = app.screen.query_one(ConfigChain)
        rows, node = [], tree.root
        while node.children:
            assert len(node.children) == 1  # no siblings anywhere
            node = node.children[0]
            rows.append(str(node.label))
        assert rows == ["base", "mid", "leaf ⧉2"]


async def test_tab_goes_through_every_panel(config_dir):
    app = make_app(config_dir, [task("a")])
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause()
        seen = []
        for i in range(5):
            if i:
                await pilot.press("tab")
            seen.append(app.focused.id)
        # no cluster here, so no cluster panel; then round again
        assert seen == ["task-queue", "gpu-timeline", "task-details", "host-panel", "task-queue"]
        await pilot.press("shift+tab")
        assert app.focused.id == "host-panel"
        await pilot.press("f")  # full screen follows Tab
        assert app.screen.zoomed == "host"
        await pilot.press("tab")
        assert app.screen.zoomed == "queue" and app.focused.id != "host-panel"


async def test_terse_cards_idle_count_and_keys_in_settings(config_dir):
    a, b = task("a"), task("b")
    b.status = TaskStatus.FAILED
    app = make_app(config_dir, [a, b])
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause()
        cards = {c.task_model.name: str(c.render()) for c in app.screen.query(TaskCard)}
        assert "pending" not in cards["a"] and "failed" in cards["b"]
        assert "4 gpus · 4 idle" in str(app.screen.timeline.border_title)
        await pilot.press("3")
        await pilot.pause()
        sections = dict((t, [b.description for b in bs]) for t, bs in app.screen.key_sections())
        keys = " ".join(d for ds in sections.values() for d in ds)
        assert app.screen.query_one("#keys").display
        assert "unassign" in keys and "history: zoom in" in keys and "derive" in keys


async def test_idle_view_stops_redrawing_and_the_first_key_only_wakes_it(config_dir):
    import time

    app = ZGMApp(mock_gpus=4, config_dir=config_dir, stats_interval=0, idle_pause=60)
    app.store.add_task(task("a"))
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause()
        app._last_input = time.monotonic() - 61
        app.tick()
        assert app.idle and "IDLE" in str(app.screen.query_one("#status-line").render())
        app.store.add_task(task("b"))  # changes while idle are not drawn
        await pilot.pause()
        assert len(app.screen.query(TaskCard)) == 1
        await pilot.press("x")  # wakes; does not delete anything
        await pilot.pause()
        assert not app.idle and len(app.screen.query(TaskCard)) == 2
        assert len(app.store.tasks) == 2
        assert "IDLE" not in str(app.screen.query_one("#status-line").render())


async def test_a_tui_that_runs_the_queue_keeps_running_it_while_idle(config_dir):
    import time

    from zgm.runtime.mock import MockRuntime

    a = task("alpha")
    app = make_app(config_dir, [a])
    app.runtime = MockRuntime(speedup=1)
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause()
        app._last_input = time.monotonic() - app.idle_pause - 1
        app.tick()
        assert app.idle
        app.assignment_service.assign(a.id, [1])
        app.tick()
        assert app.store.tasks[a.id].status == TaskStatus.RUNNING
