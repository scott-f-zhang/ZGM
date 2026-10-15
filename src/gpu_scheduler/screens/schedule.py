"""Schedule view: task queue (top), details (middle), GPU timeline (bottom)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.screen import Screen
from textual.widgets import Footer, Static

from ..models import AssignmentState, Task, TaskStatus
from ..services.scheduler import Result, Slot
from ..widgets.dialogs import new_task_form
from ..widgets.gpu_row import GPURow
from ..widgets.gpu_timeline import GPUTimeline, fmt_when
from ..widgets.task_card import TaskCard
from ..widgets.task_details import TaskDetails
from ..widgets.task_queue import TaskQueue
from ..widgets.view_tabs import ViewTabs


@dataclass
class Placement:
    """A placement in progress (keyboard 'a' mode or a mouse drag)."""

    task_id: str
    assignment_id: str | None = None  # moving an existing planned block
    via_mouse: bool = False
    from_queue: bool = False  # hand focus back to the queue when done


@dataclass
class PlacementPreview:
    gpu_ids: tuple[int, ...]
    position: int  # index in the planned queue
    result: Result
    ghost: Slot | None
    slots: list[Slot]


class ScheduleScreen(Screen):
    BINDINGS = [
        Binding("a", "place", "Assign"),
        Binding("enter", "confirm", "Confirm/Pick", show=False),
        Binding("escape", "cancel", "Cancel", show=False),
        Binding("u,delete", "unassign", "Unassign"),
        Binding("left_square_bracket,shift+left", "shift(-1)", "Earlier", show=False),
        Binding("right_square_bracket,shift+right", "shift(1)", "Later", show=False),
        Binding("n", "new_task", "New task"),
        Binding("d", "delete_task", "Delete task"),
        Binding("r", "run", "Run", show=False),
        Binding("tab", "toggle_focus", "Queue⇄Timeline"),
    ]

    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        self.placement: Placement | None = None
        self.preview: PlacementPreview | None = None
        self.last_error: str | None = None

    @property
    def svc(self):
        return self.app.assignment_service

    @property
    def store(self):
        return self.app.store

    def compose(self) -> ComposeResult:
        yield ViewTabs("schedule", id="view-tabs")
        queue = TaskQueue(id="task-queue")
        queue.border_title = "Task Queue"
        yield queue
        details = TaskDetails(id="task-details")
        details.border_title = "Task Details"
        yield details
        timeline = GPUTimeline(self.app.gpus, card_width=self.app.card_width, id="gpu-timeline")
        timeline.border_title = (
            f"GPU Timeline · {len(self.app.gpus)} GPU(s) · {self.app.gpu_source}"
            "  ·  history to scale ┃ now ┃ queue"
        )
        yield timeline
        yield Static("", id="status-line")
        yield Footer()

    # wiring ----------------------------------------------------------------
    @property
    def queue(self) -> TaskQueue:
        return self.query_one(TaskQueue)

    @property
    def timeline(self) -> GPUTimeline:
        return self.query_one(GPUTimeline)

    def on_mount(self) -> None:
        self.store.subscribe(self.refresh_all)
        self.refresh_all()
        self.queue.focus()

    def on_unmount(self) -> None:
        self.store.unsubscribe(self.refresh_all)

    def on_screen_resume(self) -> None:
        self.refresh_all()

    def refresh_all(self) -> None:
        if not self.is_mounted:
            return
        self.queue.set_tasks(self.store.ordered_tasks())
        self.refresh_timeline()
        self.refresh_details()

    @property
    def n_running(self) -> int:
        return len(self.svc.running())

    def refresh_timeline(self) -> None:
        now = datetime.now()
        tl = self.timeline
        self.preview = self.compute_preview(now)
        if self.preview is not None:
            slots = self.preview.slots
            ok, note = self.preview.result.ok, self.preview.result.error
        else:
            slots, ok, note = self.svc.schedule(now), True, None
        tl.update_data(
            slots,
            list(self.store.assignments.values()),
            self.store.tasks,
            now,
            ghost_ok=ok,
            ghost_note=note,
            show_cursor=self.placement is None and self.focused is tl,
        )
        self.refresh_status()

    def selected_task(self) -> Task | None:
        tid = self.queue.selected_id
        return self.store.tasks.get(tid) if tid else None

    def refresh_details(self) -> None:
        details = self.query_one(TaskDetails)
        task = self.selected_task()
        if self.focused is self.timeline and self.placement is None:
            slot = self.timeline.slot_under_cursor()
            if slot is not None:
                task = self.store.tasks.get(slot.task_id)
        note = None
        if self.preview and not self.preview.result.ok:
            note = Text(f"✗ {self.preview.result.message}", style="bold red")
        elif self.last_error:
            note = Text(f"✗ {self.last_error}", style="red")
        details.show(task, self.store.assignments_for_task(task.id) if task else [], note)

    def refresh_status(self) -> None:
        st = self.timeline.state
        msg = Text()
        p = self.preview
        if self.placement and p:
            task = self.store.tasks.get(self.placement.task_id)
            ok = p.result.ok
            msg.append(" PLACING ", style="bold #0b0f19 on #22c55e" if ok else "bold white on #ef4444")
            gpus = ",".join(map(str, p.gpu_ids))
            msg.append(f" {task.name if task else '?'} → GPU {gpus}, queue #{p.position + 1}")
            if p.ghost and ok:
                msg.append(f", starts ~{fmt_when(p.ghost.start, st.now)}")
            msg.append("   ←/→ position  ↑/↓ GPU  Enter confirm  Esc cancel", style="dim")
            if not ok:
                msg.append(f"   ✗ {p.result.message}", style="red")
        else:
            msg.append(
                " Tab queue⇄timeline · a assign · Enter pick up · [ ] reorder · u unassign · "
                "h/l +/- history · drag cards/blocks with the mouse",
                style="dim",
            )
        self.query_one("#status-line", Static).update(msg)

    # placement -------------------------------------------------------------
    def _base_len(self) -> int:
        moving = self.placement.assignment_id if self.placement else None
        return len([a for a in self.svc.planned() if a.id != moving])

    def compute_preview(self, now: datetime) -> PlacementPreview | None:
        p = self.placement
        if not p:
            return None
        task = self.store.tasks.get(p.task_id)
        if task is None:
            return None
        st = self.timeline.state
        n_run = self.n_running
        pos = max(0, min(self._base_len(), st.cursor_col - n_run))
        st.cursor_col = n_run + pos
        gpus = self.svc.suggest_gpus(task, st.cursor_gpu)
        res = self.svc.check(task.id, gpus, assignment_id=p.assignment_id)
        slots, ghost = self.svc.preview(task.id, gpus, pos, p.assignment_id, now)
        return PlacementPreview(gpus, pos, res, ghost, slots)

    def start_placement(self, task_id: str, assignment_id: str | None = None,
                        via_mouse: bool = False) -> None:
        task = self.store.tasks.get(task_id)
        if task is None:
            return
        if task.status in (TaskStatus.RUNNING, TaskStatus.COMPLETED):
            self.notify(f"{task.name} is {task.status.value}; it can't be re-planned.",
                        severity="warning")
            return
        if assignment_id is None:
            existing = self.store.active_assignment(task_id)
            if existing is not None and existing.state == AssignmentState.PLANNED:
                assignment_id = existing.id
        from_queue = self.focused is not self.timeline
        self.placement = Placement(task_id, assignment_id, via_mouse, from_queue)
        self.last_error = None
        st = self.timeline.state
        planned = self.svc.planned()
        if via_mouse:
            # park the ghost at the end so it doesn't shift the columns under the pointer
            st.cursor_col = self.n_running + len(planned)
        else:
            if assignment_id:
                a = self.store.assignments[assignment_id]
                st.cursor_gpu = a.gpu_ids[0]
                idx = next(i for i, x in enumerate(planned) if x.id == assignment_id)
                st.set_cursor_col(self.n_running + idx, self.n_running + len(planned))
            else:
                st.set_cursor_col(self.n_running + len(planned), self.n_running + len(planned))
            self.timeline.focus()
        self.refresh_timeline()
        self.refresh_details()

    def commit_placement(self) -> bool:
        if not self.placement:
            return False
        p = self.placement
        pv = self.compute_preview(datetime.now())
        if pv is None or not pv.result.ok:
            msg = pv.result.message if pv else "nothing to place"
            self.last_error = msg
            self.notify(f"Cannot place: {msg}", severity="error", title="Invalid placement")
            self.refresh_timeline()
            self.refresh_details()
            return False
        if p.assignment_id:
            out = self.svc.move(p.assignment_id, pv.gpu_ids, pv.position)
        else:
            out = self.svc.assign(p.task_id, pv.gpu_ids, pv.position)
        if not out.ok:
            self.last_error = out.message
            self.notify(out.message, severity="error")
            return False
        self._end_placement()
        self.last_error = None
        a = out.assignment
        task = self.store.tasks[p.task_id]
        self.notify(
            f"{task.name} → GPU {','.join(map(str, a.gpu_ids))}, "
            f"starts ~{fmt_when(a.planned_start, datetime.now())}",
            title="Planned",
        )
        self.refresh_all()
        return True

    def _end_placement(self) -> None:
        p, self.placement = self.placement, None
        if p and p.from_queue and not p.via_mouse:
            self.queue.focus()

    def cancel_placement(self) -> None:
        self._end_placement()
        self.refresh_timeline()
        self.refresh_details()

    # keyboard actions ------------------------------------------------------
    def _planned_slot_under_cursor(self) -> Slot | None:
        slot = self.timeline.slot_under_cursor()
        return slot if slot is not None and slot.kind == "planned" else None

    def action_place(self) -> None:
        if self.focused is self.timeline:
            slot = self._planned_slot_under_cursor()
            if slot is not None:
                self.start_placement(slot.task_id, slot.key)
                return
        task = self.selected_task()
        if task:
            self.start_placement(task.id)

    def action_confirm(self) -> None:
        if self.placement:
            self.commit_placement()
        elif self.focused is self.timeline:
            slot = self._planned_slot_under_cursor()
            if slot is not None:
                self.queue.select(slot.task_id)
                self.start_placement(slot.task_id, slot.key)
        else:
            self.action_place()

    def action_cancel(self) -> None:
        if self.placement:
            self.cancel_placement()

    def action_unassign(self) -> None:
        if self.focused is self.timeline:
            slot = self.timeline.slot_under_cursor()
            a = self.store.assignments.get(slot.key) if slot else None
        else:
            task = self.selected_task()
            a = self.store.active_assignment(task.id) if task else None
        if a is None:
            self.notify("Nothing to unassign here.", severity="warning")
            return
        res = self.svc.unassign(a.id)
        if not res.ok:
            self.notify(res.message, severity="error")

    def action_shift(self, delta: int) -> None:
        if self.placement or self.focused is not self.timeline:
            return
        slot = self._planned_slot_under_cursor()
        if slot is None:
            return
        res = self.svc.shift(slot.key, delta)
        if res.ok:
            st = self.timeline.state
            idx = next(i for i, a in enumerate(self.svc.planned()) if a.id == slot.key)
            st.set_cursor_col(self.n_running + idx, self.timeline.n_items)
            self.refresh_timeline()
            self.refresh_details()

    def action_new_task(self) -> None:
        def done(task: Task | None) -> None:
            if task:
                self.store.add_task(task)
                self.queue.select(task.id)

        self.app.push_screen(new_task_form(), done)

    def action_delete_task(self) -> None:
        task = self.selected_task()
        if task is None:
            return
        if task.status == TaskStatus.RUNNING:
            self.notify("Cannot delete a running task.", severity="error")
            return
        self.store.remove_task(task.id)
        self.svc.recompute()
        self.notify(f"Deleted {task.name}")

    def action_toggle_focus(self) -> None:
        if self.focused is self.timeline:
            self.queue.focus()
        else:
            self.timeline.focus()

    def action_run(self) -> None:
        self.app.run_selected(self.selected_task())

    def check_action(self, action: str, parameters) -> bool | None:
        if action == "run":
            return self.app.runtime is not None
        return True

    # events ----------------------------------------------------------------
    def on_task_queue_selection_changed(self, event: TaskQueue.SelectionChanged) -> None:
        self.refresh_details()

    def on_gputimeline_cursor_moved(self, event: GPUTimeline.CursorMoved) -> None:
        self.refresh_timeline()
        self.refresh_details()

    def on_descendant_focus(self, event) -> None:
        self.refresh_timeline()
        self.refresh_details()

    # mouse drag ------------------------------------------------------------
    def _drag_to(self, x: int, y: int) -> bool:
        """Point the placement at the column under (x, y); False outside the queue."""
        loc = self.timeline.locate(x, y)
        if loc is None:
            return False
        gpu, hit = loc
        if hit.zone != "future" or hit.col is None:
            return False
        st = self.timeline.state
        n_run = self.n_running
        ghost_col = n_run + self.preview.position if self.preview else None
        col = hit.col
        if ghost_col is not None and col == ghost_col:
            insert = ghost_col
        else:
            base = col if ghost_col is None or col < ghost_col else col - 1
            insert = base + (1 if hit.after else 0)
        st.cursor_gpu = gpu
        st.cursor_col = max(n_run, min(n_run + self._base_len(), insert))
        self.refresh_timeline()
        self.refresh_details()
        return True

    def on_task_card_drag_move(self, event: TaskCard.DragMove) -> None:
        if not self.placement or self.placement.task_id != event.task_id:
            self.start_placement(event.task_id, via_mouse=True)
        self._drag_to(event.x, event.y)

    def on_task_card_drag_end(self, event: TaskCard.DragEnd) -> None:
        self._finish_drag(event.x, event.y)

    def on_gpurow_block_pressed(self, event: GPURow.BlockPressed) -> None:
        self.timeline.focus()
        loc = self.timeline.locate(event.x, event.y)
        if loc:
            gpu, hit = loc
            self.timeline.set_cursor(gpu, hit.col if hit.zone == "future" else None)
        if event.assignment_id:
            a = self.store.assignments.get(event.assignment_id)
            if a:
                self.queue.select(a.task_id)
        self.refresh_details()

    def on_gpurow_block_drag_move(self, event: GPURow.BlockDragMove) -> None:
        a = self.store.assignments.get(event.assignment_id)
        if a is None:
            return
        if not self.placement or self.placement.assignment_id != a.id:
            self.start_placement(a.task_id, a.id, via_mouse=True)
        self._drag_to(event.x, event.y)

    def on_gpurow_block_drag_end(self, event: GPURow.BlockDragEnd) -> None:
        self._finish_drag(event.x, event.y)

    def _finish_drag(self, x: int, y: int) -> None:
        if not self.placement:
            return
        if self._drag_to(x, y):
            self.commit_placement()
            if self.placement:  # refused: drop the ghost
                self.placement = None
                self.refresh_timeline()
        else:
            self.cancel_placement()
