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
from ..widgets.dialogs import new_task_form
from ..widgets.gpu_row import Ghost, GPURow
from ..widgets.gpu_timeline import GPUTimeline
from ..widgets.task_card import TaskCard
from ..widgets.task_details import TaskDetails
from ..widgets.task_queue import TaskQueue
from ..widgets.timeline_state import ceil_time
from ..widgets.view_tabs import ViewTabs


@dataclass
class Placement:
    """A placement in progress (keyboard 'a' mode or a mouse drag)."""

    task_id: str
    assignment_id: str | None = None  # moving an existing block
    via_mouse: bool = False


class ScheduleScreen(Screen):
    BINDINGS = [
        Binding("a", "place", "Assign"),
        Binding("enter", "confirm", "Confirm/Pick", show=False),
        Binding("escape", "cancel", "Cancel", show=False),
        Binding("u,delete", "unassign", "Unassign"),
        Binding("n", "new_task", "New task"),
        Binding("d", "delete_task", "Delete task"),
        Binding("r", "run", "Run", show=False),
        Binding("tab", "toggle_focus", "Queue⇄Timeline"),
    ]

    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        self.placement: Placement | None = None
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
        timeline = GPUTimeline(self.app.gpus, id="gpu-timeline")
        timeline.border_title = f"GPU Timeline · {len(self.app.gpus)} GPU(s) · {self.app.gpu_source}"
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

    def refresh_timeline(self) -> None:
        self.timeline.update_data(
            list(self.store.assignments.values()), self.store.tasks, self.current_ghost()
        )
        self.refresh_status()

    def selected_task(self) -> Task | None:
        tid = self.queue.selected_id
        return self.store.tasks.get(tid) if tid else None

    def refresh_details(self) -> None:
        details = self.query_one(TaskDetails)
        task = self.selected_task()
        if self.focused is self.timeline and self.placement is None:
            a = self.timeline.assignment_under_cursor()
            if a is not None:
                task = self.store.tasks.get(a.task_id)
        note = None
        if self.placement and self.placement_result() and not self.placement_result().ok:
            note = Text(f"✗ {self.placement_result().message}", style="bold red")
        elif self.last_error:
            note = Text(f"✗ {self.last_error}", style="red")
        details.show(task, self.store.assignments_for_task(task.id) if task else [], note)

    def refresh_status(self) -> None:
        st = self.timeline.state
        if self.placement:
            task = self.store.tasks.get(self.placement.task_id)
            res = self.placement_result()
            ok = res is not None and res.ok
            gpus = ",".join(map(str, res.assignment.gpu_ids)) if res and res.assignment else "?"
            msg = Text()
            msg.append(" PLACING ", style="bold #0b0f19 on #22c55e" if ok else "bold white on #ef4444")
            msg.append(f" {task.name if task else '?'} → GPU {gpus} @ {st.cursor_time:%m-%d %H:%M}  ")
            msg.append("←/→ time  ↑/↓ GPU  Enter confirm  Esc cancel", style="dim")
            if res and not res.ok:
                msg.append(f"   ✗ {res.message}", style="red")
        else:
            msg = Text(
                f" cursor GPU{st.cursor_gpu} {st.cursor_time:%m-%d %H:%M}  ·  "
                "Tab: queue⇄timeline · a: assign · Enter on block: move · drag cards/blocks with the mouse",
                style="dim",
            )
        self.query_one("#status-line", Static).update(msg)

    # placement -------------------------------------------------------------
    def placement_result(self):
        if not self.placement:
            return None
        task = self.store.tasks.get(self.placement.task_id)
        if task is None:
            return None
        st = self.timeline.state
        gpus = self.svc.suggest_gpus(task, st.cursor_gpu, st.cursor_time)
        aid = self.placement.assignment_id or (
            a.id if (a := self.store.active_assignment(task.id)) else None
        )
        return self.svc.check(task.id, gpus, st.cursor_time, assignment_id=aid)

    def current_ghost(self) -> Ghost | None:
        res = self.placement_result()
        if res is None or res.assignment is None:
            return None
        a = res.assignment
        task = self.store.tasks[a.task_id]
        return Ghost(task.id, task.name, a.gpu_ids, a.planned_start, a.planned_end, res.ok,
                     self.placement.assignment_id if self.placement else None)

    def start_placement(self, task_id: str, assignment_id: str | None = None,
                        via_mouse: bool = False) -> None:
        task = self.store.tasks.get(task_id)
        if task is None:
            return
        if task.status in (TaskStatus.RUNNING, TaskStatus.COMPLETED):
            self.notify(f"{task.name} is {task.status.value}; it can't be re-planned.",
                        severity="warning")
            return
        self.placement = Placement(task_id, assignment_id, via_mouse)
        self.last_error = None
        st = self.timeline.state
        if not via_mouse:
            existing = self.store.assignments.get(assignment_id) if assignment_id else \
                self.store.active_assignment(task_id)
            if existing:
                st.cursor_gpu = existing.gpu_ids[0]
                st.set_cursor_time(existing.planned_start)
            else:
                start = ceil_time(max(datetime.now(), st.cursor_time), st.minutes_per_cell)
                gpus = self.svc.suggest_gpus(task, st.cursor_gpu, start)
                # round up: snapping down could land inside the block that ends there
                slot = self.svc.next_free_slot(gpus, start, task.estimated_duration)
                st.set_cursor_time(slot, round_up=True)
            self.timeline.focus()
        self.refresh_timeline()
        self.refresh_details()

    def commit_placement(self) -> bool:
        if not self.placement:
            return False
        res = self.placement_result()
        p = self.placement
        if res is None or not res.ok or res.assignment is None:
            msg = res.message if res else "nothing to place"
            self.last_error = msg
            self.notify(f"Cannot place: {msg}", severity="error", title="Conflict")
            self.refresh_timeline()
            self.refresh_details()
            return False
        a = res.assignment
        if p.assignment_id:
            out = self.svc.move(p.assignment_id, a.gpu_ids, a.planned_start)
        else:
            out = self.svc.assign(p.task_id, a.gpu_ids, a.planned_start)
        if not out.ok:
            self.notify(out.message, severity="error")
            return False
        self.placement = None
        self.last_error = None
        task = self.store.tasks[p.task_id]
        self.notify(f"{task.name} → GPU {','.join(map(str, a.gpu_ids))} @ {a.planned_start:%H:%M}",
                    title="Planned")
        self.refresh_all()
        return True

    def cancel_placement(self) -> None:
        self.placement = None
        self.refresh_timeline()
        self.refresh_details()

    # keyboard actions ------------------------------------------------------
    def action_place(self) -> None:
        if self.focused is self.timeline:
            a = self.timeline.assignment_under_cursor()
            if a is not None and a.is_active:
                self.start_placement(a.task_id, a.id)
                return
        task = self.selected_task()
        if task:
            self.start_placement(task.id)

    def action_confirm(self) -> None:
        if self.placement:
            self.commit_placement()
            return
        if self.focused is self.timeline:
            a = self.timeline.assignment_under_cursor()
            if a is not None and a.state == AssignmentState.PLANNED:
                self.queue.select(a.task_id)
                self.start_placement(a.task_id, a.id)
        else:
            self.action_place()

    def action_cancel(self) -> None:
        if self.placement:
            self.cancel_placement()

    def action_unassign(self) -> None:
        a = None
        if self.focused is self.timeline:
            a = self.timeline.assignment_under_cursor()
        else:
            task = self.selected_task()
            a = self.store.active_assignment(task.id) if task else None
        if a is None:
            self.notify("Nothing to unassign here.", severity="warning")
            return
        res = self.svc.unassign(a.id)
        if not res.ok:
            self.notify(res.message, severity="error")

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
        self.notify(f"Deleted {task.name}")

    def action_toggle_focus(self) -> None:
        if self.focused is self.timeline:
            self.queue.focus()
        else:
            self.timeline.focus()
        self.refresh_details()

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
        if self.placement:
            self.timeline.update_data(
                list(self.store.assignments.values()), self.store.tasks, self.current_ghost()
            )
        self.refresh_status()
        self.refresh_details()

    def on_descendant_focus(self, event) -> None:
        self.refresh_details()

    # mouse drag: cards -----------------------------------------------------
    def _drag_to(self, x: int, y: int) -> bool:
        loc = self.timeline.locate(x, y)
        if loc is None:
            return False
        gpu, t = loc
        st = self.timeline.state
        st.cursor_gpu = gpu
        st.cursor_time = t
        self.refresh_timeline()
        self.refresh_details()
        return True

    def on_task_card_drag_move(self, event: TaskCard.DragMove) -> None:
        if not self.placement or self.placement.task_id != event.task_id:
            self.start_placement(event.task_id, via_mouse=True)
        self._drag_to(event.x, event.y)

    def on_task_card_drag_end(self, event: TaskCard.DragEnd) -> None:
        self._finish_drag(event.x, event.y)

    # mouse drag: existing blocks -------------------------------------------
    def on_gpurow_block_pressed(self, event: GPURow.BlockPressed) -> None:
        self.timeline.focus()
        loc = self.timeline.locate(event.x, event.y)
        if loc:
            self.timeline.set_cursor(*loc)
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
            if self.placement:  # commit refused (conflict): drop the ghost
                self.placement = None
                self.refresh_timeline()
        else:
            self.cancel_placement()
