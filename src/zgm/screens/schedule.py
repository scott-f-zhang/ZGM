"""Schedule view, top to bottom: this machine's CPU, memory and disk, with, in
cluster mode, the groups of GPUs (e.g. Slurm jobs) beside them; the task queue;
the GPU timeline; the selected task's details. Tab goes through the panels in
that order, and each is focused with its first letter and can be shown full
screen (f); the full-screen queue lists the waiting tasks by section (auto-fill /
manual) and config folder."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.screen import Screen
from textual.widgets import Footer, Static

from ..models import AssignmentState, Task, TaskStatus
from ..services.lease import describe
from ..widgets import theme
from ..widgets.theme import badge, box_footer, box_title, hints
from ..services.scheduler import GHOST, Result, Slot
from ..widgets.dialogs import ChoiceScreen, apply_task_edit, edit_task_form, new_task_form
from ..widgets.gpu_row import GPURow
from ..widgets.gpu_timeline import GPUTimeline, pack_columns, queue_position, slot_columns
from ..widgets.cluster_panel import ClusterPanel
from ..widgets.host_panel import HostPanel
from ..widgets.task_card import TaskCard
from ..widgets.task_details import TaskDetails
from ..widgets.queue_overview import QueueOverview
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
        Binding("a", "place", "assign"),
        Binding("enter", "confirm", "confirm/pick", show=False),
        Binding("escape", "cancel", "cancel", show=False),
        Binding("u,delete", "unassign", "unassign", show=False),
        Binding("left_square_bracket,shift+left", "shift(-1)", "earlier", show=False),
        Binding("right_square_bracket,shift+right", "shift(1)", "later", show=False),
        Binding("n", "new_task", "new task"),
        Binding("e", "edit_task", "edit/rename", show=False),
        Binding("x", "delete_task", "delete task", show=False),
        Binding("r", "run", "run now", show=False),
        Binding("p", "toggle_autostart", "pause/resume auto-start", show=False),
        Binding("K", "stop", "stop"),
        Binding("tab", "cycle_focus(1)", "next panel"),
        Binding("shift+tab", "cycle_focus(-1)", "previous panel", show=False),
        Binding("f", "toggle_zoom", "full screen"),
        Binding("q", "focus_panel('queue')", "queue", show=False),
        Binding("d", "focus_panel('details')", "details", show=False),
        Binding("t", "focus_panel('timeline')", "timeline", show=False),
        Binding("s", "focus_panel('cluster')", "cluster", show=False),
    ]
    # panel -> its widget; each is focused with the first letter of its title
    # (Tab goes through them in this order)
    PANELS = {"host": "#host-panel", "cluster": "#cluster-panel", "queue": "#task-queue",
              "timeline": "#gpu-timeline", "details": "#task-details"}

    TOP_LINES = 7  # content lines of the host and cluster boxes, at most

    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        self.placement: Placement | None = None
        self.preview: PlacementPreview | None = None
        self.last_error: str | None = None
        self.zoomed: str | None = None  # the panel shown full screen
        self.subject = "queue"  # what details and actions follow: "queue" or "timeline"

    @property
    def svc(self):
        return self.app.assignment_service

    @property
    def store(self):
        return self.app.store

    def compose(self) -> ComposeResult:
        yield ViewTabs("schedule", id="view-tabs")
        path = self.store.storage.path
        with Horizontal(id="top-row"):
            yield HostPanel(None if path == ":memory:" else str(Path(path).parent), id="host-panel")
            yield ClusterPanel(id="cluster-panel")
        queue = TaskQueue(id="task-queue")
        queue.border_title = box_title("queue", key="q")
        yield queue
        overview = QueueOverview(id="queue-overview")
        overview.border_title = box_title("queue", key="q", info="waiting tasks by config folder")
        overview.border_subtitle = box_footer(
            "space auto-fill ⇄ manual · A whole folder · +/- priority · a assign · f back")
        overview.display = False
        yield overview
        timeline = GPUTimeline(self.app.gpus, card_width=self.app.card_width, id="gpu-timeline")
        timeline.border_subtitle = box_footer("history to scale ┃ now ┃ queue in order")
        yield timeline
        details = TaskDetails(id="task-details")
        details.border_title = box_title("details", key="d")
        yield details
        yield Static("", id="status-line")
        yield Footer()

    # wiring ----------------------------------------------------------------
    @property
    def queue(self) -> TaskQueue:
        return self.query_one(TaskQueue)

    @property
    def timeline(self) -> GPUTimeline:
        return self.query_one(GPUTimeline)

    @property
    def overview(self) -> QueueOverview:
        return self.query_one(QueueOverview)

    def on_mount(self) -> None:
        self.store.subscribe(self.refresh_all)
        self.retitle_timeline()
        self.update_timeline_footer()
        self.refresh_all()
        self.refresh_cluster()
        self.queue.focus()

    def update_timeline_footer(self) -> None:
        if self.timeline.state.is_tasks:
            text = "task view ┃ recent 200 runs ┃ t: time view"
        else:
            text = "time view ┃ past + estimates to scale ┃ t: task view"
        self.timeline.border_subtitle = box_footer(text)

    def retitle_timeline(self) -> None:
        """``N gpus · M idle``: idle = nothing of zgm's and no other program on it."""
        ids = {g.id for g in self.app.gpus}
        busy = {g for a in self.svc.running() for g in a.display_gpu_ids} | self.svc.blocked
        n = len(ids)
        title = box_title(
            "timeline", key="t",
            info=f"{n} gpu{'s' if n != 1 else ''} · {len(ids - busy)} idle · {self.app.gpu_source}")
        if getattr(self.timeline, "_title", None) != title:  # setting it repaints the timeline
            self.timeline.border_title = self.timeline._title = title

    def gpus_changed(self) -> None:
        """Cluster mode: a group's GPUs came or went."""
        if not self.is_mounted:
            return
        self.timeline.set_gpus(self.app.gpus)
        self.retitle_timeline()
        self.refresh_all()

    def refresh_cluster(self) -> None:
        """The top row: this machine's load, and the cluster's groups."""
        self.query_one(HostPanel).sample()
        cluster = self.app.cluster
        panels = self.query(ClusterPanel)
        if cluster is not None and panels:
            panel = panels.first()
            panel.hidden_by_zoom = self.zoomed not in (None, "cluster")
            panel.show(cluster)
        self.fit_top_row()

    def fit_top_row(self) -> None:
        """Host and cluster boxes side by side, half the width each, both as
        tall as the taller one's content (at most TOP_LINES lines)."""
        row = self.query_one("#top-row")
        if self.zoomed in ("host", "cluster"):
            row.styles.height = "1fr"
            return
        cluster = self.query_one(ClusterPanel)
        lines = max(self.query_one(HostPanel).lines, cluster.lines if cluster.display else 0)
        row.styles.height = min(lines, self.TOP_LINES) + 2  # + the border

    def on_unmount(self) -> None:
        self.store.unsubscribe(self.refresh_all)

    def on_screen_resume(self) -> None:
        self.refresh_all()

    def refresh_all(self) -> None:
        if not self.is_mounted or self.app.idle:
            return
        self.queue.set_tasks(self.store.waiting_tasks())
        if self.zoomed == "queue":
            self.refresh_overview()
        self.refresh_timeline()
        self.refresh_details()

    @property
    def run_cols(self) -> int:
        """Columns of running work on the timeline: queued work starts after them."""
        return max(pack_columns([a.display_gpu_ids for a in self.svc.running()]), default=-1) + 1

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
        self.retitle_timeline()
        self.refresh_status()

    def selected_task(self) -> Task | None:
        tid = self.queue.selected_id
        return self.store.tasks.get(tid) if tid else None

    def current_task(self) -> Task | None:
        """The task actions apply to: the card under the timeline cursor when the
        timeline has focus, otherwise the selected queue card."""
        if self.subject == "timeline":
            slot = self.timeline.slot_under_cursor()
            return self.store.tasks.get(slot.task_id) if slot else None
        return self.selected_task()

    def refresh_details(self) -> None:
        details = self.query_one(TaskDetails)
        if self.placement is not None:
            task = self.store.tasks.get(self.placement.task_id)
        else:
            task = self.current_task()
        note = None
        if self.preview and not self.preview.result.ok:
            note = Text(f"✗ {self.preview.result.message}", style="bold red")
        elif self.last_error:
            note = Text(f"✗ {self.last_error}", style="red")
        details.show(task, self.store.assignments_for_task(task.id) if task else [], note,
                     log=self.app.log_path(task) if task else None,
                     blocked=self.svc.blocked)

    def refresh_status(self) -> None:
        msg = Text()
        p = self.preview
        if self.app.idle:
            msg.append(" IDLE ", style=badge(theme.INACTIVE_FG, theme.MAIN_FG))
            msg.append(f" view paused after {self.app.idle_pause / 60:g} min without input; "
                       "any key resumes (the queue keeps running)", style=theme.SUBTLE)
        elif self.placement and p:
            task = self.store.tasks.get(self.placement.task_id)
            ok = p.result.ok
            msg.append(" PLACING ", style=badge(theme.OK if ok else theme.BAD))
            gpus = ",".join(map(str, p.gpu_ids))
            n = self.queue_number(GHOST, p.slots)
            msg.append(f" {task.name if task else '?'} → GPU {gpus}, queue #{n}  ",
                       style=theme.MAIN_FG)
            msg.append_text(hints([("←/→", "position"), ("↑/↓", "gpu"), ("Enter", "confirm"),
                                   ("Esc", "cancel")]))
            if not ok:
                msg.append(f"   ✗ {p.result.message}", style=theme.BAD)
        else:
            info = self.app.executor_info()
            if info is not None:
                if self.app.autostart:
                    msg.append(" AUTO ▶ ", style=badge(theme.OK))
                    if self.app.executor.autofill:
                        msg.append(" FILL ", style=badge(theme.OK))
                else:
                    msg.append(" PAUSED ‖ ", style=badge(theme.WARN))
                who = (f"{info.get('runtime')} runtime" if self.app.is_executor
                       else f"run by {describe(info)}")
                msg.append(f" {who}", style=theme.SUBTLE)
            else:
                msg.append(" PLAN ONLY ", style=badge(theme.INACTIVE_FG, theme.MAIN_FG))
                msg.append(" nothing runs: zgm daemon start", style=theme.SUBTLE)
            msg.append("   ")
            msg.append_text(hints([("Esc", "menu"), ("3", "all keys")]))
        if msg != getattr(self, "_status", None):  # refreshed on every tick
            self._status = msg
            self.query_one("#status-line", Static).update(msg)

    # placement -------------------------------------------------------------
    def _base_len(self) -> int:
        moving = self.placement.assignment_id if self.placement else None
        return len([a for a in self.svc.planned() if a.id != moving])

    def compute_preview(self, now: datetime) -> PlacementPreview | None:
        """The placement where the cursor points: a queue position in the card
        view, or the nearest insertion point to the cursor's time in the time
        view."""
        p = self.placement
        if not p:
            return None
        task = self.store.tasks.get(p.task_id)
        if task is None:
            return None
        st = self.timeline.state
        if st.is_tasks:
            future_col = st.cursor_col - st.now_col
            if future_col < 0:
                return None
            base = [s for s in self.svc.schedule(now) if s.key != p.assignment_id]
            pos, _ = queue_position(base, st.cursor_gpu, future_col)
        else:
            base = [a for a in self.svc.planned() if a.id != p.assignment_id]
            if st.cursor_zone == "tail":
                slot = self.timeline.slot_under_cursor()
                if slot is None:
                    pos = len(base)
                elif slot.key == GHOST:
                    pos = self.preview.position if self.preview else len(base)
                else:
                    keys = [a.id for a in base]
                    at = keys.index(slot.key) if slot.key in keys else len(base)
                    pos = at + (1 if st.cursor_after else 0)
            else:
                when = st.cursor_time
                if when < now - st.cell:
                    return None
                when = max(when, now).replace(second=0, microsecond=0)
                pos = self._time_position(base, st.cursor_gpu, when)
        gpus = self.svc.suggest_gpus(task, st.cursor_gpu)
        res = self.svc.check(task.id, gpus, assignment_id=p.assignment_id)
        slots, ghost = self.svc.preview(task.id, gpus, pos, p.assignment_id, now)
        if ghost is not None and st.is_tasks:
            # follow the ghost's column: the card view has no separate drop time
            cols, _ = slot_columns(slots)
            st.cursor_col = st.now_col + next(c for s, c in zip(slots, cols)
                                              if s.key == GHOST)
        return PlacementPreview(gpus, pos, res, ghost, slots)

    @staticmethod
    def _time_position(planned, gpu: int, when: datetime) -> int:
        """Index in ``planned`` where a task dropped at ``when`` on ``gpu`` goes."""
        when = when.replace(second=0, microsecond=0)
        here = [(i, a) for i, a in enumerate(planned) if gpu in a.gpu_ids]
        for i, a in here:
            if a.planned_start is not None and a.planned_start >= when:
                return i
        for i, a in here:
            if a.planned_start is None:
                return i
        return here[-1][0] + 1 if here else len(planned)

    def queue_number(self, assignment_id: str, slots=None) -> int | None:
        """A queued assignment's place on its GPUs (1 = next), as the timeline numbers it."""
        slots = slots if slots is not None else self.svc.schedule()
        cols, n_run = slot_columns(slots)
        return next((c - n_run + 1 for s, c in zip(slots, cols) if s.key == assignment_id), None)

    def column_of(self, key: str) -> int | None:
        return self.timeline.column_of(key)

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
        if self.zoomed:
            self.set_zoom(None)  # placing needs the timeline
        self.placement = Placement(task_id, assignment_id, via_mouse, from_queue)
        self.last_error = None
        st = self.timeline.state
        st.cursor_after = False
        if st.is_tasks:
            st.cursor_zone = "cards"
            if assignment_id:
                a = self.store.assignments[assignment_id]
                st.cursor_gpu = a.gpu_ids[0]
                col = self.column_of(assignment_id)
                st.cursor_col = col if col is not None else self.timeline.n_items
            else:  # the first GPU, the first column after now
                if self.app.gpus:
                    st.cursor_gpu = self.app.gpus[0].id
                st.cursor_col = st.now_col
        elif assignment_id:
            a = self.store.assignments[assignment_id]
            st.cursor_gpu = a.gpu_ids[0]
            if a.planned_start is not None:
                st.cursor_zone = "time"
                st.set_cursor_x(st.time_cell(a.planned_start))
            else:
                st.cursor_zone = "tail"
                col = self.column_of(assignment_id)
                st.cursor_col = col if col is not None else self.timeline.n_items
        else:
            # a new placement starts on the first GPU, right after now (behind what
            # runs there, before what is queued there), in either view
            if self.app.gpus:
                st.cursor_gpu = self.app.gpus[0].id
            st.cursor_zone = "time"
            st.set_cursor_x(st.now_x)
        if not via_mouse:
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
        self.notify(f"{task.name} → GPU {','.join(map(str, a.gpu_ids))}, "
                    f"queue #{self.queue_number(a.id)}",
                    title="Planned")
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
        """Esc: cancel the placement, else leave full screen, else the menu."""
        if self.placement:
            self.cancel_placement()
        elif self.zoomed:
            self.set_zoom(None)
        else:
            self.app.action_menu()

    # panels: focus and full screen -------------------------------------------
    def panel_widget(self, name: str):
        if name == "queue" and self.zoomed == "queue":
            return self.overview
        return self.query_one(self.PANELS[name])

    def focused_panel(self) -> str | None:
        node = self.focused
        while node is not None and node is not self:
            if node is self.overview:
                return "queue"
            for name, sel in self.PANELS.items():
                if node.id == sel[1:]:
                    return name
            node = node.parent
        return None

    def panel_available(self, name: str) -> bool:
        return name != "cluster" or self.query_one(ClusterPanel).has_groups

    def action_focus_panel(self, name: str) -> None:
        if self.placement or not self.panel_available(name):
            return
        if self.zoomed and self.zoomed != name:
            self.set_zoom(name)  # full screen follows the focus
        self.panel_widget(name).focus()

    def action_toggle_zoom(self) -> None:
        if self.placement:
            return
        name = self.focused_panel() or "queue"
        self.set_zoom(None if self.zoomed == name else name)

    def set_zoom(self, name: str | None) -> None:
        """Show ``name`` alone, over the whole view (None: every panel again)."""
        self.zoomed = name
        for pname, sel in self.PANELS.items():
            w = self.query_one(sel)
            w.set_class(pname == name, "-zoomed")
            if pname == "cluster":
                w.hidden_by_zoom = name not in (None, "cluster")
                w.display = w.has_groups and not w.hidden_by_zoom
            elif pname == "queue":
                w.display = name is None
            else:
                w.display = name in (None, pname)
        self.query_one("#top-row").display = name in (None, "host", "cluster")
        self.fit_top_row()
        self.overview.display = name == "queue"
        if name == "queue":
            self.refresh_overview()
        if name is not None:
            self.panel_widget(name).focus()
        elif self.focused is None or not self.focused.display:
            self.queue.focus()
        self.refresh_all()

    def refresh_overview(self) -> None:
        def last_run(t: Task):
            runs = [a for a in self.store.assignments_for_task(t.id) if a.actual_start]
            return max(runs, key=lambda a: a.actual_start) if runs else None

        self.overview.set_tasks(self.store.waiting_tasks(), self.app.config_repo.root,
                                last_run, selected=self.queue.selected_id)

    def on_queue_overview_selection_changed(self, event: QueueOverview.SelectionChanged) -> None:
        self.queue.select(event.task_id)

    def on_queue_overview_toggle(self, event: QueueOverview.Toggle) -> None:
        for tid in event.task_ids:
            task = self.store.tasks.get(tid)
            if task is not None:
                task.autofill = event.autofill
                self.store.update_task(task, quiet=True)
        self.store.notify()
        where = "auto-fill" if event.autofill else "manual"
        n = len(event.task_ids)
        self.notify(f"{n} task{'s' if n != 1 else ''} → {where}"
                    + ("" if not event.autofill or self.app.executor.autofill
                       else " (auto-fill is off: settings, 3)"))
        self.app.dispatch()

    def on_queue_overview_priority(self, event: QueueOverview.Priority) -> None:
        task = self.store.tasks.get(event.task_id)
        if task is None:
            return
        task.priority += event.delta
        self.store.update_task(task)
        self.notify(f"{task.name}: priority {task.priority:+d}")

    def on_option_list_option_selected(self, event) -> None:
        """Enter on a task in the full-screen queue: place it."""
        if event.option.id:
            self.queue.select(event.option.id)
            self.start_placement(event.option.id)

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
            self.refresh_timeline()
            col = self.column_of(slot.key)
            if col is not None:
                self.timeline.state.set_cursor_col(col, self.timeline.n_items)
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

    def action_cycle_focus(self, step: int) -> None:
        """Tab / Shift+Tab: the next / previous panel (in full screen, it follows)."""
        names = [n for n in self.PANELS if self.panel_available(n)]
        here = self.focused_panel()
        i = names.index(here) if here in names else -step
        self.action_focus_panel(names[(i + step) % len(names)])

    def action_run(self) -> None:
        self.app.run_selected(self.current_task())

    def action_edit_task(self) -> None:
        task = self.current_task()
        if task is None:
            self.notify("Select a task in the queue or on the timeline first.", severity="warning")
            return
        running = task.status == TaskStatus.RUNNING

        def done(values) -> None:
            if not values:
                return
            name, spec, gpus = values
            before = (task.name, task.command, task.launch, task.required_gpus, task.fingerprint)
            apply_task_edit(task, name, spec, gpus)
            a = self.store.active_assignment(task.id)
            if a is not None and a.state == AssignmentState.PLANNED and gpus != len(a.gpu_ids):
                # keep its queue position but give it the new number of GPUs
                res = self.svc.move(a.id, self.svc.suggest_gpus(task, a.gpu_ids[0]))
                if not res.ok:
                    (task.name, task.command, task.launch, task.required_gpus,
                     task.fingerprint) = before
                    self.notify(f"Not saved: {res.message}", severity="error")
                    return
            self.store.update_task(task)
            self.svc.recompute()
            self.notify(f"Saved {task.name}")

        self.app.push_screen(edit_task_form(task, name_only=running), done)

    def action_stop(self) -> None:
        task = self.current_task()
        if task is None or task.status != TaskStatus.RUNNING:
            self.notify("Put the timeline cursor on a running task (▶) to stop it.",
                        severity="warning")
            return

        def done(choice: str | None) -> None:
            if choice not in ("requeue", "delete"):
                return
            self.app.stop_task(task, requeue=choice == "requeue")
            self.notify(f"Stopped {task.name}" + (", back in the queue" if choice == "requeue"
                                                  else " and deleted it"))

        self.app.push_screen(ChoiceScreen(
            f"Stop {task.name}?",
            "The partial run stays in the history (not counted for estimates).",
            [("requeue", "Stop and put it back in the queue", "primary"),
             ("delete", "Stop and delete the task", "error"),
             ("keep", "Keep running", "default")],
        ), done)

    def action_toggle_autostart(self) -> None:
        self.app.set_autostart(not self.app.autostart)
        self.notify("Auto-start on: queued tasks start as their GPUs free up"
                    if self.app.autostart else "Auto-start paused: nothing new starts until p")
        self.refresh_status()

    def check_action(self, action: str, parameters) -> bool | None:
        if action in ("run", "toggle_autostart"):
            return self.app.can_control_runs
        return True

    # events ----------------------------------------------------------------
    def on_task_queue_selection_changed(self, event: TaskQueue.SelectionChanged) -> None:
        self.refresh_details()

    def on_gputimeline_cursor_moved(self, event: GPUTimeline.CursorMoved) -> None:
        self.refresh_timeline()
        self.refresh_details()

    def on_gputimeline_view_changed(self, event: GPUTimeline.ViewChanged) -> None:
        if self.placement:
            self.cancel_placement()
        self.update_timeline_footer()
        self.refresh_timeline()
        self.refresh_details()

    def on_descendant_focus(self, event) -> None:
        if self.focused is self.timeline:
            self.subject = "timeline"
        elif self.focused in (self.queue, self.overview):
            self.subject = "queue"
        self.refresh_timeline()
        self.refresh_details()

    # mouse drag ------------------------------------------------------------
    def _drag_to(self, x: int, y: int) -> bool:
        """Point the placement at the card/time under (x, y); False outside it."""
        loc = self.timeline.locate(x, y)
        if loc is None:
            return False
        gpu, hit = loc
        st = self.timeline.state
        if st.is_tasks:
            if hit.zone != "future" or hit.col is None or hit.col < st.now_col:
                return False
            st.cursor_gpu = gpu
            st.cursor_zone = "cards"
            st.cursor_after = hit.after
            ghost = self.preview.ghost if self.preview else None
            ghost_col = self.column_of(GHOST) if ghost else None
            offset = st.now_col
            ghost_future = None if ghost_col is None else ghost_col - offset
            col = hit.col - offset
            if ghost_future is not None and col == ghost_future and gpu in ghost.gpu_ids:
                insert = ghost_future
            else:
                # on the ghost's rows, the items after it are drawn one column further
                shifted = (ghost_future is not None and gpu in ghost.gpu_ids
                           and col > ghost_future)
                insert = (col - 1 if shifted else col) + (1 if hit.after else 0)
            st.cursor_col = offset + max(self.run_cols, insert)
        elif hit.zone == "past" and hit.x is not None:
            if hit.time is None or hit.time < datetime.now() - st.cell:
                return False  # history is not a placement target
            st.cursor_gpu = gpu
            st.cursor_zone = "time"
            st.cursor_after = False
            st.set_cursor_x(hit.x)
        elif hit.zone == "future" and hit.col is not None:
            st.cursor_gpu = gpu
            st.cursor_zone = "tail"
            st.cursor_col = min(hit.col, self.timeline.n_items)
            st.cursor_after = hit.after
        else:
            return False
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
            st = self.timeline.state
            st.cursor_gpu = gpu
            if not st.is_tasks and hit.zone == "past" and hit.x is not None:
                st.set_cursor_x(hit.x)
                st.cursor_after = False
            elif hit.col is not None:
                st.cursor_zone = "cards" if st.is_tasks else "tail"
                st.cursor_col = min(hit.col, self.timeline.n_items)
                st.cursor_after = hit.after
        if event.assignment_id:
            a = self.store.assignments.get(event.assignment_id)
            if a:
                self.queue.select(a.task_id)
        self.refresh_timeline()
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
