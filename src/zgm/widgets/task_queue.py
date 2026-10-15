"""TaskQueue: horizontal, scrollable strip of task cards."""

from __future__ import annotations

from textual.binding import Binding
from textual.containers import HorizontalScroll
from textual.message import Message
from textual.widgets import Static

from ..models import Task
from .task_card import TaskCard


class TaskQueue(HorizontalScroll, can_focus=True):
    BINDINGS = [
        Binding("left", "move(-1)", "Prev", show=False),
        Binding("right", "move(1)", "Next", show=False),
        Binding("home", "jump(0)", show=False),
        Binding("end", "jump(-1)", show=False),
    ]

    class SelectionChanged(Message):
        def __init__(self, task_id: str | None) -> None:
            super().__init__()
            self.task_id = task_id

    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        self.order: list[str] = []
        self.selected_id: str | None = None

    def set_tasks(self, tasks: list[Task]) -> None:
        ids = [t.id for t in tasks]
        cards = {c.task_model.id: c for c in self.query(TaskCard)}
        for tid, card in cards.items():
            if tid not in ids:
                card.remove()
        empty = self.query("#queue-empty")
        if ids:
            empty.remove()
        elif not empty:
            self.mount(Static("Queue empty — every task is on the timeline. Press [b]n[/b] to add "
                              "one, or [b]T[/b] in the Configs view.", id="queue-empty"))
        for i, t in enumerate(tasks):
            card = cards.get(t.id)
            if card is None:
                card = TaskCard(t)
                self.mount(card)
            else:
                card.update_task(t)
        # keep DOM order in sync with the sorted order
        for i, t in enumerate(tasks):
            card = self.query_one(f"#card-{t.id}", TaskCard)
            if i > 0:
                self.move_child(card, after=self.query_one(f"#card-{tasks[i - 1].id}", TaskCard))
        self.order = ids
        if self.selected_id not in ids:
            self.select(ids[0] if ids else None)
        else:
            self._apply_selection()

    def select(self, task_id: str | None) -> None:
        changed = task_id != self.selected_id
        self.selected_id = task_id
        self._apply_selection()
        if changed:
            self.post_message(self.SelectionChanged(task_id))

    def _apply_selection(self) -> None:
        for card in self.query(TaskCard):
            card.selected = card.task_model.id == self.selected_id
            if card.selected:
                self.call_after_refresh(self.scroll_to_widget, card, animate=False)

    def action_move(self, delta: int) -> None:
        if not self.order:
            return
        idx = self.order.index(self.selected_id) if self.selected_id in self.order else 0
        self.select(self.order[max(0, min(len(self.order) - 1, idx + delta))])

    def action_jump(self, idx: int) -> None:
        if self.order:
            self.select(self.order[idx])

    def on_task_card_selected(self, event: TaskCard.Selected) -> None:
        self.focus()
        self.select(event.task_id)
