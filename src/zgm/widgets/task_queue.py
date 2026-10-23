"""TaskQueue: the waiting tasks as cards, in two halves side by side: those
marked for auto-fill (left) and the manual ones (right), each a horizontal,
scrollable strip sorted by priority, highest first (oldest among equals: the
order auto-fill takes them in)."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, HorizontalScroll, Vertical
from textual.message import Message
from textual.widgets import Static

from ..models import Task
from .task_card import TaskCard
from . import theme

HALVES = {"auto": "auto-fill", "manual": "manual"}
EMPTY = {"auto": "No task marked for auto-fill (space in the full-screen queue, f).",
         "manual": "No manual task. Press [b]n[/b] to add one, or [b]T[/b] in the Configs view."}


def queue_order(tasks: list[Task]) -> list[Task]:
    return sorted(tasks, key=lambda t: (-t.priority, t.created_at))


class QueueStrip(HorizontalScroll, can_focus=False):
    """One half's cards; the TaskQueue around it has the focus."""


class TaskQueue(Horizontal, can_focus=True):
    BINDINGS = [
        Binding("left", "move(-1)", "previous task", show=False),
        Binding("right", "move(1)", "next task", show=False),
        Binding("home", "jump(0)", "first task", show=False),
        Binding("end", "jump(-1)", "last task", show=False),
    ]

    class SelectionChanged(Message):
        def __init__(self, task_id: str | None) -> None:
            super().__init__()
            self.task_id = task_id

    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        self.order: list[str] = []  # auto-fill half, then manual half
        self.selected_id: str | None = None

    def compose(self) -> ComposeResult:
        for kind in HALVES:
            with Vertical(id=f"queue-{kind}", classes="queue-half"):
                yield Static("", id=f"queue-{kind}-title", classes="queue-half-title")
                yield QueueStrip(id=f"queue-{kind}-strip", classes="queue-strip")

    def set_tasks(self, tasks: list[Task]) -> None:
        halves = {"auto": queue_order([t for t in tasks if t.autofill]),
                  "manual": queue_order([t for t in tasks if not t.autofill])}
        ids = [t.id for kind in HALVES for t in halves[kind]]
        cards = {c.task_model.id: c for c in self.query(TaskCard)}
        for kind, half in halves.items():
            strip = self.query_one(f"#queue-{kind}-strip", QueueStrip)
            for t in half:  # a card whose task changed halves starts over in the other
                card = cards.get(t.id)
                if card is not None and card.parent is not strip:
                    card.remove()
                    del cards[t.id]
            self._title(kind, half)
            empty = strip.query(".queue-empty")
            if half:
                empty.remove()
            elif not empty:
                strip.mount(Static(EMPTY[kind], classes="queue-empty"))
            prev = None
            for t in half:
                card = cards.get(t.id)
                if card is None:
                    card = cards[t.id] = TaskCard(t)
                    strip.mount(card)
                else:
                    card.update_task(t)
                if prev is not None:  # keep DOM order in sync with the sorted order
                    strip.move_child(card, after=prev)
                prev = card
        for tid, card in cards.items():
            if tid not in ids:
                card.remove()
        self.order = ids
        if self.selected_id not in ids:
            self.select(ids[0] if ids else None)
        else:
            self._apply_selection()

    def _title(self, kind: str, half: list[Task]) -> None:
        title = self.query_one(f"#queue-{kind}-title", Static)
        title.update(f"[b {theme.TITLE}]{HALVES[kind]}[/] [{theme.SUBTLE}]{len(half)}[/]")

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
                self.call_after_refresh(card.scroll_visible, animate=False)

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
