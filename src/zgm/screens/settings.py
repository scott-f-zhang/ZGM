"""Settings view: True/False settings stored in the database, so every zgm on it
(daemon, TUIs, ``zgm submit``) goes by the same ones; and every key of the TUI
(the views' footers show only the common ones)."""

from __future__ import annotations

from rich.columns import Columns
from rich.table import Table
from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import Footer, Static

from ..widgets import theme
from ..widgets.text_toggle import TextToggle
from ..widgets.theme import box_title
from ..widgets.view_tabs import ViewTabs

# (Executor property, name, what it does)
SETTINGS = [
    ("autostart", "Auto-start", "Start queued tasks when their GPUs free up (p)."),
    ("autofill", "Auto-fill idle GPUs",
     "Idle GPUs take auto-fill tasks, highest priority first."),
]


class SettingsScreen(Screen):
    BINDINGS = [
        Binding("down,j", "app.focus_next", "next", show=False),
        Binding("up,k", "app.focus_previous", "previous", show=False),
    ]

    def compose(self) -> ComposeResult:
        yield ViewTabs("settings", id="view-tabs")
        with Vertical(id="settings-box") as box:
            box.border_title = box_title("settings", info="shared by every zgm on this queue")
            for key, name, help_text in SETTINGS:
                with Vertical(classes="setting-row"):
                    with Horizontal(classes="setting-line"):
                        yield Static(Text(name, style=f"bold {theme.MAIN_FG}"), classes="setting-name")
                        yield TextToggle(getattr(self.app.executor, key), id=f"setting-{key}")
                    yield Static(Text(help_text, style=theme.SUBTLE), classes="setting-help")
            yield Static(id="settings-note")
        with VerticalScroll(id="keys-box") as keys:
            keys.border_title = box_title("keys")
            yield Static(id="keys")
        yield Footer()

    def key_sections(self):
        """(title, bindings) for each part of the TUI, from their BINDINGS."""
        from ..app import ZGMApp
        from ..widgets.gpu_timeline import GPUTimeline
        from ..widgets.queue_overview import QueueOverview
        from ..widgets.task_queue import TaskQueue
        from .config import ConfigScreen
        from .schedule import ScheduleScreen

        return [("everywhere", ZGMApp.BINDINGS), ("schedule", ScheduleScreen.BINDINGS),
                ("queue", TaskQueue.BINDINGS), ("full-screen queue", QueueOverview.BINDINGS),
                ("timeline", GPUTimeline.BINDINGS), ("configs", ConfigScreen.BINDINGS)]

    def key_text(self, binding: Binding) -> str:
        """``a`` / ``[ / shift+←``: each key of a binding as Textual displays it."""
        shown = []
        for key in binding.key.split(","):
            text = self.app.get_key_display(Binding(key.strip(), binding.action))
            if text not in shown:
                shown.append(text)
        return " / ".join(shown)

    def refresh_keys(self) -> None:
        tables = []
        for title, bindings in self.key_sections():
            table = Table.grid(padding=(0, 2))
            table.add_column(style=f"bold {theme.HI_FG}", no_wrap=True)
            table.add_column(style=theme.MAIN_FG)
            table.add_row(Text(title, style=f"bold {theme.TITLE}"), "")
            for b in bindings:
                if b.description:
                    table.add_row(self.key_text(b), b.description)
            tables.append(table)
        self.query_one("#keys", Static).update(Columns(tables, padding=(1, 4)))

    def on_mount(self) -> None:
        self.query(TextToggle).first().focus()
        self.refresh_note()
        self.refresh_keys()

    def on_screen_resume(self) -> None:
        """Another zgm may have changed them (e.g. p in another TUI)."""
        for key, _, _ in SETTINGS:
            toggle = self.query_one(f"#setting-{key}", TextToggle)
            on = getattr(self.app.executor, key)
            if toggle.value != on:
                with toggle.prevent(TextToggle.Changed):
                    toggle.value = on
        self.refresh_note()

    def on_text_toggle_changed(self, event: TextToggle.Changed) -> None:
        key = event.toggle.id.removeprefix("setting-")
        if getattr(self.app.executor, key) == event.value:
            return
        if key == "autostart":
            self.app.set_autostart(event.value)  # starts what is ready at once
        else:
            setattr(self.app.executor, key, event.value)
            if event.value:
                self.app.dispatch()
        name = next(n for k, n, _ in SETTINGS if k == key)
        self.notify(f"{name} {'on' if event.value else 'off'}")
        self.refresh_note()

    def refresh_note(self) -> None:
        ex = self.app.executor
        note = Text()
        if ex.autofill and not ex.autostart:
            note.append("Auto-fill waits while auto-start is off.\n", style=theme.WARN)
        if self.app.executor_info() is None:
            note.append("Nothing runs the queue yet: these take effect once one does "
                        "(zgm daemon start).", style=theme.WARN)
        self.query_one("#settings-note", Static).update(note)
