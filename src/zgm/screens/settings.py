"""Settings view: True/False settings stored in the database, so every zgm on it
(daemon, TUIs, ``zgm submit``) goes by the same ones."""

from __future__ import annotations

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
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
        yield Footer()

    def on_mount(self) -> None:
        self.query(TextToggle).first().focus()
        self.refresh_note()

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
