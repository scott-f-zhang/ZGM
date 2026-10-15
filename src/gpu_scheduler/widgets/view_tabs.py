"""btop-style view switcher strip shown at the top of every view."""

from __future__ import annotations

from rich.cells import cell_len
from rich.text import Text
from textual import events
from textual.widgets import Static

VIEWS = [("schedule", "1", "Schedule"), ("configs", "2", "Configs")]


class ViewTabs(Static):
    def __init__(self, active: str, **kw) -> None:
        super().__init__(**kw)
        self.active = active

    def render(self) -> Text:
        out = Text(no_wrap=True)
        out.append(" gpu-scheduler ", style="bold #0b0f19 on #60a5fa")
        out.append("  ")
        for mode, key, label in VIEWS:
            if mode == self.active:
                out.append(f" {key} ", style="bold #0b0f19 on #f59e0b")
                out.append(f" {label} ", style="bold #f59e0b on #1f2937")
            else:
                out.append(f" {key} ", style="#9ca3af on #1f2937")
                out.append(f" {label} ", style="#9ca3af")
            out.append("  ")
        return out

    def on_click(self, event: events.Click) -> None:
        # click a tab to switch views
        x = event.x
        pos = len(" gpu-scheduler ") + 2
        for mode, key, label in VIEWS:
            w = cell_len(f" {key} ") + cell_len(f" {label} ")
            if pos <= x < pos + w:
                self.app.switch_mode(mode)
                return
            pos += w + 2
