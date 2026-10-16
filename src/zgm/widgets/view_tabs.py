"""btop-style view switcher strip shown at the top of every view."""

from __future__ import annotations

from rich.cells import cell_len
from rich.text import Text
from textual import events
from textual.widgets import Static

from . import theme

VIEWS = [("schedule", 1, "schedule"), ("configs", 2, "configs")]
BRAND = " zgm "
GAP = "   "


class ViewTabs(Static):
    """The top line, like btop's: the name, then each view with its key as a
    red superscript; the open view is bright, the others dim."""

    def __init__(self, active: str, **kw) -> None:
        super().__init__(**kw)
        self.active = active

    def render(self) -> Text:
        out = Text(no_wrap=True)
        out.append(BRAND, style=f"bold {theme.TITLE}")
        out.append("│", style=theme.DIV_LINE)
        for mode, key, label in VIEWS:
            out.append(GAP)
            out.append(theme.sup(key), style=theme.HI_FG)
            if mode == self.active:
                out.append(label, style=f"bold {theme.TITLE} underline")
            else:
                out.append(label, style=theme.SUBTLE)
        return out

    def on_click(self, event: events.Click) -> None:
        # click a tab to switch views
        x = event.x
        pos = cell_len(BRAND) + 1
        for mode, key, label in VIEWS:
            pos += len(GAP)
            w = cell_len(theme.sup(key) + label)
            if pos <= x < pos + w:
                self.app.switch_mode(mode)
                return
            pos += w
