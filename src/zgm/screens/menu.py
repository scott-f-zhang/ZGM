"""MenuScreen: what Esc opens when there is nothing to cancel. The name in large
shaded letters and the version, then the choices, picked with ↑/↓ and Enter."""

from __future__ import annotations

from rich.color import Color
from rich.style import Style
from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import OptionList, Static
from textual.widgets.option_list import Option

from .. import __version__
from ..widgets import theme

# 4-row block letters (half blocks round the diagonals and curves)
LETTERS = {
    "Z": ["██████████", "      ▄██▀", "  ▄██▀    ", "██████████"],
    "G": [" ▄████████", "███       ", "███   ████", " ▀███████▀"],
    "M": ["████▄  ▄████", "███ ▀██▀ ███", "███      ███", "███      ███"],
}
GAP = "   "
# left-to-right colour sweep over the letters, and the drop shadow's colour
GRADIENT = ["#5fd7ff", "#af87ff", "#ff5f87"]
SHADOW = "#3a3a4a"
CHOICES = [("options", "Options", "settings (3)"),
           ("exit", "Exit", "running jobs keep running")]


def _mix(stops: list[str], t: float) -> str:
    """The colour at ``t`` (0..1) along the gradient ``stops``."""
    seg = min(len(stops) - 2, int(t * (len(stops) - 1)))
    f = t * (len(stops) - 1) - seg
    a = Color.parse(stops[seg]).triplet
    b = Color.parse(stops[seg + 1]).triplet
    return "#{:02x}{:02x}{:02x}".format(*(round(x + (y - x) * f) for x, y in zip(a, b)))


def banner(word: str) -> Text:
    """``word`` in block letters with a colour sweep, and a dotted shadow one
    cell down and to the right."""
    rows = [GAP.join(LETTERS[c][i] for c in word) for i in range(4)]
    width, height = len(rows[0]) + 1, len(rows) + 1
    ink = lambda r, c: 0 <= r < len(rows) and 0 <= c < len(rows[r]) and rows[r][c] != " "
    out = Text(justify="center", no_wrap=True)
    for r in range(height):
        for c in range(width):
            if ink(r, c):
                colour = _mix(GRADIENT, c / max(1, width - 2))
                # a little lighter on top, so the letters look lit from above
                out.append(rows[r][c], Style(color=colour, bold=r == 0))
            elif ink(r - 1, c - 1):
                out.append("░", Style(color=SHADOW))
            else:
                out.append(" ")
        if r < height - 1:
            out.append("\n")
    return out


class MenuScreen(ModalScreen[str | None]):
    """Dismisses with the chosen key ("options" / "exit"), or None (Esc)."""

    BINDINGS = [Binding("escape,q", "cancel", "back")]
    DEFAULT_CSS = """
    MenuScreen { align: center middle; }
    MenuScreen > Vertical {
        width: 80; height: auto; padding: 2 6;
        border: round #b0a9de; background: #000000;
    }
    MenuScreen #menu-banner { height: 5; width: 100%; }
    MenuScreen #menu-version { width: 100%; margin: 1 0 2 0; }
    MenuScreen OptionList, MenuScreen OptionList:focus {
        height: auto; border: none; background: #000000; padding: 0;
    }
    MenuScreen #menu-hint { width: 100%; margin-top: 2; }
    """

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static(banner("ZGM"), id="menu-banner")
            yield Static(Text(f"v{__version__}  ·  GPU queue", style=theme.SUBTLE, justify="center"),
                         id="menu-version")
            options = []
            for key, label, note in CHOICES:
                # three lines per choice, so the highlight reads as a big button
                line = Text("\n")
                line.append(f"   {label:<12}", style=f"bold {theme.MAIN_FG}")
                line.append(note, style=theme.SUBTLE)
                line.append("\n")
                options.append(Option(line, id=key))
            yield OptionList(*options, id="menu-options")
            yield Static(Text("↑/↓ choose  ·  Enter select  ·  Esc back", style=theme.SUBTLE,
                              justify="center"), id="menu-hint")

    def on_mount(self) -> None:
        options = self.query_one(OptionList)
        options.highlighted = 0
        options.focus()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(event.option.id)

    def action_cancel(self) -> None:
        self.dismiss(None)
