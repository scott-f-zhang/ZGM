"""ClusterPanel: the groups of GPUs of a cluster backend, one line each (cluster
mode only). What a line says is the backend's: for Slurm, each of your jobs,
running ones with the zgm GPU ids they got, pending ones with their expected
start."""

from __future__ import annotations

from datetime import datetime

from rich.text import Text
from textual.widgets import Static

from . import theme

STATE_COLOR = {"RUNNING": theme.OK, "PENDING": theme.WARN}
STALE = 120  # seconds: older data means no executor follows the groups any more


class ClusterPanel(Static, can_focus=True):
    has_groups = False
    hidden_by_zoom = False  # another panel is full screen
    hotkey = "s"  # ScheduleScreen focuses it with this key

    def show(self, provider) -> None:
        """``provider``: the app's ClusterProvider."""
        table = provider.table()
        self.has_groups = bool(table)
        self.display = bool(table) and not self.hidden_by_zoom
        if not table:
            return
        widths = [max(len(r[i]) for r in table) for i in range(len(table[0]) - 1)]
        text = Text()
        for i, cols in enumerate(table):
            if i:
                text.append("\n")
            for j, (col, w) in enumerate(zip(cols, widths)):
                style = (STATE_COLOR.get(col, theme.MAIN_FG) if j == 1 else
                         theme.TITLE if col.startswith("GPU ") else theme.MAIN_FG)
                text.append(col.ljust(w) + "  ", style=style)
            note = cols[-1]
            bad = note.endswith("not used") or note.startswith("probe failed")
            text.append(note, style=theme.BAD if bad else theme.SUBTLE)
        seen = provider.seen_at()
        info = f"{len(table)} group{'s' if len(table) > 1 else ''}"
        if seen:
            info += f" · as of {seen[11:16]}"
            if (datetime.now() - datetime.fromisoformat(seen)).total_seconds() > STALE:
                info += " · not updated: no executor follows them"
        self.border_title = theme.box_title(provider.backend.name, key=self.hotkey, info=info)
        self.update(text)
