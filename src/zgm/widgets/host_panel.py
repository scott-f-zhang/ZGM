"""HostPanel: CPU, memory and disk of the machine this zgm runs on, one meter
each. Disk is the filesystem of the state dir (logs, database), plus the home
directory's if that is another one: when either fills up, runs and zgm fail."""

from __future__ import annotations

import os
import socket
import time
from pathlib import Path

import psutil
from rich.text import Text
from textual.widgets import Static

from . import theme

GIB = 1024 ** 3
LOW_DISK = 0.10  # free share under which the disk line turns red
METER = 12
EVERY = 10.0  # seconds between samples (each one repaints the box)


def disk_paths(state_dir: str | None) -> list[str]:
    """The state dir and the home directory, one per filesystem."""
    out, seen = [], set()
    for p in (state_dir, str(Path.home())):
        if not p:
            continue
        try:
            dev = os.stat(p).st_dev
        except OSError:
            continue
        if dev not in seen:
            seen.add(dev)
            out.append(p)
    return out


def mount_of(path: str) -> str:
    path = os.path.realpath(path)
    while not os.path.ismount(path):
        path = os.path.dirname(path)
    return path


class HostPanel(Static, can_focus=True):
    lines = 0  # content lines shown

    def __init__(self, state_dir: str | None = None, **kw) -> None:
        super().__init__(**kw)
        self.paths = disk_paths(state_dir)
        self.border_title = theme.box_title("host", info=socket.gethostname().split(".")[0])
        self._sampled = 0.0
        psutil.cpu_percent(None)  # the first call only starts the measurement

    def on_mount(self) -> None:
        self.sample(force=True)

    def sample(self, force: bool = False) -> None:
        if not force and time.monotonic() - self._sampled < EVERY:
            return
        self._sampled = time.monotonic()
        text = Text(no_wrap=True)
        cpu = psutil.cpu_percent(None)
        cores = psutil.cpu_count() or 0
        try:
            load = f" · load {os.getloadavg()[0]:.1f}"
        except (AttributeError, OSError):  # Windows
            load = ""
        self._line(text, "CPU", cpu / 100, theme.CPU, f"{cores} cores{load}")
        mem = psutil.virtual_memory()
        self._line(text, "MEM", mem.percent / 100, theme.USED,
                   f"{(mem.total - mem.available) / GIB:.0f}/{mem.total / GIB:.0f} GiB")
        for p in self.paths:
            try:
                du = psutil.disk_usage(p)
            except OSError:
                continue
            free = du.free / du.total if du.total else 0.0
            self._line(text, "DISK", du.used / du.total if du.total else 0.0, theme.USED,
                       f"{du.free / GIB:.0f} GiB free · {mount_of(p)}",
                       bad=free < LOW_DISK)
        self.lines = len(text.split())
        if text != getattr(self, "_shown", None):  # update() re-lays out the screen
            self._shown = text
            self.update(text)

    @staticmethod
    def _line(text: Text, label: str, fraction: float, stops, note: str, bad: bool = False) -> None:
        if text:
            text.append("\n")
        text.append(f"{label:<5}", style=theme.TITLE)
        for seg in theme.meter(fraction, METER, stops):
            text.append(seg.text, style=seg.style)
        text.append(f" {fraction * 100:3.0f}%  ", style=theme.BAD if bad else theme.MAIN_FG)
        text.append(note + ("  low!" if bad else ""), style=theme.BAD if bad else theme.SUBTLE)
