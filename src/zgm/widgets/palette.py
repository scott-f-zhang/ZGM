"""Stable per-task colours shared by cards and timeline blocks: the bright
stops of btop's default gradients, so dark text stays readable on them."""

from __future__ import annotations

import zlib

from . import theme

TASK_COLORS = [
    "#77ca9b",  # cpu start
    "#cbc06c",  # cpu mid
    "#4897d4",  # temp start
    "#d9626d",  # used mid
    "#74e6fc",  # cached mid
    "#b5e685",  # free mid
    "#ffd77a",  # available mid
    "#dcafde",  # upload end
    "#ff40b6",  # temp end
    "#b0a9de",  # download end
]
HISTORY_COLOR = "#4e5f68"
GHOST_OK = theme.OK
GHOST_BAD = theme.BAD


def task_color(task_id: str) -> str:
    return TASK_COLORS[zlib.crc32(task_id.encode()) % len(TASK_COLORS)]
