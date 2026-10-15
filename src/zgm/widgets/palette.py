"""Stable per-task colours shared by cards and timeline blocks."""

from __future__ import annotations

import zlib

TASK_COLORS = [
    "#3b82f6",
    "#10b981",
    "#f59e0b",
    "#8b5cf6",
    "#ec4899",
    "#14b8a6",
    "#f97316",
    "#6366f1",
    "#84cc16",
    "#06b6d4",
]
HISTORY_COLOR = "#4b5563"
GHOST_OK = "#22c55e"
GHOST_BAD = "#ef4444"


def task_color(task_id: str) -> str:
    return TASK_COLORS[zlib.crc32(task_id.encode()) % len(TASK_COLORS)]
