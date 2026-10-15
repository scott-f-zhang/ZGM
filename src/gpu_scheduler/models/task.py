"""Task: a unit of work that needs GPUs. Where it runs lives in Assignment."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum


class TaskStatus(str, Enum):
    PENDING = "pending"  # not on the timeline
    PLANNED = "planned"  # has an active planned assignment
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


def new_id() -> str:
    return uuid.uuid4().hex[:8]


@dataclass
class Task:
    name: str
    command: str
    required_gpus: int = 1
    estimated_duration: timedelta = timedelta(hours=1)
    status: TaskStatus = TaskStatus.PENDING
    id: str = field(default_factory=new_id)
    created_at: datetime = field(default_factory=lambda: datetime.now().replace(microsecond=0))
    notes: str = ""
    config_path: str | None = None

    def __post_init__(self) -> None:
        if self.required_gpus < 1:
            raise ValueError("required_gpus must be >= 1")
        if self.estimated_duration <= timedelta(0):
            raise ValueError("estimated_duration must be positive")


def format_duration(d: timedelta) -> str:
    minutes = int(d.total_seconds() // 60)
    h, m = divmod(minutes, 60)
    if h and m:
        return f"{h}h{m:02d}m"
    if h:
        return f"{h}h"
    return f"{m}m"


def parse_duration(text: str) -> timedelta:
    """Parse '90m', '2h', '1h30m', '1.5h' or a bare number of minutes."""
    s = str(text).strip().lower().replace(" ", "")
    if not s:
        raise ValueError("empty duration")
    try:
        minutes = float(s)
    except ValueError:
        pass
    else:
        if minutes <= 0:
            raise ValueError(f"duration must be positive: {text!r}")
        return timedelta(minutes=minutes)
    total = 0.0
    num = ""
    matched = False
    for ch in s:
        if ch.isdigit() or ch == ".":
            num += ch
            continue
        if not num:
            raise ValueError(f"invalid duration: {text!r}")
        unit = {"d": 1440, "h": 60, "m": 1, "s": 1 / 60}.get(ch)
        if unit is None:
            raise ValueError(f"invalid duration unit in {text!r}")
        total += float(num) * unit
        num = ""
        matched = True
    if num or not matched or total <= 0:
        raise ValueError(f"invalid duration: {text!r}")
    return timedelta(minutes=total)
