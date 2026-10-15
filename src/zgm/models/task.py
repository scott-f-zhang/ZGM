"""Task: a unit of work that needs GPUs. Where it runs lives in Assignment."""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from .launch import LaunchSpec


class TaskStatus(str, Enum):
    PENDING = "pending"  # not on the timeline
    PLANNED = "planned"  # has an active planned assignment
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


def new_id() -> str:
    return uuid.uuid4().hex[:8]


def _digest(payload: object) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def command_fingerprint(command: str, gpus: int) -> str:
    """Identity of a plain-command task's workload: the command and its GPU count."""
    return "cmd:" + _digest({"command": " ".join(command.split()), "gpus": gpus})


def launch_fingerprint(spec: LaunchSpec, gpus: int) -> str:
    """Identity of a plain task's workload: what it runs, where, and its GPU count.
    The environment is not part of it. A bare command keeps its old fingerprint."""
    if spec.command and not spec.cwd:
        return command_fingerprint(spec.command, gpus)
    return "cmd:" + _digest({
        "cwd": spec.cwd, "script": spec.script, "module": spec.module,
        "command": spec.command, "args": spec.args, "gpus": gpus,
    })


def config_fingerprint(params: dict, gpus: int) -> str:
    """Identity of a config-driven workload: resolved parameters and GPU count."""
    return "cfg:" + _digest({"params": params, "gpus": gpus})


@dataclass
class Task:
    """A unit of work. ``estimated_duration`` is never typed in by the user: it
    is derived from past completed runs with the same ``fingerprint`` (same
    config parameters, or same command), and is ``None`` when there are none."""

    name: str
    command: str = ""  # one-line summary of ``launch`` (shown on cards)
    required_gpus: int = 1
    estimated_duration: timedelta | None = None
    status: TaskStatus = TaskStatus.PENDING
    id: str = field(default_factory=new_id)
    created_at: datetime = field(default_factory=lambda: datetime.now().replace(microsecond=0))
    notes: str = ""
    config_path: str | None = None
    fingerprint: str = ""
    estimate_runs: int = 0  # how many past runs the estimate is based on (not stored)
    launch: LaunchSpec | None = None  # how to start it; a bare command if not given

    def __post_init__(self) -> None:
        if self.required_gpus < 1:
            raise ValueError("required_gpus must be >= 1")
        if self.estimated_duration is not None and self.estimated_duration <= timedelta(0):
            raise ValueError("estimated_duration must be positive")
        if self.launch is None:
            self.launch = LaunchSpec(command=self.command)
        elif not self.command:
            self.command = self.launch.describe()
        if not self.fingerprint:
            self.fingerprint = launch_fingerprint(self.launch, self.required_gpus)


def format_estimate(task: Task, long: bool = False) -> str:
    if task.estimated_duration is None:
        return "no run history" if long else "no data"
    text = "~" + format_duration(task.estimated_duration)
    if long:
        runs = task.estimate_runs
        text += f" (median of {runs} past run{'s' if runs != 1 else ''})"
    return text


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
