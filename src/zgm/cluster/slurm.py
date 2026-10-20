"""The Slurm backend: each of your running, single-node Slurm jobs with GPUs is a
group, reached by ssh to its node.

``squeue -u $USER`` lists the jobs, every 30 s. Ssh to a node where you have a
job lands in that job when the cluster uses pam_slurm_adopt (most do; it is
what lets users ssh to their nodes at all), so the probe there sees exactly
the job's GPUs, numbered from 0, and a task started there lives as long as
the job.

Not used, with a note: jobs on several nodes, and jobs on a node where you
have another running job (ssh would land in the newest of them, whose GPUs
may be other ones).
"""

from __future__ import annotations

import getpass
import os
import subprocess
from collections.abc import Callable
from datetime import datetime

from .base import Group

SQUEUE_FORMAT = "%i|%T|%j|%b|%N|%S|%e|%L|%r|%D"


def _time(value: str) -> str | None:
    """squeue's time column as ISO text; None for N/A, Unknown, ..."""
    try:
        return datetime.fromisoformat(value).isoformat(timespec="seconds")
    except ValueError:
        return None


def parse_gpus(gres: str) -> int:
    """GPUs per node in squeue's TRES column: ``gres/gpu:h100:4`` -> 4."""
    total = 0
    for item in gres.split(","):
        if "gpu" not in item:
            continue
        last = item.rsplit(":", 1)[-1].split("(")[0]
        total += int(last) if last.isdigit() else 1
    return total


def gpu_type(gres: str) -> str:
    """``gres/gpu:h100:4`` -> ``h100``; '' if the job names no type."""
    for item in gres.split(","):
        parts = item.split(":")
        if "gpu" in parts[0] and len(parts) == 3:
            return parts[1]
    return ""


class SlurmBackend:
    name = "slurm"
    interval = 30.0
    host_env = ("SLURM_",)  # the job zgm runs in, not the one a task runs in

    def __init__(self, run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
                 user: str | None = None, job: str | None = None) -> None:
        self.run = run  # subprocess.run; tests pass a fake
        self.user = user or getpass.getuser()
        self.job = job if job is not None else os.environ.get("SLURM_JOB_ID") or None

    def local_key(self) -> str | None:
        return self.job

    def title(self, key: str) -> str:
        return f"Slurm job {key}"

    def groups(self) -> list[Group]:
        try:
            done = self.run(["squeue", "-h", "-u", self.user, "-o", SQUEUE_FORMAT],
                            capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.TimeoutExpired) as e:
            raise RuntimeError(f"squeue: {e}") from None
        if done.returncode != 0:
            raise RuntimeError(f"squeue: {(done.stderr or done.stdout).strip()}")
        jobs = []
        for line in done.stdout.splitlines():
            parts = line.strip().split("|")
            if len(parts) == 10:
                jobs.append(parts)
        per_node: dict[str, int] = {}
        for _, state, _, _, node, *_ in jobs:
            if state == "RUNNING" and node:
                per_node[node] = per_node.get(node, 0) + 1
        out = []
        for jid, state, name, gres, node, start, end, left, reason, nodes in jobs:
            gpus = parse_gpus(gres)
            note = ""
            if state == "RUNNING" and gpus:
                if nodes.isdigit() and int(nodes) > 1:
                    note = "runs on several nodes: not used"
                elif jid != self.job and per_node.get(node, 0) > 1:
                    note = "shares its node with another of your jobs: not used"
            out.append(Group(
                key=jid, host=node if state == "RUNNING" else "",
                usable=state == "RUNNING" and gpus > 0 and bool(node) and not note,
                note=note,
                info={"state": state, "name": name, "gres": gres, "start": _time(start),
                      "end": _time(end), "time_left": left,
                      "reason": reason if reason != "None" else ""}))
        return out

    def columns(self, row: dict, local: bool) -> list[str]:
        """id, state, node, GPUs asked for, then the note: time left, expected
        start, why it waits, why zgm doesn't use it."""
        info = row["info"]
        gres = info.get("gres", "")
        n, kind = parse_gpus(gres), gpu_type(gres)
        asked = f"{n} x {kind}" if n and kind else (f"{n} GPU" if n else "no GPU")
        state = info.get("state", "?") if row["state"] != "gone" else "ENDED"
        if state == "RUNNING":
            note = row["note"] or (f"{info['time_left']} left" if info.get("time_left") else "")
            if local and not row["note"]:
                note = f"this job · {note}" if note else "this job"
        elif state == "PENDING":
            start = info.get("start")
            note = f"starts ~{datetime.fromisoformat(start):%m-%d %H:%M}" if start else \
                "start unknown"
            if info.get("reason"):
                note += f" ({info['reason']})"
        else:
            note = info.get("reason", "")
        return [row["key"], state, row["host"] or "-", asked, note]

    def order(self, row: dict) -> tuple:
        state = row["info"].get("state", "")
        key = row["key"]
        return ({"RUNNING": 0, "PENDING": 1}.get(state, 2), row["info"].get("start") or "~",
                int(key) if key.isdigit() else 0, key)
