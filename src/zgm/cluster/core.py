"""The part of cluster mode that is the same for every backend.

- ``ClusterWatch`` (only in the executor) asks the backend for its groups every
  ``backend.interval`` seconds and keeps them in the ``gpu_groups`` table. It
  probes each usable group every 10 s: ``python -m zgm.probe`` on its host,
  through its transport (in-process for the group this process runs in).
- A group's GPUs get zgm ids ``gpu_base ..`` the first time its probe works,
  and keep them while it is usable. When it stops being usable, the work
  queued on its GPUs goes back to the waiting queue, and the ids are freed once
  nothing runs there any more.
- ``ClusterProvider`` (in every zgm process: daemon, TUI, submit) builds the
  GPU devices from that table, like the NVML/mock providers do on one machine.
  ``locate`` tells the local runtime which host a GPU is on and how to reach it.

Hosts share the state directory with this one (the runners' files are how a
run on another host is followed), and run zgm with the same Python path.
``zgm check`` tests both.
"""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime

from ..models import AssignmentState, GPUDevice, GPUProcess
from ..runtime.local import Placement
from ..services.scheduler import AssignmentService
from ..services.task_store import TaskStore
from . import backend as load_backend, configured
from .base import Backend
from .transport import Transport, last_line, transport

PROBE_EVERY = 10.0  # seconds between probes of a group
PROBE_TIMEOUT = 30.0
PROBE_STALE = 60.0  # older probes don't count as knowing a GPU's processes

Message = tuple[str, str]  # (text, severity)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def short_name(name: str) -> str:
    return name.replace("NVIDIA ", "").replace("GeForce ", "")


def span(ids: Sequence[int]) -> str:
    return f"{ids[0]}-{ids[-1]}" if len(ids) > 1 else str(ids[0])


def group_gpu_ids(row: dict) -> list[int]:
    """The zgm GPU ids a group's row holds (none without a base or a probe)."""
    if row.get("gpu_base") is None or not row.get("probe"):
        return []
    return [row["gpu_base"] + i for i in range(len(row["probe"].get("gpus", [])))]


def probe_host(t: Transport, python: str = sys.executable,
               timeout: float = PROBE_TIMEOUT) -> dict:
    """``zgm.probe`` on the transport's host; raises RuntimeError if it fails."""
    try:
        done = t.run(shlex.join([python, "-m", "zgm.probe"]), timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise RuntimeError(f"{t.host}: {e}") from None
    lines = done.stdout.strip().splitlines()
    try:
        data = json.loads(lines[-1]) if lines else {}
    except ValueError:
        data = {}
    if done.returncode != 0 or "gpus" not in data:
        raise RuntimeError(f"{t.host}: {data.get('error') or last_line(done)}")
    return data


class ClusterWatch:
    """Keeps ``gpu_groups`` current. Only the executor runs one."""

    def __init__(self, store: TaskStore, backend: Backend, local) -> None:
        self.store = store
        self.backend = backend
        self.local = local  # this machine's NVML (or mock) provider
        self.mine = backend.local_key()
        self.pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="zgm-probe")
        self.pending: dict[str, Future] = {}
        self._listed = float("-inf")
        self._probed: dict[str, float] = {}
        self._failed: set[str] = set()  # groups whose probe failure was reported
        self._list_failed = False

    def close(self) -> None:
        self.pool.shutdown(wait=False, cancel_futures=True)

    def refresh(self, svc: AssignmentService, wait: bool = False) -> list[Message]:
        """Ask the backend and the hosts what is due; returns what changed.
        ``wait``: wait for the probes (at startup, so the GPUs are known)."""
        storage = self.store.storage
        rows = {r["key"]: r for r in storage.gpu_groups(self.backend.name)}
        changed: set[str] = set()
        messages: list[Message] = []
        now = time.monotonic()
        if now - self._listed >= self.backend.interval:
            try:
                groups = self.backend.groups()
            except RuntimeError as e:
                if not self._list_failed:
                    messages.append((f"{self.backend.name}: {e}", "error"))
                self._list_failed = True
            else:
                self._list_failed = False
                self._listed = now
                self._update(rows, groups, changed)
        for key, row in rows.items():
            if row["state"] != "up" or key in self.pending:
                continue
            if now - self._probed.get(key, float("-inf")) < PROBE_EVERY and not (
                    wait and key == self.mine):
                continue
            self._probed[key] = now
            if key == self.mine:
                from ..probe import probe

                self._probe_done(rows, row, lambda: probe(self.local), changed, messages)
            else:
                t = transport(row["transport"], row["host"])
                self.pending[key] = self.pool.submit(probe_host, t)
        for key, future in list(self.pending.items()):
            if wait:
                try:
                    future.result(timeout=PROBE_TIMEOUT)
                except Exception:
                    pass
            if not future.done():
                continue
            del self.pending[key]
            if key in rows and rows[key]["state"] == "up":
                self._probe_done(rows, rows[key], future.result, changed, messages)
        self._release(rows, svc, changed, messages)
        storage.save_gpu_groups([rows[k] for k in changed])
        return messages

    def _update(self, rows: dict[str, dict], groups, changed: set[str]) -> None:
        seen = _now()
        for g in groups:
            row = rows.setdefault(g.key, {"key": g.key, "backend": self.backend.name,
                                          "gpu_base": None, "probe": None, "probed_at": None,
                                          "problem": ""})
            row.update(state="up" if g.usable else "down", host=g.host, transport=g.transport,
                       note=g.note, info=g.info, seen_at=seen)
            changed.add(g.key)
        listed = {g.key for g in groups}
        for key, row in rows.items():
            if key not in listed and row["state"] != "gone":
                row.update(state="gone", note="", problem="")
                changed.add(key)

    def _probe_done(self, rows: dict[str, dict], row: dict, result: Callable[[], dict],
                    changed: set[str], messages: list[Message]) -> None:
        key = row["key"]
        changed.add(key)
        try:
            data = result()
        except Exception as e:
            row["problem"] = f"probe failed: {e}"
            if key not in self._failed:
                self._failed.add(key)
                messages.append((f"{self.backend.title(key)} on {row['host']}: can't read its "
                                 f"GPUs ({e}); zgm check tells why", "warning"))
            return
        self._failed.discard(key)
        gpus = data.get("gpus", [])
        old = len(row["probe"]["gpus"]) if row.get("probe") else None
        row.update(probe=data, probed_at=_now(), problem="")
        if row["gpu_base"] is not None and old != len(gpus):
            row["gpu_base"] = None  # its GPUs changed shape: new ids below
        if row["gpu_base"] is None and gpus:
            row["gpu_base"] = self._free_base(rows, len(gpus), exclude=key)
            names = ", ".join(sorted({short_name(g["name"]) for g in gpus}))
            messages.append((f"{row['host']} ({self.backend.title(key)}): {len(gpus)} x "
                             f"{names} added as GPU {span(group_gpu_ids(row))}", "information"))

    def _free_base(self, rows: dict[str, dict], count: int, exclude: str) -> int:
        """The lowest id where ``count`` GPUs fit beside the other groups' GPUs
        and the GPUs of work that is still running."""
        used = {g for r in rows.values() if r["key"] != exclude for g in group_gpu_ids(r)}
        used |= {g for a in self.store.assignments.values()
                 if a.state == AssignmentState.RUNNING for g in a.display_gpu_ids}
        base = 0
        while any(base + i in used for i in range(count)):
            base += 1
        return base

    def _release(self, rows: dict[str, dict], svc: AssignmentService, changed: set[str],
                 messages: list[Message]) -> None:
        for key, row in rows.items():
            ids = group_gpu_ids(row)
            if not ids or row["state"] == "up":
                continue
            gone = set(ids)
            title = self.backend.title(key)
            for a in svc.planned():
                if gone & set(a.gpu_ids):
                    task = self.store.tasks.get(a.task_id)
                    svc.unassign(a.id)
                    messages.append((f"{task.name if task else a.task_id}: back to the waiting "
                                     f"queue (GPU {span(ids)} left with {title})", "warning"))
            if any(gone & set(a.display_gpu_ids) for a in svc.running()):
                continue  # its runs end on their own (no heartbeat); free the ids then
            row["gpu_base"] = None
            changed.add(key)
            messages.append((f"{row['host'] or title} ({title}) "
                             f"{'ended' if row['state'] == 'gone' else 'is not used'}: "
                             f"GPU {span(ids)} removed", "warning"))


class ClusterProvider:
    """GPU devices from ``gpu_groups``: what NVML/mock providers are on one
    machine. The group this process runs in is read from the local provider;
    the others from their latest probe."""

    def __init__(self, storage, backend: Backend, local) -> None:
        self.storage = storage
        self.backend = backend
        self.local = local
        self.mine = backend.local_key()
        self._rows: list[dict] = []
        self._version: int | None = None

    def rows(self) -> list[dict]:
        version = self.storage.data_version()
        if version != self._version:  # re-read only after another process wrote
            self._version = version
            self._rows = self.storage.gpu_groups(self.backend.name)
        return self._rows

    def invalidate(self) -> None:
        """Re-read the table next time (after this process itself wrote to it)."""
        self._version = None

    def discover(self) -> list[GPUDevice]:
        out = []
        for row in self.rows():
            if row["state"] != "up":
                continue  # gone: its ids stay reserved while a run is still there
            for gid, g in zip(group_gpu_ids(row), row["probe"]["gpus"] if row["probe"] else []):
                out.append(GPUDevice(
                    id=gid, uuid=g["uuid"], name=g["name"], memory_total=g["memory_total"],
                    memory_used=g["memory_used"], utilization=g["utilization"],
                    group=row["key"], host=row["host"], local_index=g["index"]))
        out.sort(key=lambda d: d.id)
        self._refresh_local(out)
        return out

    def _refresh_local(self, devices: list[GPUDevice]) -> None:
        mine = [d for d in devices if d.group == self.mine]
        if not mine:
            return
        here = [replace(d, id=d.local_index) for d in mine]
        self.local.refresh(here)
        for d, h in zip(mine, here):
            d.memory_used, d.utilization = h.memory_used, h.utilization

    def refresh(self, devices: list[GPUDevice]) -> None:
        self.sync(devices)

    def sync(self, devices: list[GPUDevice]) -> bool:
        """Bring ``devices`` up to date, in place. True if the set of GPUs changed."""
        fresh = self.discover()
        key = lambda ds: [(d.id, d.group, d.uuid) for d in ds]  # noqa: E731
        if key(fresh) != key(devices):
            devices[:] = fresh
            return True
        for d, f in zip(devices, fresh):
            d.memory_used, d.utilization = f.memory_used, f.utilization
        return False

    def node_of(self, devices: list[GPUDevice]) -> dict[int, str]:
        return {d.id: d.group or "" for d in devices}

    def _find(self, gpu_id: int) -> tuple[dict, dict] | None:
        for row in self.rows():
            ids = group_gpu_ids(row) if row["state"] == "up" else []
            if gpu_id in ids:
                return row, row["probe"]["gpus"][ids.index(gpu_id)]
        return None

    def processes(self, gpu_id: int) -> list[GPUProcess]:
        found = self._find(gpu_id)
        if found is None:
            raise RuntimeError(f"GPU {gpu_id} is in no usable group")
        row, g = found
        if row["key"] == self.mine:
            return self.local.processes(g["index"])
        age = (datetime.now() - datetime.fromisoformat(row["probed_at"])).total_seconds() \
            if row.get("probed_at") else float("inf")
        if age > PROBE_STALE or not g.get("processes_visible", True):
            raise RuntimeError(f"no recent probe of {row['host']}")
        host = row["probe"].get("host") or row["host"]
        return [GPUProcess(p["pid"], p.get("name", "?"), p.get("used_memory"), host=host,
                           zgm_task=p.get("zgm_task")) for p in g.get("processes", [])]

    def locate(self, gpu_ids: Sequence[int]) -> Placement:
        """Where these GPUs are; raises ValueError if not all in one usable group."""
        places = [self._find(g) for g in gpu_ids]
        if any(p is None for p in places):
            missing = [g for g, p in zip(gpu_ids, places) if p is None]
            raise ValueError(f"GPU {','.join(map(str, missing))} is in no usable group")
        if len({row["key"] for row, _ in places}) > 1:
            raise ValueError("a task's GPUs must be on one host")
        row = places[0][0]
        remote = None if row["key"] == self.mine else transport(row["transport"], row["host"])
        return Placement(row["key"], row["host"], [g["index"] for _, g in places], remote,
                         drop_env=tuple(getattr(self.backend, "host_env", ())))

    def describe(self, gpus: list[GPUDevice]) -> str:
        if not gpus:
            return (f"no GPUs yet: no usable {self.backend.name} group has been probed "
                    f"(cluster: {self.backend.name})")
        hosts: dict[str, list[GPUDevice]] = {}
        for g in gpus:
            hosts.setdefault(g.host or "?", []).append(g)
        parts = []
        for host, gs in hosts.items():
            names = ", ".join(sorted({short_name(g.name) for g in gs}))
            parts.append(f"{host}: {len(gs)} x {names} (GPU {span([g.id for g in gs])})")
        return "; ".join(parts) + f" (cluster: {self.backend.name})"

    # display ------------------------------------------------------------------------
    def table(self) -> list[list[str]]:
        """The groups as display columns: the backend's, the zgm GPU ids, a note.
        Gone groups are left out."""
        rows = sorted((r for r in self.rows() if r["state"] != "gone"), key=self.backend.order)
        out = []
        for row in rows:
            cols = self.backend.columns(row, row["key"] == self.mine)
            ids = group_gpu_ids(row) if row["state"] == "up" else []
            note = cols[-1]
            if row.get("problem"):
                note = row["problem"]
            elif row["state"] == "up" and not ids:
                note = f"probing {row['host']}..."
            out.append([*cols[:-1], f"GPU {span(ids)}" if ids else "", note])
        return out

    def lines(self, indent: str = "") -> list[str]:
        table = self.table()
        if not table:
            return []
        widths = [max(len(r[i]) for r in table) for i in range(len(table[0]) - 1)]
        return [indent + "  ".join(c.ljust(w) for c, w in zip(r, widths)) + "  " + r[-1]
                for r in table]

    def seen_at(self) -> str | None:
        return max((r["seen_at"] for r in self.rows()), default=None)


def gpu_source(storage, mock_count: int | None = None):
    """The GPU provider for this database: its cluster's, or this machine's."""
    from ..services.gpu_discovery import make_provider

    local = make_provider(mock_count)
    name = configured(storage)
    return ClusterProvider(storage, load_backend(name), local) if name else local
