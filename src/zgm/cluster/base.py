"""What a cluster backend is: a source of GPU groups.

zgm on one machine needs no backend: its GPUs are this machine's. A backend
lets one zgm queue span several machines. It only answers "which groups of
GPUs exist now, and how is each one reached": a Slurm job on some node, one
PC of a list, ... Everything else is shared (``cluster.core``): probing a
group's GPUs, giving them zgm ids, starting tasks there, and taking the GPUs
away again when the group goes.

A new backend is a module with a class that implements ``Backend``, added to
``BACKENDS`` in ``cluster/__init__.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class Group:
    """GPUs that come and go together, on one host."""

    key: str  # unique within the backend: a job id, a host name, ...
    host: str  # where it is reached ("" while it has no host yet, e.g. a pending job)
    usable: bool  # zgm may put work on its GPUs now
    note: str = ""  # why it is not usable, if it exists but isn't
    transport: str = "ssh"  # how its host is reached (cluster/transport.py)
    info: dict = field(default_factory=dict)  # the backend's own data, for columns()


class Backend(Protocol):
    name: str  # as in ``zgm daemon start --cluster NAME``
    interval: float  # seconds between calls to groups()
    # prefixes of environment variables that describe the group this process
    # runs in (e.g. SLURM_): a task started on another host doesn't get them
    host_env: tuple[str, ...]

    def groups(self) -> list[Group]:
        """The groups that exist now. Raises RuntimeError if it can't tell."""
        ...

    def local_key(self) -> str | None:
        """The group this process runs in, if any: its GPUs are read and used
        directly, without the transport."""
        ...

    def title(self, key: str) -> str:
        """A group's name in messages, e.g. "Slurm job 1001"."""
        ...

    def columns(self, row: dict, local: bool) -> list[str]:
        """A group as display columns (dashboard, ``zgm daemon status``): the key
        first, the note last. ``row`` is its ``gpu_groups`` row; ``local``: this
        process runs in it."""
        ...

    def order(self, row: dict) -> tuple:
        """Display order: usable groups first, say."""
        ...
