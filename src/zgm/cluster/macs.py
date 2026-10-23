"""The Macs backend: each Mac in a list is a group with one Apple GPU.

The hosts come from ``ZGM_MACS`` (comma- or space-separated) or from
``$ZGM_HOME/macs`` (one host per line, ``#`` starts a comment). They are
reached by ssh, and each must:

- mount the state directory at the same path (runs are followed through files);
- run zgm at the same Python path (``python -m zgm.probe``);
- accept a login without a password or prompt.

``zgm check --cluster macs`` tests all three. This machine's own entry (its
host name, short or long) is read directly, without ssh.
"""

from __future__ import annotations

import os
import re
import socket

from ..storage.sqlite import state_dir
from .base import Group


def configured_hosts() -> list[str]:
    """The Macs to use: ``ZGM_MACS`` if set, else ``$ZGM_HOME/macs``."""
    raw = os.environ.get("ZGM_MACS", "")
    if raw:
        hosts = re.split(r"[,\s]+", raw)
    else:
        root = state_dir()
        try:
            lines = (root / "macs").read_text(encoding="utf-8").splitlines()
        except OSError:
            return []
        hosts = [line.split("#", 1)[0] for line in lines]
    out: list[str] = []
    for host in (h.strip() for h in hosts):
        if host and host not in out:
            out.append(host)
    return out


def _same_host(a: str, b: str) -> bool:
    return a.lower() == b.lower() or a.split(".")[0].lower() == b.split(".")[0].lower()


class MacsBackend:
    name = "macs"
    interval = 60.0  # the list is static; the group's probe is what changes
    host_env: tuple[str, ...] = ()

    def __init__(self, hosts: list[str] | None = None,
                 local: str | None = None) -> None:
        self.hosts = list(hosts) if hosts is not None else configured_hosts()
        self.local = local if local is not None else socket.gethostname()

    def local_key(self) -> str | None:
        return next((h for h in self.hosts if _same_host(h, self.local)), None)

    def groups(self) -> list[Group]:
        if not self.hosts:
            raise RuntimeError("no Macs configured: set ZGM_MACS or write one host "
                               "per line to $ZGM_HOME/macs")
        return [Group(key=h, host=h, usable=True, transport="ssh") for h in self.hosts]

    def title(self, key: str) -> str:
        return f"Mac {key}"

    def columns(self, row: dict, local: bool) -> list[str]:
        state = "up" if row["state"] == "up" else "gone"
        note = row.get("note") or ("this Mac" if local else "")
        return [row["key"], state, note]

    def order(self, row: dict) -> tuple:
        return (row["state"] != "up", row["key"].lower())
