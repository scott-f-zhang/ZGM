"""How zgm runs a command on another host: a shell script, non-interactively.

Probes (``python -m zgm.probe``), remote runners and ``zgm check`` all go
through a transport, so a backend whose hosts are reached in another way only
needs another class here.

``ssh`` never asks for anything (``BatchMode``): a host must accept a key or
other non-interactive login. Extra options come from ``ZGM_SSH_OPTIONS``,
e.g. ``-o ProxyJump=gateway -i ~/.ssh/cluster_key``.
"""

from __future__ import annotations

import os
import shlex
import subprocess
from typing import Protocol

SSH_OPTIONS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
               "-o", "StrictHostKeyChecking=accept-new"]


class Transport(Protocol):
    host: str

    def argv(self, script: str) -> list[str]:
        """The command that runs shell ``script`` on the host."""
        ...

    def run(self, script: str, timeout: float = 60) -> subprocess.CompletedProcess:
        """Run ``script`` there; stdout/stderr as text. Raises OSError or
        subprocess.TimeoutExpired."""
        ...


class LocalTransport:
    def __init__(self, host: str = "localhost") -> None:
        self.host = host

    def argv(self, script: str) -> list[str]:
        return ["sh", "-c", script]

    def run(self, script: str, timeout: float = 60) -> subprocess.CompletedProcess:
        return subprocess.run(self.argv(script), capture_output=True, text=True,
                              timeout=timeout, stdin=subprocess.DEVNULL)


class SshTransport(LocalTransport):
    def argv(self, script: str) -> list[str]:
        extra = shlex.split(os.environ.get("ZGM_SSH_OPTIONS", ""))
        return ["ssh", *SSH_OPTIONS, *extra, self.host, script]


TRANSPORTS = {"ssh": SshTransport, "local": LocalTransport}


def transport(kind: str, host: str) -> Transport:
    try:
        return TRANSPORTS[kind](host)
    except KeyError:
        raise ValueError(f"unknown transport {kind!r} (known: {', '.join(TRANSPORTS)})") from None


def last_line(done: subprocess.CompletedProcess) -> str:
    """The most telling line of a failed command's output."""
    lines = (done.stderr or done.stdout or "").strip().splitlines()
    return lines[-1] if lines else f"exit code {done.returncode}"
