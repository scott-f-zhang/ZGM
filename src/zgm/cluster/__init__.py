"""One zgm queue across several machines (optional; see cluster/base.py).

The database setting ``cluster`` names the backend in use (``zgm daemon start
--cluster NAME``); unset, zgm uses this machine's GPUs only and nothing in
this package runs.
"""

from __future__ import annotations

import importlib

from .base import Backend, Group

# name -> "module:Class"; a new backend is one more entry
BACKENDS = {
    "slurm": "zgm.cluster.slurm:SlurmBackend",
}

__all__ = ["BACKENDS", "Backend", "Group", "backend", "configured"]


def backend(name: str, **kw) -> Backend:
    try:
        module, cls = BACKENDS[name].split(":")
    except KeyError:
        raise ValueError(f"unknown cluster backend {name!r} "
                         f"(known: {', '.join(BACKENDS)})") from None
    return getattr(importlib.import_module(module), cls)(**kw)


def configured(storage) -> str | None:
    """The backend this database uses, or None (this machine only)."""
    return storage.get_setting("cluster") or None
