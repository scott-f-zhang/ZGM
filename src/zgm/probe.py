"""``python -m zgm.probe``: this node's GPUs and their compute processes, as JSON.

Cluster mode runs it on the other hosts, through their transport (see
cluster/core.py). Each process carries its ``ZGM_TASK_ID``, if it has one, so
the executor can tell zgm's tasks from other programs.

    {"host": "node07", "gpus": [{"index": 0, "uuid": ..., "name": ..., "memory_total": ...,
      "memory_used": ..., "utilization": ..., "processes_visible": true,
      "processes": [{"pid": 4242, "name": "python", "used_memory": ..., "zgm_task": null}]}]}

On failure: {"host": ..., "error": "..."} and exit code 1.
"""

from __future__ import annotations

import json
import os
import socket
import sys

import psutil

from .services.gpu_discovery import MockGPUProvider, make_provider


def task_of(pid: int) -> str | None:
    """The ``ZGM_TASK_ID`` in the process's environment, or else its nearest
    ancestor's: a program that sets its process title (vLLM's ``VLLM::EngineCore``,
    via setproctitle) can blank what /proc shows of its own environment."""
    try:
        proc = psutil.Process(pid)
        while proc is not None:
            task = proc.environ().get("ZGM_TASK_ID")
            if task:
                return task
            proc = proc.parent()
    except (psutil.Error, OSError):  # gone, or another user's process
        pass
    return None


def probe(provider=None) -> dict:
    """``provider``: an NVML/Apple/mock provider to read (default: a new one, which
    must see real GPUs unless ZGM_MOCK_GPUS asks for fake ones)."""
    if provider is None:
        provider = make_provider()
        if isinstance(provider, MockGPUProvider) and not os.environ.get("ZGM_MOCK_GPUS"):
            raise RuntimeError("no GPU is usable here (no NVIDIA driver/NVML and no "
                               "Apple GPU visible)")
    gpus = []
    for g in provider.discover():
        entry = {"index": g.id, "uuid": g.uuid, "name": g.name, "memory_total": g.memory_total,
                 "memory_used": g.memory_used, "utilization": g.utilization,
                 "processes_visible": True, "processes": []}
        try:
            procs = provider.processes(g.id)
        except Exception:
            entry["processes_visible"] = False
            procs = []
        entry["processes"] = [{"pid": p.pid, "name": p.name, "used_memory": p.used_memory,
                               "zgm_task": task_of(p.pid)} for p in procs]
        gpus.append(entry)
    return {"host": socket.gethostname(), "gpus": gpus}


def main() -> int:
    try:
        out, code = probe(), 0
    except Exception as e:
        out, code = {"host": socket.gethostname(), "error": str(e)}, 1
    print(json.dumps(out))
    return code


if __name__ == "__main__":
    sys.exit(main())
