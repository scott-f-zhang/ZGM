"""Turn a task's LaunchSpec into a concrete process: argv, cwd and environment.

Environments:

- ``python``: the interpreter runs scripts/modules, and its directory goes first
  on PATH, so a ``command`` like ``torchrun ...`` finds the same environment's tools.
- ``conda``: the env's *activated* variables are captured once with ``conda run``
  (activation hooks included) and cached; the process then starts directly with
  them. No ``conda run`` wrapper stays in between, so output is live and the
  shell command needs no extra quoting.

The process also gets ``CUDA_VISIBLE_DEVICES`` for the planned GPUs (so the
user's code just uses ``cuda:0..n-1``), ``ZGM_NUM_GPUS``, ``ZGM_GPU_IDS``,
``ZGM_TASK_ID``, ``ZGM_TASK_NAME`` and ``PYTHONUNBUFFERED=1``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ..models import LaunchSpec, Task
from ..models.launch import fill
from . import config_loader as cl

_ENV_MARKER = "__ZGM_ENV__"
_DUMP_ENV = f"import os,json;print('{_ENV_MARKER}'+json.dumps(dict(os.environ)))"
_conda_env_cache: dict[tuple[str, str], dict[str, str]] = {}


@dataclass
class Process:
    args: list[str] | str  # a string means "run with the shell"
    cwd: str
    env: dict[str, str]  # the complete environment

    @property
    def shell(self) -> bool:
        return isinstance(self.args, str)

    def overrides(self) -> dict[str, str]:
        """Variables that differ from this process's own environment."""
        return {k: v for k, v in self.env.items() if os.environ.get(k) != v}

    def display(self) -> str:
        return self.args if isinstance(self.args, str) else subprocess.list2cmdline(self.args)


def find_conda() -> str:
    """The conda executable: $CONDA_EXE, PATH, or a usual install location."""
    candidates = [os.environ.get("CONDA_EXE"), shutil.which("conda")]
    home = Path.home()
    for root in ("anaconda3", "miniconda3", "miniforge3", "mambaforge"):
        base = home / root
        candidates += [base / "Scripts" / "conda.exe", base / "condabin" / "conda.bat",
                       base / "bin" / "conda"]
    for c in candidates:
        if c and Path(c).is_file():
            return str(c)
    raise ValueError("conda not found: put it on PATH or set CONDA_EXE")


def _conda_target(env: str) -> tuple[str, str]:
    if "/" in env or "\\" in env or env.startswith("~"):
        return "-p", str(Path(env).expanduser())
    return "-n", env


def conda_environment(env: str) -> dict[str, str]:
    """The activated environment variables of conda env ``env`` (name or prefix)."""
    key = _conda_target(env)
    if key not in _conda_env_cache:
        try:
            proc = subprocess.run(
                [find_conda(), "run", *key, "python", "-c", _DUMP_ENV],
                capture_output=True, text=True, timeout=180,
            )
        except subprocess.TimeoutExpired:
            raise ValueError(f"conda env {env!r}: activation timed out") from None
        line = next((x for x in proc.stdout.splitlines() if x.startswith(_ENV_MARKER)), None)
        if proc.returncode != 0 or line is None:
            detail = (proc.stderr or proc.stdout).strip().splitlines()[-1:] or ["no output"]
            raise ValueError(f"conda env {env!r} could not be activated: {detail[0]}")
        _conda_env_cache[key] = json.loads(line[len(_ENV_MARKER):])
    return dict(_conda_env_cache[key])


def _prepend_path(env: dict[str, str], *dirs: Path) -> None:
    key = next((k for k in env if k.upper() == "PATH"), "PATH")
    parts = [str(d) for d in dirs if d.is_dir()]
    env[key] = os.pathsep.join([*parts, env.get(key, "")])


def _which_python(path_var: str | None) -> str | None:
    """python/python3 on ``path_var``, skipping the Microsoft Store stubs in
    WindowsApps (they only print "Python was not found")."""
    for d in (path_var or "").split(os.pathsep):
        if not d or Path(d).name.lower() == "windowsapps":
            continue
        found = shutil.which("python", path=d) or shutil.which("python3", path=d)
        if found:
            return found
    return None


def _resolve_python(spec: LaunchSpec, cwd: Path, env: dict[str, str]) -> str:
    if spec.python:
        path = Path(spec.python).expanduser()
        if not path.is_absolute():
            path = cwd / path
        if path.is_dir():  # a venv/conda prefix instead of the interpreter itself
            exe = "python.exe" if os.name == "nt" else "python"
            for sub in ("", "Scripts", "bin"):
                if (path / sub / exe).is_file():
                    path = path / sub / exe
                    break
        if not path.is_file():
            raise ValueError(f"python not found: {path}")
        return str(path)
    path_var = next((v for k, v in env.items() if k.upper() == "PATH"), None)
    found = _which_python(path_var)
    if spec.conda and not found:
        raise ValueError(f"conda env {spec.conda!r} has no python")
    return found or sys.executable


def write_config_snapshot(config_path: str | Path, dest: Path) -> Path:
    """The config with inheritance resolved, without ``_base_``/``_scheduler_``:
    what the user's code reads, and a record of exactly what the run used."""
    try:
        resolved, _ = cl.resolve(Path(config_path))
    except Exception as e:
        raise ValueError(f"config {Path(config_path).name}: {e}") from None
    resolved.pop(cl.SCHEDULER_KEY, None)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(cl.dump_yaml(cl.to_plain(resolved)), encoding="utf-8")
    return dest


def build_process(task: Task, gpu_ids: Sequence[int] | None,
                  snapshot: Path | None = None) -> Process:
    """``gpu_ids=None`` leaves CUDA_VISIBLE_DEVICES alone (Ray sets it itself).
    ``snapshot``: where to write the resolved config if the task uses ``{config}``
    (default: a temporary file)."""
    spec = task.launch
    values = {"path": task.config_path or "", "name": task.name,
              "num_gpus": task.required_gpus}
    if spec.uses("config"):
        if not task.config_path:
            raise ValueError("{config} needs a task created from a config")
        if snapshot is None:
            fd, name = tempfile.mkstemp(prefix="zgm-", suffix=".yaml")
            os.close(fd)
            snapshot = Path(name)
        values["config"] = str(write_config_snapshot(task.config_path, snapshot))
    cwd = Path(spec.cwd).expanduser() if spec.cwd else Path.cwd()
    if not cwd.is_dir():
        raise ValueError(f"working directory not found: {cwd}")

    env = conda_environment(spec.conda) if spec.conda else dict(os.environ)
    python = _resolve_python(spec, cwd, env)
    if spec.python:
        exe_dir = Path(python).parent
        _prepend_path(env, exe_dir, exe_dir / "Scripts")
    # number GPUs like nvidia-smi/NVML (PCI bus order), not CUDA's default
    # fastest-first, so "GPU 2" is the same card for zgm, nvidia-smi and the task
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # output goes to a log file, where Python would otherwise use the locale's
    # encoding (GBK on Chinese Windows) and fail on the first non-ASCII character
    env["PYTHONIOENCODING"] = "utf-8"
    env.update({k: fill(v, values) for k, v in spec.env.items()})
    env.update({
        "PYTHONUNBUFFERED": "1",
        "ZGM_TASK_ID": task.id,
        "ZGM_TASK_NAME": task.name,
        "ZGM_NUM_GPUS": str(task.required_gpus),
    })
    if gpu_ids is not None:
        ids = ",".join(str(g) for g in gpu_ids)
        env["CUDA_VISIBLE_DEVICES"] = env["ZGM_GPU_IDS"] = ids

    if spec.command:
        return Process(fill(spec.command, values), str(cwd), env)
    if spec.script:
        script = Path(spec.script).expanduser()
        if not (cwd / script).is_file():
            raise ValueError(f"script not found: {cwd / script}")
        head = [str(script)]
    else:
        head = ["-m", spec.module]
    return Process([python, *head, *(fill(a, values) for a in spec.args)], str(cwd), env)
