"""LaunchSpec: how to start a task's process — where, in which environment, what.

A task runs exactly one of:

- ``script``: a Python file, run by the task's interpreter (``python train.py *args``);
- ``module``: a Python module (``python -m pkg.train *args``);
- ``command``: a shell command line (``torchrun ... train.py``), run by the shell.

The environment is either ``python`` (path to an interpreter, e.g. a venv's) or
``conda`` (an env name, or a path to an env prefix); empty means ``python`` on PATH.
``args``/``command``/``env`` may contain ``{config}`` (a snapshot of the task's
config with inheritance resolved, written when the task starts), ``{path}`` (the
config file itself, ``_base_`` and all), ``{name}`` (the task name) and
``{num_gpus}``; they are substituted when the task starts.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from typing import Any

PLACEHOLDERS = ("config", "path", "name", "num_gpus")
CONDA_PREFIX = "conda:"


def split_args(text: str) -> list[str]:
    """Split like a POSIX shell, but keep backslashes literal (Windows paths)."""
    lex = shlex.shlex(text, posix=True)
    lex.whitespace_split = True
    lex.escape = ""
    lex.commenters = ""
    return list(lex)


def join_args(args: list[str]) -> str:
    return shlex.join(args)


def fill(text: str, values: dict[str, Any]) -> str:
    """Substitute the known ``{placeholders}`` only; other braces stay as they are."""
    for key in PLACEHOLDERS:
        if key in values:
            text = text.replace("{" + key + "}", str(values[key]))
    return text


@dataclass
class LaunchSpec:
    cwd: str | None = None
    python: str | None = None
    conda: str | None = None
    script: str | None = None
    module: str | None = None
    command: str | None = None
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        targets = [k for k in ("script", "module", "command") if getattr(self, k)]
        if len(targets) != 1:
            raise ValueError("give exactly one of script, module or command")
        if self.python and self.conda:
            raise ValueError("give either python or conda, not both")
        if self.command and self.args:
            raise ValueError("args go with script/module; put them in the command instead")

    # text forms used by the forms and the details pane ----------------------
    @property
    def environment(self) -> str:
        """'' (python on PATH), 'conda:NAME', or an interpreter path."""
        if self.conda:
            return CONDA_PREFIX + self.conda
        return self.python or ""

    @property
    def target(self) -> str:
        """What to run, as typed: 'train.py --lr 3e-4', '-m pkg.train', or a command."""
        if self.command:
            return self.command
        head = [self.script] if self.script else ["-m", self.module]
        return join_args([*head, *self.args])

    def uses(self, placeholder: str) -> bool:
        token = "{" + placeholder + "}"
        texts = [self.command or "", *self.args, *self.env.values()]
        return any(token in t for t in texts)

    @property
    def env_text(self) -> str:
        return "; ".join(f"{k}={v}" for k, v in self.env.items())

    def describe(self) -> str:
        """One line for cards and the task's ``command`` field."""
        return self.command or "python " + self.target

    @classmethod
    def from_text(cls, target: str, environment: str = "", cwd: str = "",
                  env: str = "") -> LaunchSpec:
        """Parse the New/Edit task form.

        ``target``: 'train.py ...' or '-m mod ...' run with the environment's
        Python; anything else is a shell command.
        """
        target = target.strip()
        if not target:
            raise ValueError("what to run is required")
        tokens = split_args(target)
        kw: dict[str, Any] = {}
        if tokens[0] == "-m":
            if len(tokens) < 2:
                raise ValueError("-m needs a module name")
            kw.update(module=tokens[1], args=tokens[2:])
        elif tokens[0].lower().endswith(".py"):
            kw.update(script=tokens[0], args=tokens[1:])
        else:
            kw.update(command=target)
        environment = environment.strip()
        if environment.startswith(CONDA_PREFIX):
            name = environment[len(CONDA_PREFIX):].strip()
            if not name:
                raise ValueError("conda: needs an env name or path")
            kw["conda"] = name
        elif environment:
            kw["python"] = environment
        return cls(cwd=cwd.strip() or None, env=parse_env(env), **kw)

    # persistence / YAML ------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key in ("cwd", "python", "conda", "script", "module", "command"):
            if getattr(self, key):
                out[key] = getattr(self, key)
        if self.args:
            out["args"] = list(self.args)
        if self.env:
            out["env"] = dict(self.env)
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LaunchSpec:
        """From storage or a config's ``_scheduler_.run`` block."""
        if not isinstance(data, dict):
            raise ValueError("run must be a mapping")
        unknown = set(data) - {"cwd", "python", "conda", "script", "module", "command",
                               "args", "env"}
        if unknown:
            raise ValueError(f"unknown run key(s): {', '.join(sorted(unknown))}")
        args = data.get("args") or []
        if isinstance(args, str):
            args = split_args(args)
        env = data.get("env") or {}
        if not isinstance(env, dict):
            raise ValueError("run.env must be a mapping")
        text = {k: (str(data[k]) if data.get(k) is not None else None)
                for k in ("cwd", "python", "conda", "script", "module", "command")}
        return cls(args=[str(a) for a in args],
                   env={str(k): str(v) for k, v in env.items()}, **text)


def parse_env(text: str) -> dict[str, str]:
    """'A=1; B=two' -> {'A': '1', 'B': 'two'}."""
    out: dict[str, str] = {}
    for part in text.split(";"):
        part = part.strip()
        if not part:
            continue
        key, sep, value = part.partition("=")
        if not sep or not key.strip():
            raise ValueError(f"environment variables are KEY=VALUE; got {part!r}")
        out[key.strip()] = value.strip()
    return out
