"""``zgm submit``: queue a task from the command line, from your project directory.

    zgm submit -g 2 -- train.py --lr 3e-4
    zgm submit --conda torch -- -m pkg.train --bs 64
    zgm submit --cmd "torchrun --nproc_per_node={num_gpus} train.py"
    zgm submit --config configs/rn50_adamw.yaml

The task runs where you are (the current directory) in the environment you are
in (an activated venv or conda env, else ``python`` on PATH), unless told
otherwise. It goes to the end of the queue on the GPUs where it would start
first, and a running zgm picks it up within a couple of seconds.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

from .commands import task_record
from .models import LaunchSpec, Task, format_estimate
from .services import launcher
from .services.config_repo import DEFAULT_COMMAND_TEMPLATE, ConfigRepository
from .services.gpu_discovery import make_provider
from .services.gpu_usage import GPUWatch, describe as describe_procs
from .services.lease import ExecutorLease, describe
from .services.scheduler import AssignmentService
from .services.task_store import TaskStore
from .storage.sqlite import SqliteStorage, default_db_path


class SubmitError(Exception):
    pass


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="zgm submit",
        description="Queue a task. RUN is 'script.py args...', '-m module args...', or any "
                    "other command (run by the shell). Put '--' before RUN when it starts "
                    "with '-' or takes options zgm also has.",
        epilog="Placeholders in RUN, --cmd and --env, filled in when the task starts: "
               "{config} (resolved config snapshot), {path} (config file), {name}, "
               "{num_gpus}.",
    )
    p.add_argument("run", nargs=argparse.REMAINDER, metavar="RUN",
                   help="what to run (see above)")
    p.add_argument("--cmd", metavar="STRING",
                   help="a shell command line, instead of RUN (pipes, &&, $VARS ...)")
    p.add_argument("-n", "--name", help="task name (default: from the script/module/config)")
    gpus = p.add_mutually_exclusive_group()
    gpus.add_argument("-g", "--gpus", type=int, metavar="N",
                      help="number of GPUs (default: 1, or the config's _scheduler_.gpus)")
    gpus.add_argument("--gpu-ids", metavar="0,1",
                      help="exact GPUs to queue on (default: where it would start first)")
    env = p.add_mutually_exclusive_group()
    env.add_argument("--python", metavar="PATH", help="Python interpreter (or env directory)")
    env.add_argument("--conda", metavar="ENV", help="conda env name or prefix path")
    p.add_argument("-C", "--cwd", metavar="DIR",
                   help="working directory (default: the current directory)")
    p.add_argument("-e", "--env", action="append", default=[], metavar="KEY=VALUE",
                   help="environment variable for the task (repeatable)")
    p.add_argument("--config", metavar="FILE",
                   help="create the task from a config: its _scheduler_ settings, and "
                        "{config}/{path} for RUN; without RUN, its _scheduler_.run block")
    where = p.add_mutually_exclusive_group()
    where.add_argument("--first", action="store_true", help="put it at the front of the queue")
    where.add_argument("--hold", action="store_true",
                       help="add it to the waiting queue without placing it on a GPU")
    p.add_argument("--json", action="store_true",
                   help="print the queued task as JSON (the fields of 'zgm list --json', "
                        "plus env_source, executor, autostart, warnings); on error, "
                        '{"error": ...} and exit code 2')
    p.add_argument("--no-check", dest="check", action="store_false",
                   help="don't check the working dir, script and environment now")
    p.add_argument("--db", default=str(default_db_path()),
                   help="zgm database (default: %(default)s)")
    p.add_argument("--mock-gpus", type=int, default=None, metavar="N",
                   help="use N fake GPUs instead of NVML (also: ZGM_MOCK_GPUS)")
    return p


def detect_environment() -> tuple[dict[str, str], str]:
    """The environment ``zgm submit`` runs in: (LaunchSpec fields, description)."""
    venv = os.environ.get("VIRTUAL_ENV")
    if venv:
        return {"python": venv}, f"active venv {venv}"
    conda = os.environ.get("CONDA_PREFIX")
    if conda:
        name = os.environ.get("CONDA_DEFAULT_ENV") or conda
        return {"conda": conda}, f"active conda env {name}"
    found = launcher._which_python(os.environ.get("PATH"))
    if found:
        return {"python": str(Path(found).resolve())}, "python on PATH"
    return {"python": sys.executable}, "zgm's own python: no venv/conda active, none on PATH"


def _target_from_argv(run: list[str]) -> dict:
    if run and run[0] == "--":
        run = run[1:]
    if not run:
        return {}
    if run[0] == "-m":
        if len(run) < 2:
            raise SubmitError("-m needs a module name")
        return {"module": run[1], "args": run[2:]}
    if run[0].lower().endswith(".py"):
        return {"script": run[0], "args": run[1:]}
    join = subprocess.list2cmdline if os.name == "nt" else shlex.join
    return {"command": join(run)}


def _default_name(spec: LaunchSpec) -> str:
    if spec.script:
        return Path(spec.script).stem
    if spec.module:
        return spec.module.rsplit(".", 1)[-1]
    first = spec.command.split()[0] if spec.command.split() else "task"
    return Path(first.strip("\"'")).stem or "task"


def make_task(args: argparse.Namespace) -> tuple[Task, str]:
    """The task described by the command line, and where its environment came from."""
    target = _target_from_argv(args.run)
    if args.cmd:
        if target:
            raise SubmitError("give either RUN or --cmd, not both")
        target = {"command": args.cmd}
    try:
        env_vars = dict(launcher_env(args.env))
    except ValueError as e:
        raise SubmitError(str(e)) from None

    task: Task | None = None
    if args.config:
        path = Path(args.config).expanduser().resolve()
        if not path.is_file():
            raise SubmitError(f"config not found: {args.config}")
        repo = ConfigRepository(path.parent)
        node = repo.get(str(path))
        try:
            task = repo.make_task(node, DEFAULT_COMMAND_TEMPLATE)
        except ValueError as e:
            raise SubmitError(str(e)) from None
        base = task.launch.to_dict()
    elif not target:
        raise SubmitError("nothing to run: give RUN, --cmd or --config")
    else:
        base = {}

    fields = {k: v for k, v in base.items() if k not in ("script", "module", "command", "args")}
    fields.update(target or {k: v for k, v in base.items()
                             if k in ("script", "module", "command", "args")})
    if args.cwd:
        cwd = Path(args.cwd).expanduser()
        if not cwd.is_dir():
            raise SubmitError(f"working directory not found: {args.cwd}")
        fields["cwd"] = str(cwd.resolve())
    elif not args.config or target:
        fields["cwd"] = os.getcwd()
    env_source = "config"
    if args.python or args.conda:
        fields.pop("python", None)
        fields.pop("conda", None)
        fields.update({"python": args.python} if args.python else {"conda": args.conda})
        env_source = "--python" if args.python else "--conda"
    elif not (fields.get("python") or fields.get("conda")):
        detected, env_source = detect_environment()
        fields.update(detected)
    fields["env"] = {**fields.get("env", {}), **env_vars}
    try:
        spec = LaunchSpec(**fields)
    except ValueError as e:
        raise SubmitError(str(e)) from None

    gpus = args.gpus or (len(args.gpu_ids.split(",")) if args.gpu_ids else None)
    if task is not None:
        task.launch = spec
        task.command = spec.describe()
        task.required_gpus = gpus or task.required_gpus
        task.fingerprint = repo.params_fingerprint(node, task.required_gpus)
        if args.name:
            task.name = args.name
    else:
        try:
            task = Task(name=args.name or _default_name(spec), launch=spec,
                        required_gpus=gpus or 1)
        except ValueError as e:
            raise SubmitError(str(e)) from None
    return task, env_source


def launcher_env(pairs: list[str]):
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep or not key.strip():
            raise ValueError(f"--env takes KEY=VALUE, got {pair!r}")
        yield key.strip(), value


def check(task: Task) -> None:
    """Build the process once, so a wrong path or env fails now, not at run time."""
    with tempfile.TemporaryDirectory() as tmp:
        try:
            launcher.build_process(task, tuple(range(task.required_gpus)),
                                   snapshot=Path(tmp) / "config.yaml")
        except ValueError as e:
            raise SubmitError(str(e)) from None


def submit(args: argparse.Namespace) -> str:
    """Queue the task; returns the report printed to the user."""
    return submit_result(args)[0]


def submit_result(args: argparse.Namespace) -> tuple[str, dict]:
    """Queue the task; returns (the report, the same as data for --json)."""
    task, env_source = make_task(args)
    if args.check:
        check(task)

    store = TaskStore(SqliteStorage(Path(args.db).expanduser()))
    provider = make_provider(args.mock_gpus)
    devices = provider.discover()
    svc = AssignmentService(store, [g.id for g in devices])
    svc.blocked, _ = GPUWatch(provider, devices).refresh(store)  # best_gpus avoids them
    store.add_task(task)
    lines = [f'Queued "{task.name}" (task {task.id})',
             f"  run   {task.launch.describe()}",
             f"  in    {task.launch.cwd or os.getcwd()}",
             f"  env   {task.launch.environment or 'python on PATH'} ({env_source})"]
    if task.launch.env:
        lines.append(f"  vars  {task.launch.env_text}")
    holder = ExecutorLease(Path(args.db).expanduser()).holder()
    autostart = store.storage.get_setting("autostart", "1") == "1"
    data = {"env_source": env_source, "executor": describe(holder) if holder else None,
            "autostart": autostart, "warnings": []}
    if args.hold:
        lines.append("  waiting in the queue: place it on GPUs in zgm (a)")
        return "\n".join(lines), {**task_record(store, task), **data}

    if args.gpu_ids:
        try:
            gpus = tuple(int(g) for g in args.gpu_ids.split(","))
        except ValueError:
            store.remove_task(task.id)
            raise SubmitError(f"--gpu-ids takes numbers like 0,1, got {args.gpu_ids!r}") from None
    else:
        gpus = svc.best_gpus(task)
    res = svc.assign(task.id, gpus, 0 if args.first else None)
    if not res.ok:
        store.remove_task(task.id)
        raise SubmitError(res.message)
    a = res.assignment
    position = [x.id for x in svc.planned()].index(a.id) + 1
    if a.planned_start is None and set(a.gpu_ids) & svc.blocked:
        when = "start unknown (its GPU is in use outside zgm)"
    elif a.planned_start is None:
        when = "start unknown (work ahead of it has no run history)"
    else:
        when = f"starts ~{a.planned_start:%H:%M}"
    estimate = format_estimate(store.tasks[task.id], long=True)
    lines.append(f"  GPU   {','.join(map(str, a.gpu_ids))}, queue #{position}, {when}")
    lines.append(f"  est.  {estimate}")
    for g in devices:
        if g.id in a.gpu_ids and g.external:
            warning = (f"GPU {g.id} is in use outside zgm "
                       f"({describe_procs(g.external)}): it waits until that ends")
            data["warnings"].append(warning)
            lines.append(f"  !     {warning}")
    if holder is None:
        warning = "nothing runs the queue yet: start it with 'zgm daemon start'"
        data["warnings"].append(warning)
        lines.append(f"  {warning}")
    elif not autostart:
        warning = f"{describe(holder)} runs the queue, but auto-start is paused (p in the TUI)"
        data["warnings"].append(warning)
        lines.append(f"  {warning}")
    else:
        lines.append(f"  {describe(holder)} starts it when its GPUs are free")
    return "\n".join(lines), {**task_record(store, task), **data}


def main(argv: list[str]) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report, data = submit_result(args)
    except SubmitError as e:
        if args.json:
            print(json.dumps({"error": str(e)}))
        parser.exit(2, f"zgm submit: error: {e}\n")
    if args.json:
        print(json.dumps(data, indent=2))
    else:
        enc = sys.stdout.encoding or "utf-8"
        print(report.encode(enc, errors="replace").decode(enc))
    return 0
