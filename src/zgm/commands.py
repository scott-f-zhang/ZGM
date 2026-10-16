"""``zgm list``, ``zgm stop`` and ``zgm logs``: the queue from the command line.

    zgm list [--json] [--status running,planned]
    zgm stop ID [--remove] [--json]
    zgm logs ID [-f] [-n LINES] [--path]

ID is a task id, a unique prefix of one, or a unique task name. These work
whether or not anything runs the queue: they read and change the database (and,
to stop a run, the run's own files), and the executor sees the change on its
next check.

Exit codes: 0 done; 1 nothing to do (``stop`` on a task that is neither running
nor queued; ``logs`` for a task that never ran; ``logs -f`` whose run didn't
complete); 2 bad arguments or unknown task.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from .models import Assignment, AssignmentState, Task
from .runtime.local import read_run
from .services.task_store import TaskStore
from .storage.sqlite import SqliteStorage, default_db_path

FINAL = {"completed", "failed", "cancelled"}


class CommandError(Exception):
    pass


# shared --------------------------------------------------------------------------------------
def open_store(db: str | Path) -> TaskStore | None:
    """The database's tasks, or None if there is no database yet (nothing created)."""
    path = Path(db).expanduser()
    if not path.exists():
        return None
    return TaskStore(SqliteStorage(path))


def latest_run(store: TaskStore, task: Task) -> Assignment | None:
    """The task's current placement, else its most recent run."""
    runs = store.assignments_for_task(task.id)
    return store.active_assignment(task.id) or max(
        runs, key=lambda r: r.actual_start or datetime.min, default=None)


def _iso(t: datetime | None) -> str | None:
    return t.isoformat(timespec="seconds") if t else None


def task_record(store: TaskStore, task: Task) -> dict:
    """Everything about a task, as ``zgm list --json`` and ``zgm submit --json`` print it."""
    a = latest_run(store, task)
    run = read_run(Path(a.log_path).with_suffix("")) if a and a.log_path else None
    planned = sorted((x for x in store.assignments.values()
                      if x.state == AssignmentState.PLANNED), key=lambda x: x.seq)
    position = next((i + 1 for i, x in enumerate(planned) if a and x.id == a.id), None)
    spec = task.launch
    return {
        "id": task.id,
        "name": task.name,
        "status": task.status.value,
        "last_run": a.state.value if a else None,
        "gpus": list(a.display_gpu_ids) if a else [],
        "required_gpus": task.required_gpus,
        "queue_position": position,
        "planned_start": _iso(a.planned_start) if a and a.state.value == "planned" else None,
        "estimate_seconds": (int(task.estimated_duration.total_seconds())
                             if task.estimated_duration else None),
        "started": _iso(a.actual_start) if a else None,
        "ended": _iso(a.actual_end) if a else None,
        "exit_code": (run or {}).get("exit_code"),
        "log": a.log_path if a else None,
        "command": spec.describe(),
        "cwd": spec.cwd,
        "environment": spec.environment or None,
        "config": task.config_path,
        "created": _iso(task.created_at),
    }


def resolve(store: TaskStore | None, ref: str) -> Task:
    """A task by id, unique id prefix, or unique name."""
    tasks = list(store.tasks.values()) if store else []
    for match in ([t for t in tasks if t.id == ref],
                  [t for t in tasks if t.id.startswith(ref)],
                  [t for t in tasks if t.name == ref]):
        if len(match) == 1:
            return match[0]
        if len(match) > 1:
            names = ", ".join(f"{t.id} ({t.name})" for t in match[:5])
            raise CommandError(f"{ref!r} matches more than one task: {names}")
    raise CommandError(f"no task {ref!r}")


def _emit(text: str) -> None:
    """Print, replacing what the console's encoding can't show (e.g. GBK)."""
    enc = sys.stdout.encoding or "utf-8"
    sys.stdout.write(text.encode(enc, errors="replace").decode(enc))
    sys.stdout.flush()


# list ---------------------------------------------------------------------------------------
def cmd_list(args) -> int:
    store = open_store(args.db)
    records = [task_record(store, t) for t in
               sorted(store.tasks.values(), key=lambda t: t.created_at)] if store else []
    if args.status:
        wanted = {s.strip() for s in args.status.split(",") if s.strip()}
        records = [r for r in records if r["status"] in wanted]
    if args.json:
        print(json.dumps(records, indent=2))
        return 0
    if not records:
        _emit("no tasks\n")
        return 0
    # running, then the queue in order, then waiting; then finished, latest first
    order = {"running": 0, "planned": 1, "pending": 2}
    active = sorted((r for r in records if r["status"] in order),
                    key=lambda r: (order[r["status"]], r["queue_position"] or 0))
    done = sorted((r for r in records if r["status"] not in order),
                  key=lambda r: r["ended"] or r["created"], reverse=True)
    records = active + done
    rows = [("ID", "STATUS", "GPUS", "#", "STARTED", "EXIT", "NAME")]
    for r in records:
        started = r["started"][5:16].replace("T", " ") if r["started"] else "-"
        rows.append((r["id"], r["status"], ",".join(map(str, r["gpus"])) or "-",
                     str(r["queue_position"] or "-"), started,
                     "-" if r["exit_code"] is None else str(r["exit_code"]), r["name"]))
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]) - 1)]
    for row in rows:
        _emit("  ".join(c.ljust(w) for c, w in zip(row, widths)) + "  " + row[-1] + "\n")
    return 0


# stop ---------------------------------------------------------------------------------------
def cmd_stop(args) -> int:
    from .runtime.local import LocalRuntime
    from .services.executor import Executor
    from .services.scheduler import AssignmentService

    store = open_store(args.db)
    task = resolve(store, args.task)
    svc = AssignmentService(store, None)
    a = store.active_assignment(task.id)
    if a is not None and a.state == AssignmentState.RUNNING:
        # the run's process tree is killed; the task is kept unless --remove
        Executor(store, svc, LocalRuntime(".")).stop(task, requeue=not args.remove)
        action = "stopped and removed" if args.remove else "stopped"
    elif a is not None:
        svc.unassign(a.id)
        if args.remove:
            store.remove_task(task.id)
        action = "unqueued and removed" if args.remove else "unqueued"
    elif args.remove:
        store.remove_task(task.id)
        action = "removed"
    else:
        message = f"{task.name} is {task.status.value}: neither running nor queued"
        if args.json:
            print(json.dumps({"id": task.id, "name": task.name, "action": None,
                              "error": message}))
        else:
            print(f"zgm stop: {message} (--remove deletes it)", file=sys.stderr)
        return 1
    if args.json:
        record = task_record(store, task) if task.id in store.tasks else {"id": task.id}
        print(json.dumps({**record, "name": task.name, "action": action}, indent=2))
    else:
        _emit(f"{action} {task.name} ({task.id})\n")
    return 0


# logs ---------------------------------------------------------------------------------------
def cmd_logs(args) -> int:
    store = open_store(args.db)
    task = resolve(store, args.task)
    a = latest_run(store, task)
    while args.follow and a is not None and a.state == AssignmentState.PLANNED:
        time.sleep(0.5)  # still queued: wait for it to start
        store.sync()
        a = latest_run(store, task)
    if a is None or not a.log_path:
        state = a.state.value if a else task.status.value
        raise _NoLog(f"{task.name} has no log yet ({state})")
    path = Path(a.log_path)
    if args.path:
        print(path)
        return 0
    if not path.exists():
        raise _NoLog(f"log file is gone: {path}")
    out = sys.stdout.buffer
    with open(path, "rb") as f:
        data = f.read()
        if args.lines is not None:
            data = b"\n".join(data.splitlines()[-args.lines:]) + (b"\n" if data else b"")
        out.write(data)
        out.flush()
        if not args.follow:
            return 0
        # follow until the run is over and its last output is out
        while True:
            chunk = f.read()
            if chunk:
                out.write(chunk)
                out.flush()
                continue
            store.sync()
            run = store.assignments.get(a.id)
            if run is None or run.state != AssignmentState.RUNNING:
                rest = f.read()
                if rest:
                    out.write(rest)
                    out.flush()
                return 0 if run is not None and run.state == AssignmentState.COMPLETED else 1
            time.sleep(0.5)


class _NoLog(Exception):
    pass


# entry --------------------------------------------------------------------------------------
def build_parser(prog: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=f"zgm {prog}", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", default=str(default_db_path()),
                   help="zgm database (default: %(default)s)")
    if prog == "list":
        p.add_argument("--json", action="store_true",
                       help="print a JSON array, one object per task")
        p.add_argument("--status", metavar="S[,S]",
                       help="only these statuses: pending, planned, running, completed, "
                            "failed, cancelled")
    elif prog == "stop":
        p.add_argument("task", metavar="ID", help="task id, unique id prefix, or name")
        p.add_argument("--remove", action="store_true",
                       help="also delete the task (and its history) from the database")
        p.add_argument("--json", action="store_true", help="print the result as JSON")
    else:
        p.add_argument("task", metavar="ID", help="task id, unique id prefix, or name")
        p.add_argument("-f", "--follow", action="store_true",
                       help="wait for a queued task to start, print until its run ends, then "
                            "exit 0 if it completed, 1 if it failed or was stopped")
        p.add_argument("-n", "--lines", type=int, metavar="N",
                       help="only the last N lines (before following)")
        p.add_argument("--path", action="store_true", help="print the log's path only")
    return p


def main(prog: str, argv: list[str]) -> int:
    parser = build_parser(prog)
    args = parser.parse_args(argv)
    handler = {"list": cmd_list, "stop": cmd_stop, "logs": cmd_logs}[prog]
    try:
        return handler(args)
    except CommandError as e:
        parser.exit(2, f"zgm {prog}: error: {e}\n")
    except _NoLog as e:
        print(f"zgm {prog}: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
