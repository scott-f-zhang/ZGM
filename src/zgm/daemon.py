"""``zgm daemon``: run the queue in the background, with or without a TUI open.

    zgm daemon start      start it in the background (log: daemon.log next to the db)
    zgm daemon status     who runs the queue, what is running and queued
    zgm daemon stop       stop it; running tasks keep running
    zgm daemon run        run it in the foreground (Ctrl+C stops it)

The daemon is the executor: it starts queued tasks as their GPUs free up and
records how they end. TUIs and ``zgm submit`` on the same database are its
clients; p/r/s in a TUI reach it through the database. Tasks run detached, so
stopping or restarting the daemon (or the machine's zgm, or a TUI) doesn't stop
them: whoever runs the queue next picks them up.

``--cluster NAME`` makes this database's queue span several hosts (see
cluster/): with ``slurm``, the GPUs of all of your running Slurm jobs are
zgm's GPUs. The daemon keeps track of them. ``--cluster none`` turns it off.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from .models import AssignmentState
from .services.executor import Executor
from .services.filelock import is_locked
from .services.gpu_discovery import describe_gpus
from .services.gpu_usage import GPUWatch, describe as describe_procs
from .services.lease import ExecutorLease, describe
from .services.scheduler import AssignmentService
from .cluster import BACKENDS
from .cluster.core import ClusterProvider, ClusterWatch, gpu_source
from .services.task_store import TaskStore
from .storage.sqlite import SqliteStorage, default_db_path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="zgm daemon", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", metavar="{start,status,stop,run}")

    def common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--db", default=str(default_db_path()),
                        help="zgm database (default: %(default)s)")

    def runner_options(sp: argparse.ArgumentParser) -> None:
        common(sp)
        sp.add_argument("--runtime", choices=["local", "ray", "mock"], default="local",
                        help="how tasks run (default: %(default)s); ray and mock runs "
                             "can't be followed after the daemon exits")
        sp.add_argument("--ray-address", default=None,
                        help="Ray cluster address (default: start a local one)")
        sp.add_argument("--log-dir", default=None,
                        help="task logs (default: logs/ next to the database)")
        sp.add_argument("--mock-gpus", type=int, default=None, metavar="N",
                        help="use N fake GPUs instead of real ones (also: ZGM_MOCK_GPUS)")
        sp.add_argument("--interval", type=float, default=2.0, metavar="SECONDS",
                        help="how often to check runs and the queue (default: %(default)s)")
        sp.add_argument("--no-autostart", dest="autostart", action="store_const", const=False,
                        default=None, help="pause auto-start (p in a TUI turns it on)")
        sp.add_argument("--cluster", choices=[*BACKENDS, "none"], default=None,
                        help="span several hosts: 'slurm' = the GPUs of all your running "
                             "Slurm jobs; 'none' = this machine only (stays set for "
                             "this database; run 'zgm check' first)")

    runner_options(sub.add_parser("start", help="start the daemon in the background"))
    runner_options(sub.add_parser("run", help="run the daemon in the foreground"))
    common(sub.add_parser("status", help="show who runs the queue and what is running"))
    common(sub.add_parser("stop", help="stop the daemon (running tasks keep running)"))
    return p


def _db(args) -> Path:
    return Path(args.db).expanduser()


def _log_dir(args) -> Path:
    return Path(args.log_dir).expanduser() if args.log_dir else _db(args).parent / "logs"


def daemon_log(db: Path) -> Path:
    return db.parent / "daemon.log"


# run ---------------------------------------------------------------------------------
def run(args) -> int:
    db = _db(args)
    store = TaskStore(SqliteStorage(db))
    if args.cluster is not None:
        store.storage.set_setting("cluster", "" if args.cluster == "none" else args.cluster)
    provider = gpu_source(store.storage, args.mock_gpus)
    cluster = provider if isinstance(provider, ClusterProvider) else None
    if cluster and args.runtime != "local":
        print(f"zgm daemon: error: cluster mode needs the local runtime, not {args.runtime}",
              file=sys.stderr)
        return 2
    gpus = provider.discover()
    svc = AssignmentService(store, [g.id for g in gpus],
                            node_of=cluster.node_of(gpus) if cluster else None)
    watch = GPUWatch(provider, gpus)
    if args.runtime == "local":
        from .runtime.local import LocalRuntime

        runtime = LocalRuntime(_log_dir(args), locate=cluster.locate if cluster else None)
    elif args.runtime == "ray":
        from .runtime.ray_adapter import RayRuntime

        runtime = RayRuntime(address=args.ray_address)
    else:
        from .runtime.mock import MockRuntime

        runtime = MockRuntime()

    def log(message: str, severity: str = "information") -> None:
        mark = {"warning": "!", "error": "x"}.get(severity, "-")
        print(f"{datetime.now():%Y-%m-%d %H:%M:%S} {mark} {message}", flush=True)

    stop = threading.Event()

    def on_signal(signum, frame) -> None:
        stop.set()

    for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), on_signal)

    lease = ExecutorLease(db)
    log_dir = str(_log_dir(args)) if args.runtime == "local" else None
    if not lease.acquire("daemon", runtime.name, log_dir):
        log(f"waiting: {describe(lease.holder())} runs the queue; taking over when it exits")
        while not lease.acquire("daemon", runtime.name, log_dir):
            if stop.wait(args.interval):
                return 0
    groups = ClusterWatch(store, cluster.backend, cluster.local) if cluster else None

    def follow_cluster(wait: bool = False) -> None:
        """Cluster mode: update the groups, then the GPUs if they came or went."""
        if groups is None:
            return
        for message, severity in groups.refresh(svc, wait=wait):
            log(message, severity)
        cluster.invalidate()
        if provider.sync(gpus):
            svc.set_gpus([g.id for g in gpus], cluster.node_of(gpus))
            svc.recompute()
            log(f"GPUs now: {describe_gpus(provider, gpus)}")

    try:
        executor = Executor(store, svc, runtime, notify=log)
        if args.autostart is not None:
            executor.autostart = args.autostart
        follow_cluster(wait=True)
        log(f"zgm daemon (pid {os.getpid()}) runs the queue in {db}: "
            f"{describe_gpus(provider, gpus)}, "
            f"{runtime.name} runtime, auto-start {'on' if executor.autostart else 'paused'}")
        executor.adopt()
        if executor.runs:
            log(f"following {len(executor.runs)} task(s) already running")
        while not stop.is_set():
            store.sync()
            follow_cluster()
            svc.blocked, changes = watch.refresh(store)
            for message in changes:
                log(message, "warning")
            executor.tick()
            stop.wait(args.interval)
    finally:
        if groups is not None:
            groups.close()
        lease.release()
        log("stopped; running tasks keep running")
    return 0


# start / stop / status -------------------------------------------------------------------
def start(args) -> int:
    db = _db(args)
    lease = ExecutorLease(db)
    holder = lease.holder()
    if holder and holder.get("role") == "daemon":
        print(f"already running: {describe(holder)}")
        return 0
    argv = [sys.executable, "-m", "zgm", "daemon", "run", "--db", str(db),
            "--runtime", args.runtime, "--interval", str(args.interval)]
    if args.log_dir:
        argv += ["--log-dir", args.log_dir]
    if args.mock_gpus is not None:
        argv += ["--mock-gpus", str(args.mock_gpus)]
    if args.ray_address:
        argv += ["--ray-address", args.ray_address]
    if args.autostart is False:
        argv.append("--no-autostart")
    if args.cluster is not None:
        argv += ["--cluster", args.cluster]
    from .runtime.local import spawn_detached

    db.parent.mkdir(parents=True, exist_ok=True)
    launched = datetime.now().replace(microsecond=0)
    proc = spawn_detached(argv, daemon_log(db))
    if holder:  # a TUI runs the queue: the daemon waits for it
        print(f"started (pid {proc.pid}); it takes over when {describe(holder)} exits")
        print(f"log: {daemon_log(db)}")
        return 0
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        holder = lease.holder()
        # not by pid: on Windows a venv's python.exe starts the real one as a child
        if (holder and holder.get("role") == "daemon"
                and datetime.fromisoformat(holder.get("since", "1970-01-01")) >= launched):
            print(f"started: {describe(holder)}")
            print(f"log: {daemon_log(db)}")
            return 0
        if proc.poll() is not None:
            break
        time.sleep(0.1)
    print(f"zgm daemon: error: it did not start; see {daemon_log(db)}", file=sys.stderr)
    return 1


def stop(args) -> int:
    lease = ExecutorLease(_db(args))
    holder = lease.holder()
    if not holder:
        print("not running")
        return 0
    if holder.get("role") != "daemon":
        print(f"zgm daemon: error: {describe(holder)} runs the queue, not a daemon; "
              "close that TUI instead", file=sys.stderr)
        return 1
    pid = int(holder["pid"])
    if os.name == "nt":  # no /T: the tasks are not the daemon's to take down
        subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True)
    else:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 15
    while is_locked(lease.lock_path) and time.monotonic() < deadline:
        time.sleep(0.1)
    if is_locked(lease.lock_path):
        print(f"zgm daemon: error: pid {pid} is still running", file=sys.stderr)
        return 1
    lease.info_path.unlink(missing_ok=True)  # a killed daemon couldn't remove it
    print(f"stopped (pid {pid}); running tasks keep running")
    return 0


def status(args) -> int:
    db = _db(args)
    holder = ExecutorLease(db).holder()
    print(f"queue:      {db}")
    print(f"executor:   {describe(holder)}" + (f" since {holder.get('since')}"
                                               if holder and holder.get("since") else ""))
    if not db.exists():
        from .services.gpu_discovery import make_provider

        provider = make_provider(getattr(args, "mock_gpus", None))
        print(f"gpus:       {describe_gpus(provider, provider.discover())}")
        return 0
    store = TaskStore(SqliteStorage(db))
    provider = gpu_source(store.storage, getattr(args, "mock_gpus", None))
    gpus = provider.discover()
    print(f"gpus:       {describe_gpus(provider, gpus)}")
    if isinstance(provider, ClusterProvider):
        seen = provider.seen_at()
        print(f"cluster:    {provider.backend.name}"
              + (f" (as of {datetime.fromisoformat(seen):%H:%M:%S})"
                 if seen else " (nothing seen yet: does a daemon run?)"))
        for line in provider.lines(indent="  "):
            print(line)
    autostart = store.storage.get_setting("autostart", "1") == "1"
    print(f"auto-start: {'on' if autostart else 'paused'}")
    autofill = store.storage.get_setting("autofill", "1") == "1"
    marked = sum(1 for t in store.tasks.values() if t.autofill and t.status.value == "pending")
    print(f"auto-fill:  {'on' if autofill else 'off'} ({marked} waiting task"
          f"{'s' if marked != 1 else ''} marked for it)")
    running = [a for a in store.assignments.values() if a.state == AssignmentState.RUNNING]
    planned = [a for a in store.assignments.values() if a.state == AssignmentState.PLANNED]
    print(f"running:    {len(running)}")
    for a in sorted(running, key=lambda a: a.actual_start or datetime.min):
        task = store.tasks.get(a.task_id)
        ids = ",".join(map(str, a.display_gpu_ids))
        since = f"{a.actual_start:%H:%M}" if a.actual_start else "?"
        print(f"  {task.name if task else a.task_id}  GPU {ids}  since {since}"
              + (f"  log {a.log_path}" if a.log_path else ""))
    print(f"queued:     {len(planned)}")
    GPUWatch(provider, gpus).refresh(store)
    for g in gpus:
        if g.external:
            print(f"GPU {g.id} in use outside zgm: {describe_procs(g.external)}")
        elif not g.processes_visible:
            print(f"GPU {g.id}: its processes can't be listed")
    if holder and holder.get("log_dir"):
        print(f"task logs:  {holder['log_dir']}")
    if holder and holder.get("role") == "daemon":
        print(f"daemon log: {daemon_log(db)}")
    return 0


def main(argv: list[str]) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 2
    return {"run": run, "start": start, "stop": stop, "status": status}[args.command](args)
