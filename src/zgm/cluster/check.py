"""``zgm check``: can this setup run zgm, and across which hosts?

Run it before turning cluster mode on, and when a host's GPUs don't show up.
It checks, and says how to fix what fails:

- the state directory (``ZGM_HOME`` or ``~/.local/share/zgm``): yours, not
  writable by others, and, on storage other users share, a path of your own;
- this machine's GPUs;
- with a backend (``--cluster NAME``, or the one the database uses): that it
  can list its groups;
- for every host zgm would use (and every ``--host``):
  - that it is reached without a password or prompt;
  - that it sees the state directory at the same path (a file written here is
    read there, and the other way round);
  - that zgm runs there with the same Python (``python -m zgm.probe``) and sees GPUs.

Exit code 0 when nothing failed (warnings allowed), 1 otherwise.
"""

from __future__ import annotations

import argparse
import getpass
import os
import secrets
import shlex
import stat
import subprocess
import sys
import time
from pathlib import Path

from ..storage.sqlite import SqliteStorage, default_db_path
from . import BACKENDS, backend as load_backend, configured
from .core import probe_host, short_name
from .transport import TRANSPORTS, last_line, transport

OK, WARN, FAIL = "ok", "warn", "FAIL"


class Report:
    def __init__(self) -> None:
        self.failed = False

    def line(self, level: str, what: str, text: str, hint: str = "", indent: str = "  ") -> None:
        self.failed |= level == FAIL
        print(f"{indent}[{level:>4}] {what:<10} {text}")
        for h in hint.splitlines():
            print(f"{indent}{'':7} {'':<10} {h}")


def check_state_dir(r: Report, root: Path) -> None:
    try:
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        st = root.stat()
    except OSError as e:
        r.line(FAIL, "state dir", f"{root}: {e}", "set ZGM_HOME to a directory you can write")
        return
    problems, level = [], OK
    if hasattr(os, "getuid") and st.st_uid != os.getuid():
        problems.append("it belongs to another user")
        level = FAIL
    if st.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        problems.append("other users can write to it")
        level = FAIL
    elif st.st_mode & (stat.S_IRGRP | stat.S_IROTH):
        problems.append("other users can read it (task logs, pids)")
        level = WARN if level == OK else level
    hints = []
    if any("other users" in p for p in problems):
        hints.append(f"chmod 700 {shlex.quote(str(root))}")
    user = getpass.getuser()
    home = Path.home().resolve()
    resolved = root.resolve()
    if home not in (resolved, *resolved.parents) and user not in resolved.parts:
        problems.append(f"it is outside your home and its path doesn't name you ({user})")
        level = WARN if level == OK else level
        hints.append(f"on storage several users share, use a directory of your own, "
                     f"e.g. ZGM_HOME={root.parent / user / root.name}")
    hint = "\n".join(hints)
    r.line(level, "state dir", f"{root}" + (f": {'; '.join(problems)}" if problems else
                                            " (yours, private)"), hint)


def check_gpus_here(r: Report) -> None:
    from ..services.gpu_discovery import MockGPUProvider, describe_gpus, make_provider

    provider = make_provider()
    gpus = provider.discover()
    level = WARN if isinstance(provider, MockGPUProvider) else OK
    r.line(level, "GPUs here", describe_gpus(provider, gpus))


def check_host(r: Report, title: str, t, root: Path, python: str = sys.executable) -> None:
    print(f"  {title}")
    host = t.host
    started = time.monotonic()
    try:
        done = t.run("true", timeout=30)
        reached, detail = done.returncode == 0, last_line(done)
    except (OSError, subprocess.TimeoutExpired) as e:
        reached, detail = False, str(e)
    if not reached:
        hint = "it must accept a login without a password or a prompt:"
        if isinstance(t, TRANSPORTS["ssh"]):
            hint += (f"\n  ssh -o BatchMode=yes {host} true      (must print nothing and work)"
                     f"\n  add your public key to ~/.ssh/authorized_keys there (with a shared"
                     f"\n  home: there too), or pass options in ZGM_SSH_OPTIONS"
                     f"\n  on Slurm, ssh to a node only works while you have a job on it")
        r.line(FAIL, "reach", f"{host}: {detail}", hint, indent="    ")
        return
    r.line(OK, "reach", f"{host} ({time.monotonic() - started:.1f} s)", indent="    ")

    nonce, reply = secrets.token_hex(8), secrets.token_hex(8)
    probe = root / f".zgm-check-{nonce}"
    answer = probe.with_name(probe.name + ".reply")
    try:
        probe.write_text(nonce, encoding="utf-8")
        q = shlex.quote(str(probe))
        done = t.run(f"cat {q} && printf %s {reply} > {shlex.quote(str(answer))}", timeout=30)
        seen = done.returncode == 0 and done.stdout.strip() == nonce
        back = False
        deadline = time.monotonic() + 10  # a shared filesystem may show it a bit later
        while seen and not back and time.monotonic() < deadline:
            try:
                back = answer.read_text(encoding="utf-8") == reply
            except OSError:
                time.sleep(0.5)
        if seen and back:
            r.line(OK, "shared", f"{host} sees {root}", indent="    ")
        else:
            why = "can't read" if not seen else "its writes don't reach this machine in"
            r.line(FAIL, "shared", f"{host} {why} {root}",
                   "put ZGM_HOME on a filesystem every host mounts at the same path\n"
                   "(e.g. a shared home or project directory); runs on other hosts\n"
                   "are followed through the files there", indent="    ")
            return
    except (OSError, subprocess.TimeoutExpired) as e:
        r.line(FAIL, "shared", f"{root}: {e}", indent="    ")
        return
    finally:
        probe.unlink(missing_ok=True)
        answer.unlink(missing_ok=True)

    try:
        data = probe_host(t, python)
    except RuntimeError as e:
        r.line(FAIL, "zgm", f"{python} -m zgm.probe: {e}",
               "zgm must run there at the same path: install it in a directory\n"
               "every host shares (e.g. a venv in your shared home)", indent="    ")
        return
    gpus = data.get("gpus", [])
    names = ", ".join(sorted({short_name(g["name"]) for g in gpus})) or "none"
    r.line(OK if gpus else WARN, "zgm", f"runs there; GPUs: {len(gpus)} x {names}",
           "" if gpus else "no GPUs visible there", indent="    ")


def run_check(db: Path, cluster: str | None, hosts: list[str], kind: str) -> int:
    r = Report()
    root = db.parent
    print("zgm check")
    check_state_dir(r, root)
    check_gpus_here(r)
    if cluster is None and db.exists():
        cluster = configured(SqliteStorage(db))
    targets = []
    if cluster:
        b = load_backend(cluster)
        try:
            groups = b.groups()
        except RuntimeError as e:
            r.line(FAIL, cluster, str(e))
            groups = []
        else:
            usable = [g for g in groups if g.usable and g.key != b.local_key()]
            r.line(OK, cluster, f"{len(groups)} group(s); {len(usable)} usable on other hosts"
                   + ("" if usable else " (none to check now)"))
            targets += [(f"{g.host} ({b.title(g.key)})", transport(g.transport, g.host))
                        for g in usable]
    targets += [(h, transport(kind, h)) for h in hosts]
    for title, t in targets:
        check_host(r, title, t, root)
    if not cluster and not hosts:
        print("  this machine only: pass --cluster NAME or --host HOST to check other hosts")
    print("result: " + ("problems found (see FAIL above)" if r.failed else "ready"))
    return 1 if r.failed else 0


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="zgm check", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", default=str(default_db_path()),
                   help="zgm database; its directory is the state dir (default: %(default)s)")
    p.add_argument("--cluster", choices=list(BACKENDS), default=None,
                   help="check this backend's hosts (default: the database's, if any)")
    p.add_argument("--host", action="append", default=[], metavar="HOST",
                   help="also check this host (repeatable)")
    p.add_argument("--transport", choices=list(TRANSPORTS), default="ssh",
                   help="how --host is reached (default: %(default)s)")
    args = p.parse_args(argv)
    return run_check(Path(args.db).expanduser(), args.cluster, args.host, args.transport)
