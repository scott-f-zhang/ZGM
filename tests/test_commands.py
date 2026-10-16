"""zgm list / stop / logs and zgm submit --json: the commands agents use."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from zgm.commands import CommandError, main as command_main, resolve
from zgm.models import Task
from zgm.services.scheduler import AssignmentService
from zgm.services.task_store import TaskStore
from zgm.storage.sqlite import SqliteStorage

PY = sys.executable
ROOT = Path(__file__).parent.parent

JOB = """import sys, time
for i in range(int(sys.argv[2])):
    print("step", i, "\\u2713 \\u5b8c\\u6210", flush=True)  # non-ASCII: must not crash on GBK
    time.sleep(float(sys.argv[3]))
sys.exit(int(sys.argv[1]))
"""


def zgm(*args, check=False, timeout=60):
    return subprocess.run([PY, "-m", "zgm", *args], capture_output=True, cwd=ROOT,
                          timeout=timeout, check=check)


def out(result) -> str:
    return result.stdout.decode("utf-8", errors="replace")


# without a daemon -----------------------------------------------------------------------
@pytest.fixture
def db(tmp_path):
    return tmp_path / "s.db"


def seed(db, *names):
    store = TaskStore(SqliteStorage(db))
    svc = AssignmentService(store, [0, 1])
    tasks = []
    for name in names:
        t = store.add_task(Task(name, f"echo {name}"))
        tasks.append(t)
    return store, svc, tasks


def test_list_without_a_database_creates_nothing(tmp_path, capsys):
    db = tmp_path / "none.db"
    assert command_main("list", ["--db", str(db), "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == []
    assert command_main("list", ["--db", str(db)]) == 0
    assert "no tasks" in capsys.readouterr().out
    assert not db.exists()


def test_resolve_by_id_prefix_or_name(db):
    store, _, (a, b) = seed(db, "alpha", "beta")
    assert resolve(store, a.id) is a
    assert resolve(store, a.id[:5]) is a
    assert resolve(store, "beta") is b
    with pytest.raises(CommandError, match="no task"):
        resolve(store, "gamma")
    twin = store.add_task(Task("alpha", "echo again"))
    with pytest.raises(CommandError, match="more than one"):
        resolve(store, "alpha")
    assert resolve(store, twin.id) is twin


def test_list_json_fields_and_filter(db, capsys):
    store, svc, (a, b) = seed(db, "alpha", "beta")
    svc.assign(a.id, [1])
    assert command_main("list", ["--db", str(db), "--json"]) == 0
    rows = {r["name"]: r for r in json.loads(capsys.readouterr().out)}
    assert rows["alpha"]["status"] == "planned" and rows["alpha"]["gpus"] == [1]
    assert rows["alpha"]["queue_position"] == 1 and rows["alpha"]["last_run"] == "planned"
    assert rows["beta"]["status"] == "pending" and rows["beta"]["last_run"] is None
    assert set(rows["alpha"]) >= {"id", "exit_code", "log", "started", "ended", "command",
                                  "cwd", "environment", "estimate_seconds", "required_gpus"}
    command_main("list", ["--db", str(db), "--json", "--status", "pending"])
    assert [r["name"] for r in json.loads(capsys.readouterr().out)] == ["beta"]


def test_stop_unqueues_or_removes_without_a_run(db, capsys):
    store, svc, (a, b) = seed(db, "alpha", "beta")
    svc.assign(a.id, [0])
    assert command_main("stop", ["--db", str(db), "alpha", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["action"] == "unqueued"
    store.sync()
    assert store.tasks[a.id].status.value == "pending"
    assert command_main("stop", ["--db", str(db), "alpha"]) == 1  # nothing left to stop
    assert "neither running nor queued" in capsys.readouterr().err
    assert command_main("stop", ["--db", str(db), "beta", "--remove"]) == 0
    assert "removed beta" in capsys.readouterr().out
    assert command_main("logs", ["--db", str(db), "alpha"]) == 1
    assert "no log yet" in capsys.readouterr().err
    with pytest.raises(SystemExit) as e:
        command_main("stop", ["--db", str(db), "nope"])
    assert e.value.code == 2


def test_submit_json_error(db, tmp_path):
    r = zgm("submit", "--db", str(db), "--json", "--python", PY, "-C", str(tmp_path),
            "--", "missing.py")
    assert r.returncode == 2
    assert "script not found" in json.loads(out(r))["error"]


# the whole loop, as an agent runs it ------------------------------------------------------------
@pytest.fixture
def daemon(db):
    proc = subprocess.Popen([PY, "-m", "zgm", "daemon", "run", "--db", str(db),
                             "--mock-gpus", "4", "--interval", "0.2"], cwd=ROOT,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    yield db
    zgm("daemon", "stop", "--db", str(db))
    proc.kill()
    proc.wait()


def submit(db, tmp_path, name, *job, gpus="0"):
    r = zgm("submit", "--db", str(db), "--json", "--python", PY, "-C", str(tmp_path),
            "-n", name, "--gpu-ids", gpus, "--", "job.py", *job, check=True)
    return json.loads(out(r))


def records(db):
    return {r["name"]: r for r in json.loads(out(zgm("list", "--db", str(db), "--json")))}


def test_submit_wait_read_stop(daemon, tmp_path):
    db = daemon
    (tmp_path / "job.py").write_text(JOB, encoding="utf-8")
    good = submit(db, tmp_path, "good", "0", "3", "0.2")
    assert good["status"] in ("planned", "running") and good["gpus"] == [0]
    assert good["executor"].startswith("zgm daemon") and good["warnings"] == []
    bad = submit(db, tmp_path, "bad", "3", "1", "0")  # queued behind good on GPU 0

    r = zgm("logs", "--db", str(db), "-f", good["id"], timeout=60)  # waits for the end
    assert r.returncode == 0
    text = out(r)
    assert text.count("step") == 3 and "完成" in text and "exited with code 0" in text
    assert text.count("# zgm: good started") == 1  # ran once
    assert zgm("logs", "--db", str(db), "-f", "bad", timeout=60).returncode == 1

    rows = records(db)
    assert (rows["good"]["status"], rows["good"]["exit_code"]) == ("completed", 0)
    assert (rows["bad"]["status"], rows["bad"]["exit_code"]) == ("failed", 3)
    assert Path(rows["good"]["log"]).exists()
    assert zgm("logs", "--db", str(db), "--path", "good").stdout.strip()

    long = submit(db, tmp_path, "long", "0", "600", "0.1", gpus="1")
    deadline = time.time() + 30
    while records(db)["long"]["status"] != "running" and time.time() < deadline:
        time.sleep(0.2)
    r = zgm("stop", "--db", str(db), long["id"][:6], "--json")
    assert r.returncode == 0 and json.loads(out(r))["action"] == "stopped"
    rows = records(db)
    assert (rows["long"]["status"], rows["long"]["last_run"]) == ("pending", "cancelled")
    assert zgm("stop", "--db", str(db), "long").returncode == 1

    table = out(zgm("list", "--db", str(db)))
    assert table.splitlines()[0].split()[:2] == ["ID", "STATUS"]
    assert table.splitlines()[1].split()[1] == "pending"  # waiting before finished
