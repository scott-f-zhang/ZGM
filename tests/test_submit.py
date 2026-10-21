"""zgm submit: queue tasks from the command line; a running TUI picks them up."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

from conftest import make_task
from test_dispatch import FakeRuntime
from zgm.__main__ import main as zgm_main
from zgm.app import ZGMApp
from zgm.models import AssignmentState, TaskStatus
from zgm.services.task_store import TaskStore
from zgm.storage.sqlite import SqliteStorage
from zgm.submit import SubmitError, build_parser, make_task as submit_task, submit

EXAMPLES = Path(__file__).parent.parent / "examples"
PY = sys.executable


@pytest.fixture
def proj(tmp_path, monkeypatch):
    """A project directory to submit from, with a clean environment."""
    d = tmp_path / "proj"
    d.mkdir()
    (d / "train.py").write_text("print('hi')\n", encoding="utf-8")
    monkeypatch.chdir(d)
    for var in ("VIRTUAL_ENV", "CONDA_PREFIX", "CONDA_DEFAULT_ENV"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("ZGM_MOCK_GPUS", "4")
    return d


def args(*argv: str, db: Path | None = None):
    extra = ["--db", str(db)] if db else []
    return build_parser().parse_args([*extra, *argv])


def queued(db: Path):
    store = TaskStore(SqliteStorage(db))
    return store, sorted((a for a in store.assignments.values()
                          if a.state == AssignmentState.PLANNED), key=lambda a: a.seq)


# what to run ------------------------------------------------------------------------
def test_script_with_args_from_where_you_are(proj):
    task, _ = submit_task(args("-g", "2", "train.py", "--lr", "3e-4", "--gpus", "8"))
    spec = task.launch
    assert (spec.script, spec.args) == ("train.py", ["--lr", "3e-4", "--gpus", "8"])
    assert spec.cwd == str(proj)
    assert (task.name, task.required_gpus) == ("train", 2)


def test_module_after_double_dash(proj):
    task, _ = submit_task(args("--", "-m", "pkg.train", "--bs", "64"))
    assert (task.launch.module, task.launch.args, task.name) == ("pkg.train", ["--bs", "64"],
                                                                 "train")


def test_other_programs_become_shell_commands(proj):
    task, _ = submit_task(args("--", "torchrun", "--nproc_per_node=2", "train.py"))
    assert task.launch.command == "torchrun --nproc_per_node=2 train.py"
    assert task.name == "torchrun"
    task, _ = submit_task(args("--cmd", "echo a && echo b", "-n", "two"))
    assert (task.launch.command, task.name) == ("echo a && echo b", "two")


@pytest.mark.parametrize("argv,match", [
    ((), "nothing to run"),
    (("--cmd", "echo", "train.py"), "either RUN or --cmd"),
    (("-e", "NOVALUE", "train.py"), "KEY=VALUE"),
    (("-C", "/definitely/missing", "train.py"), "working directory"),
])
def test_usage_errors(proj, argv, match):
    with pytest.raises(SubmitError, match=match):
        submit_task(args(*argv))


# which environment ----------------------------------------------------------------
def test_uses_the_active_venv(proj, monkeypatch):
    monkeypatch.setenv("VIRTUAL_ENV", "/home/me/proj/.venv")
    task, source = submit_task(args("train.py"))
    assert task.launch.python == "/home/me/proj/.venv"
    assert "venv" in source


def test_uses_the_active_conda_env(proj, monkeypatch):
    monkeypatch.setenv("CONDA_PREFIX", "/opt/conda/envs/torch")
    monkeypatch.setenv("CONDA_DEFAULT_ENV", "torch")
    task, source = submit_task(args("train.py"))
    assert (task.launch.conda, task.launch.python) == ("/opt/conda/envs/torch", None)
    assert "torch" in source


def test_explicit_environment_wins(proj, monkeypatch):
    monkeypatch.setenv("VIRTUAL_ENV", "/somewhere/.venv")
    task, _ = submit_task(args("--conda", "torch", "train.py"))
    assert (task.launch.conda, task.launch.python) == ("torch", None)
    task, _ = submit_task(args("--python", PY, "train.py"))
    assert task.launch.python == PY


def test_env_vars(proj):
    task, _ = submit_task(args("-e", "A=1", "-e", "B=x=y", "train.py"))
    assert task.launch.env == {"A": "1", "B": "x=y"}


# from a config ----------------------------------------------------------------------
def test_config_alone_uses_its_run_block(proj):
    cfg = EXAMPLES / "configs" / "resnet50_adamw_bs512.yaml"
    task, _ = submit_task(args("--config", str(cfg)))
    assert task.name == "resnet50_adamw_bs512"
    assert task.config_path == str(cfg.resolve())
    assert task.launch.script == "train.py"
    assert task.launch.cwd == str((EXAMPLES / "project").resolve())  # from the config
    assert task.fingerprint.startswith("cfg:")


def test_config_with_run_and_more_gpus(proj):
    cfg = EXAMPLES / "configs" / "resnet50_base.yaml"
    one, _ = submit_task(args("--config", str(cfg), "train.py", "--config", "{config}"))
    assert one.launch.cwd == str(proj)  # RUN given: run it from here
    assert one.launch.args == ["--config", "{config}"]
    four, _ = submit_task(args("--config", str(cfg), "-g", "4", "train.py"))
    assert four.required_gpus == 4
    assert four.fingerprint != one.fingerprint  # same parameters on 4 GPUs: other workload


# checks before queueing --------------------------------------------------------------
def test_check_catches_mistakes_before_queueing(proj, tmp_path):
    db = tmp_path / "s.db"
    with pytest.raises(SubmitError, match="script not found"):
        submit(args("--python", PY, "missing.py", db=db))
    with pytest.raises(SubmitError, match="python not found"):
        submit(args("--python", "nope/python", "train.py", db=db))
    store, planned = queued(db)
    assert not store.tasks
    submit(args("--no-check", "--python", PY, "missing.py", db=db))
    assert len(TaskStore(SqliteStorage(db)).tasks) == 1


# where it goes in the queue ------------------------------------------------------------
def test_best_gpus_prefers_the_earliest_start(store, svc):
    long = make_task(store, "long", hours=5)
    svc.assign(long.id, [0])
    short = make_task(store, "short", hours=1)
    for g in (1, 2, 3):
        svc.assign(make_task(store, f"s{g}", hours=1).id, [g])
    svc.assign(short.id, [1])  # GPU 1 now has 2h queued; 2 and 3 have 1h
    new = make_task(store, "new", gpus=1)
    assert svc.best_gpus(new) == (2,)
    pair = make_task(store, "pair", gpus=2)
    assert svc.best_gpus(pair) == (2, 3)


def test_best_gpus_known_start_beats_unknown(store, svc):
    for g in range(3):
        svc.assign(make_task(store, f"x{g}", hours=None).id, [g])  # no history
    svc.assign(make_task(store, "y", hours=8).id, [3])
    assert svc.best_gpus(make_task(store, "new")) == (3,)


def test_submit_queues_at_the_end_or_first(proj, tmp_path):
    db = tmp_path / "s.db"
    for name in ("a", "b"):
        submit(args("--python", PY, "-n", name, "--gpu-ids", "0", "train.py", db=db))
    report = submit(args("--python", PY, "-n", "c", "--first", "--gpu-ids", "0", "train.py",
                         db=db))
    assert "queue #1" in report
    store, planned = queued(db)
    assert [store.tasks[a.task_id].name for a in planned] == ["c", "a", "b"]


def test_submit_hold_and_gpu_id_errors(proj, tmp_path):
    db = tmp_path / "s.db"
    report = submit(args("--python", PY, "--hold", "train.py", db=db))
    assert "waiting in the queue" in report
    store, planned = queued(db)
    (t,) = store.tasks.values()
    assert t.status == TaskStatus.PENDING and not planned
    with pytest.raises(SubmitError, match="unknown GPU"):
        submit(args("--python", PY, "--gpu-ids", "7", "train.py", db=db))
    with pytest.raises(SubmitError, match="numbers"):
        submit(args("--python", PY, "--gpu-ids", "a,b", "train.py", db=db))
    assert len(TaskStore(SqliteStorage(db)).tasks) == 1  # failed submits left nothing



def test_submit_autofill_waits_marked_for_it(proj, tmp_path):
    db = tmp_path / "s.db"
    report = submit(args("--python", PY, "--autofill", "train.py", db=db))
    assert "waiting for auto-fill" in report
    store, planned = queued(db)
    (t,) = store.tasks.values()
    assert t.autofill and t.status == TaskStatus.PENDING and not planned
    store.storage.set_setting("autofill", "0")
    report = submit(args("--python", PY, "--autofill", "train.py", db=db))
    assert "auto-fill is off" in report
    submit(args("--python", PY, "--autofill", "--priority", "2", "train.py", db=db))
    store, _ = queued(db)
    assert sorted(t.priority for t in store.tasks.values()) == [0, 0, 2]

def test_cli_entry_point(proj, tmp_path, capsys):
    db = tmp_path / "s.db"
    with pytest.raises(SystemExit) as ok:
        zgm_main(["submit", "--db", str(db), "--python", PY, "train.py"])
    assert ok.value.code == 0
    assert 'Queued "train"' in capsys.readouterr().out
    with pytest.raises(SystemExit) as bad:
        zgm_main(["submit", "--db", str(db), "--python", PY, "missing.py"])
    assert bad.value.code == 2
    assert "script not found" in capsys.readouterr().err


# a running TUI picks submitted tasks up ---------------------------------------------------
async def test_running_tui_picks_up_and_starts_submitted_task(proj, tmp_path):
    db = tmp_path / "s.db"
    config_dir = tmp_path / "configs"
    shutil.copytree(EXAMPLES / "configs", config_dir)
    rt = FakeRuntime()
    app = ZGMApp(storage=SqliteStorage(db), mock_gpus=4, config_dir=config_dir,
                 stats_interval=0, runtime=rt)
    async with app.run_test(size=(140, 42)) as pilot:
        await pilot.pause()
        submit(args("--python", PY, "-n", "from-cli", "--gpu-ids", "1", "train.py", db=db))
        assert not app.store.tasks  # not seen yet
        app.tick()
        await pilot.pause()
        (t,) = app.store.tasks.values()
        assert t.name == "from-cli" and t.status == TaskStatus.RUNNING
        assert rt.started == ["from-cli"]
        # and the TUI's own writes don't make it reload
        assert not app.store.sync()
