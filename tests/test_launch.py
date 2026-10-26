"""Launch specs, process building, and the local runtime (real child processes)."""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime
from pathlib import Path

import pytest

from zgm.models import Assignment, LaunchSpec, Task, command_fingerprint
from zgm.runtime.base import RunState
from zgm.runtime.local import LocalRuntime, read_run
from zgm.services import launcher
from zgm.services.config_repo import ConfigRepository
from zgm.storage.sqlite import SqliteStorage
from zgm.widgets.dialogs import _task_fields

PY = sys.executable


def pid_alive(pid: int) -> bool:
    if sys.platform == "win32":
        import subprocess

        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                             capture_output=True, text=True).stdout
        return str(pid) in out
    import psutil

    try:  # a killed child of this process stays a zombie until it is reaped
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


# LaunchSpec -----------------------------------------------------------------
@pytest.mark.parametrize("target,kind,value,args", [
    ("train.py --lr 3e-4", "script", "train.py", ["--lr", "3e-4"]),
    ("-m pkg.train --bs 64", "module", "pkg.train", ["--bs", "64"]),
    ("torchrun --nproc_per_node=2 train.py", "command",
     "torchrun --nproc_per_node=2 train.py", []),
    (r"C:\proj\train.py --out 'a b'", "script", r"C:\proj\train.py", ["--out", "a b"]),
])
def test_from_text_kinds(target, kind, value, args):
    spec = LaunchSpec.from_text(target)
    assert getattr(spec, kind) == value
    assert spec.args == args


def test_from_text_environment_and_env_vars():
    spec = LaunchSpec.from_text("train.py", "conda:torch", "", "A=1; B = two words ")
    assert (spec.conda, spec.python) == ("torch", None)
    assert spec.env == {"A": "1", "B": "two words"}
    assert spec.environment == "conda:torch"
    spec = LaunchSpec.from_text("train.py", ".venv/bin/python")
    assert (spec.conda, spec.python) == (None, ".venv/bin/python")


@pytest.mark.parametrize("target,environment,env", [
    ("", "", ""), ("-m", "", ""), ("train.py", "conda:", ""), ("train.py", "", "NOVALUE"),
])
def test_from_text_errors(target, environment, env):
    with pytest.raises(ValueError):
        LaunchSpec.from_text(target, environment, "", env)


def test_spec_validation():
    with pytest.raises(ValueError):
        LaunchSpec(script="a.py", command="echo")
    with pytest.raises(ValueError):
        LaunchSpec(script="a.py", python="py", conda="env")
    with pytest.raises(ValueError):
        LaunchSpec(command="echo", args=["x"])
    with pytest.raises(ValueError):
        LaunchSpec.from_dict({"script": "a.py", "interpreter": "x"})


def test_target_round_trips_through_the_form():
    spec = LaunchSpec(script="train.py", args=["--name", "a b", "--q", "it's"])
    assert LaunchSpec.from_text(spec.target).args == spec.args


def test_legacy_command_keeps_its_fingerprint():
    t = Task("old", "python train.py", 2)
    assert t.launch == LaunchSpec(command="python train.py")
    assert t.fingerprint == command_fingerprint("python train.py", 2)


def test_fingerprint_depends_on_what_and_where_not_on_environment():
    def fp(**kw):
        return Task("t", launch=LaunchSpec(**kw)).fingerprint

    assert fp(script="a.py", cwd="/p") == fp(script="a.py", cwd="/p", conda="x")
    assert fp(script="a.py", cwd="/p") != fp(script="a.py", cwd="/q")
    assert fp(script="a.py", cwd="/p") != fp(script="a.py", cwd="/p", args=["--lr", "1"])


def test_storage_round_trip(tmp_path):
    db = SqliteStorage(tmp_path / "s.db")
    spec = LaunchSpec(cwd="/p", conda="torch", script="train.py", args=["--x", "1"],
                      env={"A": "1"})
    t = Task("job", launch=spec, required_gpus=2)
    db.save_task(t)
    (loaded,) = SqliteStorage(tmp_path / "s.db").load_tasks()
    assert loaded.launch == spec
    assert loaded.command == "python train.py --x 1"
    assert loaded.fingerprint == t.fingerprint


def test_form_fields(tmp_path):
    name, spec, gpus = _task_fields({
        "name": "sweep", "gpus": "2", "run": "-m pkg.train --lr 1",
        "environment": "conda:torch", "cwd": str(tmp_path), "env": "WANDB_MODE=offline",
    })
    assert (name, gpus, spec.module, spec.conda) == ("sweep", 2, "pkg.train", "torch")
    assert spec.cwd == str(tmp_path.resolve())
    with pytest.raises(ValueError, match="working directory"):
        _task_fields({"name": "x", "gpus": "1", "run": "a.py", "environment": "",
                      "cwd": str(tmp_path / "missing"), "env": ""})


# configs: _scheduler_.run ---------------------------------------------------------
def test_config_run_block_is_inherited(tmp_path):
    (tmp_path / "proj").mkdir()
    (tmp_path / "base.yaml").write_text(
        "_scheduler_:\n  gpus: 1\n  run:\n    cwd: proj\n    conda: torch\n"
        "    script: train.py\n    args: [--config, '{path}']\n    env: {WANDB_MODE: offline}\n"
        "lr: 0.1\n", encoding="utf-8")
    (tmp_path / "child.yaml").write_text(
        "_base_: base.yaml\n_scheduler_:\n  gpus: 2\nlr: 0.2\n", encoding="utf-8")
    repo = ConfigRepository(tmp_path)
    task = repo.make_task(repo.get("child"))
    spec = task.launch
    assert task.required_gpus == 2
    assert spec.cwd == str((tmp_path / "proj").resolve())
    assert (spec.conda, spec.script, spec.args) == ("torch", "train.py", ["--config", "{path}"])
    assert spec.env == {"WANDB_MODE": "offline"}
    # the run block is not a parameter: it doesn't change the workload's identity
    assert task.fingerprint == repo.params_fingerprint(repo.get("child"), 2)


def test_config_run_block_errors(tmp_path):
    (tmp_path / "bad.yaml").write_text(
        "_scheduler_:\n  run:\n    script: a.py\n    command: echo\n", encoding="utf-8")
    repo = ConfigRepository(tmp_path)
    with pytest.raises(ValueError, match=r"_scheduler_\.run"):
        repo.make_task(repo.get("bad"))


# launcher ---------------------------------------------------------------------
def test_build_process_script(tmp_path):
    (tmp_path / "train.py").write_text("", encoding="utf-8")
    t = Task("job", launch=LaunchSpec(cwd=str(tmp_path), python=PY, script="train.py",
                                      args=["--name", "{name}", "--n", "{num_gpus}"],
                                      env={"RUN": "{name}"}), required_gpus=2)
    proc = launcher.build_process(t, (2, 3))
    assert proc.args == [PY, "train.py", "--name", "job", "--n", "2"]
    assert proc.cwd == str(tmp_path)
    assert proc.env["CUDA_VISIBLE_DEVICES"] == "2,3"
    assert proc.env["ZGM_NUM_GPUS"] == "2"
    assert proc.env["RUN"] == "job"
    assert proc.env["PYTHONUNBUFFERED"] == "1"
    # the interpreter's directory comes first on PATH, for tools like torchrun
    path_key = next(k for k in proc.env if k.upper() == "PATH")
    assert proc.env[path_key].startswith(str(Path(PY).parent))


def test_build_process_leaves_cuda_alone_for_ray(tmp_path):
    t = Task("job", launch=LaunchSpec(cwd=str(tmp_path), command="echo hi"))
    proc = launcher.build_process(t, None)
    assert proc.shell and proc.args == "echo hi"
    assert "ZGM_GPU_IDS" not in proc.env


def test_build_process_python_may_be_an_env_dir(tmp_path):
    env_dir = Path(PY).parent
    if env_dir.name.lower() in ("scripts", "bin"):
        env_dir = env_dir.parent
    t = Task("job", launch=LaunchSpec(cwd=str(tmp_path), python=str(env_dir), module="json"))
    assert Path(launcher.build_process(t, (0,)).args[0]).samefile(PY)


@pytest.mark.parametrize("spec,match", [
    (dict(cwd="/definitely/missing", command="echo"), "working directory"),
    (dict(script="missing.py"), "script not found"),
    (dict(python="nope/python", module="json"), "python not found"),
])
def test_build_process_errors(tmp_path, spec, match):
    spec.setdefault("cwd", str(tmp_path))
    with pytest.raises(ValueError, match=match):
        launcher.build_process(Task("job", launch=LaunchSpec(**spec)), (0,))


def test_build_process_conda_uses_activated_env(tmp_path, monkeypatch):
    env_bin = tmp_path / "env"
    env_bin.mkdir()
    exe = env_bin / ("python.exe" if sys.platform == "win32" else "python")
    exe.write_text("", encoding="utf-8")
    exe.chmod(0o755)
    seen = []

    def fake(name):
        seen.append(name)
        return {"PATH": str(env_bin), "CONDA_DEFAULT_ENV": name}

    monkeypatch.setattr(launcher, "conda_environment", fake)
    t = Task("job", launch=LaunchSpec(cwd=str(tmp_path), conda="torch", module="pkg.train"))
    proc = launcher.build_process(t, (1,))
    assert seen == ["torch"]
    assert Path(proc.args[0]) == exe
    assert proc.env["CONDA_DEFAULT_ENV"] == "torch"


def test_real_conda_activation():
    try:
        launcher.find_conda()
    except ValueError:
        pytest.skip("conda is not installed")
    env = launcher.conda_environment("base")
    assert env.get("CONDA_DEFAULT_ENV") == "base"
    with pytest.raises(ValueError, match="could not be activated"):
        launcher.conda_environment("zgm-no-such-env-xyz")


# local runtime ------------------------------------------------------------------
def _assignment(task: Task, gpus=(1,)) -> Assignment:
    now = datetime.now()
    return Assignment(task.id, gpus, now, None)


def _wait(rt, handle, timeout=30):
    deadline = time.time() + timeout
    while time.time() < deadline:
        state = rt.status(handle)
        if state not in (RunState.PENDING, RunState.RUNNING):
            return state
        time.sleep(0.1)
    raise TimeoutError


def test_local_runtime_runs_script_on_planned_gpus(tmp_path):
    (tmp_path / "train.py").write_text(
        "import os, sys\n"
        "print('cwd', os.getcwd())\n"
        "print('gpus', os.environ['CUDA_VISIBLE_DEVICES'], os.environ['ZGM_NUM_GPUS'])\n"
        "print('args', sys.argv[1:])\n", encoding="utf-8")
    rt = LocalRuntime(tmp_path / "logs")
    t = Task("job", launch=LaunchSpec(cwd=str(tmp_path), python=PY, script="train.py",
                                      args=["--lr", "3e-4"]), required_gpus=2)
    a = _assignment(t, (2, 3))
    handle = rt.submit(t, a)
    assert a.actual_gpu_ids == (2, 3)
    assert _wait(rt, handle) == RunState.SUCCEEDED
    log = rt.log_path(a.id).read_text(encoding="utf-8")
    assert f"cwd {tmp_path}" in log
    assert "gpus 2,3 2" in log
    assert "args ['--lr', '3e-4']" in log
    assert "exited with code 0" in log
    assert rt.result(handle)["gpu_ids"] == [2, 3]


def test_local_runtime_shell_command_and_failure(tmp_path):
    rt = LocalRuntime(tmp_path / "logs")
    t = Task("fail", launch=LaunchSpec(cwd=str(tmp_path), command="echo before && exit 3"))
    a = _assignment(t)
    handle = rt.submit(t, a)
    assert _wait(rt, handle) == RunState.FAILED
    log = rt.log_path(a.id).read_text(encoding="utf-8")
    assert "before" in log and "exited with code 3" in log


def test_local_runtime_cancel_kills_the_process_tree(tmp_path):
    (tmp_path / "child.py").write_text("import time\ntime.sleep(60)\n", encoding="utf-8")
    # a shell -> python chain: stopping must not leave the python process behind
    cmd = f'"{PY}" child.py' if sys.platform == "win32" else f"'{PY}' child.py"
    t = Task("long", launch=LaunchSpec(cwd=str(tmp_path), command=cmd))
    rt = LocalRuntime(tmp_path / "logs")
    a = _assignment(t)
    handle = rt.submit(t, a)
    time.sleep(1.0)
    info = read_run(handle.ref)
    assert pid_alive(info["runner_pid"]) and pid_alive(info["child_pid"])
    rt.cancel(handle)
    assert rt.status(handle) == RunState.CANCELLED
    info = read_run(handle.ref)
    assert not pid_alive(info["runner_pid"]) and not pid_alive(info["child_pid"])
    assert "stopped" in rt.log_path(a.id).read_text(encoding="utf-8")


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
def test_local_runtime_cancel_kills_a_task_that_ignores_sigterm(tmp_path):
    # the runner dies on SIGTERM at once, so its lock frees before such a task exits;
    # stopping must still not leave it running (on a GPU)
    (tmp_path / "stubborn.py").write_text(
        "import os, signal, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "open('pid', 'w').write(str(os.getpid()))\n"
        "time.sleep(60)\n", encoding="utf-8")
    t = Task("stubborn", launch=LaunchSpec(cwd=str(tmp_path), command=f"'{PY}' stubborn.py"))
    rt = LocalRuntime(tmp_path / "logs")
    handle = rt.submit(t, _assignment(t))
    deadline = time.monotonic() + 10
    while not (tmp_path / "pid").exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    pid = int((tmp_path / "pid").read_text())
    rt.cancel(handle)
    deadline = time.monotonic() + 5
    while pid_alive(pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not pid_alive(pid)


def test_local_runtime_reports_bad_spec(tmp_path):
    rt = LocalRuntime(tmp_path / "logs")
    t = Task("job", launch=LaunchSpec(cwd=str(tmp_path / "missing"), command="echo"))
    with pytest.raises(ValueError, match="working directory"):
        rt.submit(t, _assignment(t))


# {config}: a resolved snapshot for the user's code -------------------------------
EXAMPLES = Path(__file__).parent.parent / "examples" / "configs"


def test_config_placeholder_writes_resolved_snapshot(tmp_path):
    repo = ConfigRepository(EXAMPLES)
    t = repo.make_task(repo.get("resnet50_adamw_bs512"))
    proc = launcher.build_process(t, (0,), snapshot=tmp_path / "snap.yaml")
    i = proc.args.index("--config")
    assert proc.args[i + 1] == str(tmp_path / "snap.yaml")
    text = (tmp_path / "snap.yaml").read_text(encoding="utf-8")
    assert "_base_" not in text and "_scheduler_" not in text
    assert "batch_size: 512" in text and "arch: resnet50" in text  # own + inherited


def test_config_placeholder_needs_a_config(tmp_path):
    t = Task("job", launch=LaunchSpec(cwd=str(tmp_path), command="echo {config}"))
    with pytest.raises(ValueError, match="needs a task created from a config"):
        launcher.build_process(t, (0,))


def test_example_config_runs_end_to_end(tmp_path):
    repo = ConfigRepository(EXAMPLES)
    t = repo.make_task(repo.get("resnet50_adamw_bs512"))
    t.launch.args = [*t.launch.args, "--epochs", "1"]
    rt = LocalRuntime(tmp_path / "logs")
    a = _assignment(t, (3,))
    handle = rt.submit(t, a)
    assert _wait(rt, handle) == RunState.SUCCEEDED
    log = rt.log_path(a.id).read_text(encoding="utf-8")
    assert "CUDA_VISIBLE_DEVICES=3" in log
    assert "batch_size: 512" in log
    assert "epoch 1/1 done" in log
    assert (tmp_path / "logs" / f"{a.id}.config.yaml").exists()


def test_python_on_path_skips_windows_store_stub(tmp_path):
    stub_dir = tmp_path / "WindowsApps"
    stub_dir.mkdir()
    name = "python.exe" if sys.platform == "win32" else "python"
    stub = stub_dir / name
    stub.write_text("", encoding="utf-8")
    stub.chmod(0o755)
    assert launcher._which_python(str(stub_dir)) is None
    real = tmp_path / "bin"
    real.mkdir()
    (real / name).write_text("", encoding="utf-8")
    (real / name).chmod(0o755)
    found = launcher._which_python(os.pathsep.join([str(stub_dir), str(real)]))
    assert Path(found).parent == real
