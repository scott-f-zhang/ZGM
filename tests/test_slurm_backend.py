"""The Slurm backend: squeue -> groups, and the jobs zgm can't use. The daemon
test runs the whole loop with fake squeue and ssh commands on PATH."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from zgm.cluster.slurm import SlurmBackend, parse_gpus

PY = sys.executable
ROOT = Path(__file__).parent.parent
pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="Slurm clusters run Linux")

LISTING = (
    "1001|RUNNING|train|gres/gpu:h100:1|node01|2030-01-01T09:00:00|2030-01-02T09:00:00|"
    "12:01:02|None|1\n"
    "1002|PENDING|big|gres/gpu:h100:4||2030-01-01T11:21:52|N/A|7-00:00:00|Priority|1\n"
    "1003|RUNNING|wide|gres/gpu:2|node02,node03|N/A|N/A|1:00:00|None|2\n"
    "1004|RUNNING|a|gres/gpu:1|node04|N/A|N/A|1:00:00|None|1\n"
    "1005|RUNNING|b|gres/gpu:1|node04|N/A|N/A|1:00:00|None|1\n"
    "1006|RUNNING|cpu|N/A|node05|N/A|N/A|1:00:00|None|1\n")


def backend(listing=LISTING, job="1001"):
    run = lambda *a, **k: subprocess.CompletedProcess(a, 0, listing, "")  # noqa: E731
    return SlurmBackend(run=run, user="someone", job=job)


def test_jobs_become_groups():
    groups = {g.key: g for g in backend().groups()}
    mine, pending = groups["1001"], groups["1002"]
    assert (mine.host, mine.usable, mine.info["time_left"]) == ("node01", True, "12:01:02")
    assert (pending.host, pending.usable, pending.info["start"]) == (
        "", False, "2030-01-01T11:21:52")
    assert groups["1003"].note == "runs on several nodes: not used"
    assert groups["1004"].note == groups["1005"].note == (
        "shares its node with another of your jobs: not used")
    assert not groups["1006"].usable and groups["1006"].note == ""  # no GPUs: just listed
    assert parse_gpus("gres/gpu:4") == 4 and parse_gpus("N/A") == 0
    assert parse_gpus("gres/gpu:h100:2,gres/gpu:l4:1") == 3


def test_columns_say_what_each_job_is():
    b = backend()
    rows = {g.key: {"key": g.key, "state": "up" if g.usable else "down", "host": g.host,
                    "note": g.note, "info": g.info} for g in b.groups()}
    assert b.columns(rows["1001"], local=True) == [
        "1001", "RUNNING", "node01", "1 x h100", "this job · 12:01:02 left"]
    assert b.columns(rows["1002"], local=False)[-1] == "starts ~01-01 11:21 (Priority)"
    order = sorted(rows.values(), key=b.order)
    assert [r["key"] for r in order][:2] == ["1001", "1003"] and order[-1]["key"] == "1002"


def test_squeue_failure_is_reported():
    b = SlurmBackend(run=lambda *a, **k: subprocess.CompletedProcess(a, 1, "", "no controller"),
                     user="someone", job=None)
    with pytest.raises(RuntimeError, match="squeue: no controller"):
        b.groups()


# the daemon, end to end -------------------------------------------------------------------
FAKE_SSH = """#!/bin/sh
while [ "$1" = "-o" ]; do shift 2; done
shift
exec sh -c "$1"
"""
JOB = """import os, sys
print("cvd", os.environ["CUDA_VISIBLE_DEVICES"], "zgm", os.environ["ZGM_GPU_IDS"],
      "slurm", sorted(k for k in os.environ if k.startswith("SLURM_")), flush=True)
"""


def zgm_cli(*args, env, timeout=60):
    return subprocess.run([PY, "-m", "zgm", *args], capture_output=True, text=True, cwd=ROOT,
                          env=env, timeout=timeout)


def test_daemon_runs_tasks_on_another_jobs_gpus(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "ssh").write_text(FAKE_SSH)
    (bin_dir / "squeue").write_text('#!/bin/sh\ncat "$ZGM_TEST_SQUEUE"\n')
    for f in bin_dir.iterdir():
        f.chmod(0o755)
    listing = tmp_path / "squeue.txt"
    listing.write_text("1|RUNNING|here|gres/gpu:1|nodeA|N/A|N/A|10:00:00|None|1\n"
                       "2|RUNNING|far|gres/gpu:2|nodeB|N/A|N/A|5-00:00:00|None|1\n"
                       "3|PENDING|later|gres/gpu:4||2030-01-01T11:21:52|N/A|7-00:00:00|"
                       "Priority|1\n")
    db = tmp_path / "s.db"
    # local GPUs: 1 mock; the "remote" probe (run here by the fake ssh) sees 2
    env = {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
           "SLURM_JOB_ID": "1", "ZGM_MOCK_GPUS": "2", "ZGM_TEST_SQUEUE": str(listing)}
    proc = subprocess.Popen([PY, "-m", "zgm", "daemon", "run", "--db", str(db),
                             "--cluster", "slurm", "--mock-gpus", "1", "--interval", "0.2"],
                            cwd=ROOT, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    try:
        deadline = time.time() + 30
        while time.time() < deadline:
            status = zgm_cli("daemon", "status", "--db", str(db), env=env).stdout
            if "GPU 1-2" in status:
                break
            time.sleep(0.3)
        assert "nodeB: 2 x Mock A100-SXM4-80GB (GPU 1-2)" in status
        assert "cluster:    slurm" in status and "PENDING" in status and "this job" in status

        (tmp_path / "job.py").write_text(JOB)
        r = zgm_cli("submit", "--db", str(db), "--json", "--python", PY, "-C", str(tmp_path),
                    "-g", "2", "-n", "far", "--", "job.py", env=env)
        assert r.returncode == 0, r.stdout + r.stderr
        assert json.loads(r.stdout)["gpus"] == [1, 2]
        r = zgm_cli("logs", "--db", str(db), "-f", "far", env=env, timeout=90)
        assert r.returncode == 0, r.stdout + r.stderr
        assert "cvd 0,1 zgm 1,2 slurm []" in r.stdout
        r = zgm_cli("submit", "--db", str(db), "--python", PY, "-C", str(tmp_path), "-g", "4",
                    "--", "job.py", env=env)
        assert r.returncode == 2 and "--hold" in r.stderr
    finally:
        zgm_cli("daemon", "stop", "--db", str(db), env=env)
        proc.kill()
        out = proc.communicate()[0]
    assert "nodeB (Slurm job 2): 2 x Mock A100-SXM4-80GB added as GPU 1-2" in out
