"""zgm check: the state dir, and for each other host: reached without a
prompt, sees the state dir, runs zgm. Hosts are faked with an ``ssh`` on PATH
that runs the script here (optionally rewriting a path, to fake a host that
doesn't share it)."""

from __future__ import annotations

import os
import sys

import pytest

from zgm.cluster import check
from zgm.cluster.base import Group

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="remote hosts run POSIX shells")

# skip the options and the host; run the script here, with $ZGM_TEST_FROM -> $ZGM_TEST_TO
FAKE_SSH = """#!/bin/sh
while [ "$1" = "-o" ]; do shift 2; done
shift
script="$1"
if [ -n "$ZGM_TEST_FROM" ]; then
  script=$(printf '%s' "$script" | sed "s#$ZGM_TEST_FROM#$ZGM_TEST_TO#g")
fi
exec sh -c "$script"
"""
DENY_SSH = "#!/bin/sh\necho 'Permission denied (publickey)' >&2\nexit 255\n"


@pytest.fixture
def setup(tmp_path, monkeypatch):
    """A private state dir of user "someone", and a fake ssh to write."""
    monkeypatch.setattr(check.getpass, "getuser", lambda: "someone")
    monkeypatch.setenv("ZGM_MOCK_GPUS", "2")
    for var in ("ZGM_TEST_FROM", "ZGM_TEST_TO"):
        monkeypatch.delenv(var, raising=False)
    root = tmp_path / "shared" / "someone" / "zgm"
    root.mkdir(parents=True, mode=0o700)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")

    def ssh(script: str) -> None:
        (bin_dir / "ssh").write_text(script)
        (bin_dir / "ssh").chmod(0o755)

    ssh(FAKE_SSH)
    return root / "state.db", ssh


def run(db, capsys, hosts=("nodeX",), cluster=None):
    code = check.run_check(db, cluster, list(hosts), "ssh")
    return code, capsys.readouterr().out


def test_a_host_that_works(setup, capsys):
    db, _ = setup
    code, out = run(db, capsys)
    assert code == 0, out
    assert "[  ok] state dir" in out and "(yours, private)" in out
    assert "[  ok] reach      nodeX" in out
    assert f"[  ok] shared     nodeX sees {db.parent}" in out
    assert "[  ok] zgm        runs there; GPUs: 2 x Mock A100-SXM4-80GB" in out
    assert out.rstrip().endswith("result: ready")
    assert not list(db.parent.glob(".zgm-check-*"))  # cleans up after itself


def test_a_host_that_refuses_ssh(setup, capsys):
    db, ssh = setup
    ssh(DENY_SSH)
    code, out = run(db, capsys)
    assert code == 1
    assert "[FAIL] reach      nodeX: Permission denied (publickey)" in out
    assert "ssh -o BatchMode=yes nodeX true" in out and "authorized_keys" in out
    assert "] shared" not in out  # nothing more to learn about that host


def test_a_host_that_does_not_share_the_state_dir(setup, capsys, monkeypatch, tmp_path):
    db, _ = setup
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.setenv("ZGM_TEST_FROM", str(db.parent))
    monkeypatch.setenv("ZGM_TEST_TO", str(elsewhere))
    code, out = run(db, capsys)
    assert code == 1
    assert f"[FAIL] shared     nodeX can't read {db.parent}" in out
    assert "every host mounts at the same path" in out


def test_a_host_without_zgm(setup, capsys, monkeypatch):
    db, _ = setup
    monkeypatch.setenv("ZGM_TEST_FROM", sys.executable)
    monkeypatch.setenv("ZGM_TEST_TO", "/nonexistent/python")
    code, out = run(db, capsys)
    assert code == 1
    assert "[FAIL] zgm" in out and "install it in a directory" in out


def test_state_dir_others_can_write(setup, capsys):
    db, _ = setup
    db.parent.chmod(0o777)
    code, out = run(db, capsys, hosts=())
    assert code == 1
    assert "[FAIL] state dir" in out and "other users can write to it" in out
    assert f"chmod 700 {db.parent}" in out


def test_shared_state_dir_should_name_its_user(setup, capsys, tmp_path, monkeypatch):
    _, _ = setup
    monkeypatch.setattr(check.Path, "home", classmethod(lambda cls: tmp_path / "home"))
    db = tmp_path / "team" / "zgm" / "state.db"  # outside home, no user name in it
    code, out = run(db, capsys, hosts=())
    assert code == 0  # a warning, not a failure
    assert "[warn] state dir" in out and "doesn't name you (someone)" in out
    assert f"ZGM_HOME={tmp_path / 'team' / 'someone' / 'zgm'}" in out


class TwoHosts:
    name = "fake"
    interval = 0.0
    host_env = ()

    def groups(self):
        return [Group("1", "nodeA", True), Group("2", "nodeB", True),
                Group("3", "", False, info={})]

    def local_key(self):
        return "1"

    def title(self, key):
        return f"group {key}"


def test_checks_the_hosts_a_backend_would_use(setup, capsys, monkeypatch):
    db, _ = setup
    monkeypatch.setattr(check, "load_backend", lambda name: TwoHosts())
    code, out = run(db, capsys, hosts=(), cluster="fake")
    assert code == 0, out
    assert "[  ok] fake       3 group(s); 1 usable on other hosts" in out
    assert "nodeB (group 2)" in out and "nodeA" not in out  # its own group needs no ssh
