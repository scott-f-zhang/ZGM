"""The Macs cluster backend: one Mac is one group with one Apple GPU."""

from __future__ import annotations

import pytest

from zgm.cluster.macs import MacsBackend, configured_hosts


def test_each_mac_is_a_group():
    b = MacsBackend(hosts=["mac-a", "mac-b"], local="mac-a")
    groups = b.groups()
    assert [(g.key, g.host, g.usable, g.transport) for g in groups] == [
        ("mac-a", "mac-a", True, "ssh"), ("mac-b", "mac-b", True, "ssh")]
    assert b.local_key() == "mac-a"
    assert b.title("mac-b") == "Mac mac-b"


def test_local_key_matches_short_and_long_names():
    assert MacsBackend(hosts=["mac-a.local"], local="mac-a").local_key() == "mac-a.local"
    assert MacsBackend(hosts=["mac-a"], local="mac-a.local").local_key() == "mac-a"
    assert MacsBackend(hosts=["mac-a"], local="elsewhere").local_key() is None


def test_no_macs_is_explained():
    with pytest.raises(RuntimeError, match="ZGM_MACS"):
        MacsBackend(hosts=[], local="mac-a").groups()


def test_columns_and_order_put_reachable_macs_first():
    b = MacsBackend(hosts=["mac-b", "mac-a"], local="mac-a")
    gone = {"key": "mac-b", "host": "mac-b", "state": "gone", "note": "probe failed"}
    up = {"key": "mac-a", "host": "mac-a", "state": "up", "note": ""}
    assert b.columns(up, True) == ["mac-a", "up", "this Mac"]
    assert b.columns(gone, False) == ["mac-b", "gone", "probe failed"]
    assert sorted([gone, up], key=b.order)[0] is up


def test_hosts_from_environment(monkeypatch):
    monkeypatch.setenv("ZGM_MACS", "mac-a, mac-b  mac-a")
    assert configured_hosts() == ["mac-a", "mac-b"]


def test_hosts_from_state_dir_file(monkeypatch, tmp_path):
    monkeypatch.delenv("ZGM_MACS", raising=False)
    monkeypatch.setenv("ZGM_HOME", str(tmp_path))
    (tmp_path / "macs").write_text("# workers\nmac-a\n\nmac-b  # second\n", encoding="utf-8")
    assert configured_hosts() == ["mac-a", "mac-b"]
