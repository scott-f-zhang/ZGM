from __future__ import annotations

import textwrap
from datetime import timedelta
from pathlib import Path

import pytest

from gpu_scheduler.models import Origin
from gpu_scheduler.services import config_loader as cl
from gpu_scheduler.services.config_repo import ConfigRepository


def write(d: Path, name: str, body: str) -> Path:
    p = d / name
    p.write_text(textwrap.dedent(body).lstrip(), encoding="utf-8")
    return p


@pytest.fixture
def cfg(tmp_path: Path) -> Path:
    write(tmp_path, "base.yaml", """
        # base recipe
        _scheduler_:
          gpus: 1
          estimated_duration: 2h
        model:
          arch: resnet50
        optimizer:
          type: sgd
          lr: 0.1  # tuned for bs 256
          momentum: 0.9
        data:
          batch_size: 256
          augment: [crop, flip]
        """)
    write(tmp_path, "child.yaml", """
        _base_: base.yaml
        optimizer:
          lr: 0.05
        """)
    write(tmp_path, "grandchild.yaml", """
        _base_: child.yaml
        _scheduler_:
          gpus: 2
        data:
          batch_size: 512
          extra: true
        """)
    return tmp_path


def test_single_and_multi_level_inheritance(cfg):
    repo = ConfigRepository(cfg)
    child = repo.get("child")
    assert child.error is None
    assert child.resolved["optimizer"] == {"type": "sgd", "lr": 0.05, "momentum": 0.9}
    gc = repo.get("grandchild")
    assert gc.resolved["optimizer"]["lr"] == 0.05
    assert gc.resolved["data"] == {"batch_size": 512, "augment": ["crop", "flip"], "extra": True}
    assert cl.BASE_KEY not in gc.resolved


def test_provenance(cfg):
    repo = ConfigRepository(cfg)
    gc = repo.get("grandchild")
    p = gc.provenance
    assert p["optimizer.lr"].source.name == "child.yaml"
    assert p["optimizer.lr"].origin == Origin.INHERITED
    assert p["optimizer.momentum"].source.name == "base.yaml"
    assert p["data.batch_size"].source.name == "grandchild.yaml"
    assert p["data.batch_size"].origin == Origin.OVERRIDDEN
    assert p["data.extra"].origin == Origin.NEW
    child = repo.get("child")
    assert child.provenance["optimizer.lr"].origin == Origin.OVERRIDDEN


def test_multiple_bases_merge_left_to_right(tmp_path):
    write(tmp_path, "a.yaml", "x: 1\ny: {p: 1, q: 1}\n")
    write(tmp_path, "b.yaml", "y: {q: 2}\nz: 3\n")
    write(tmp_path, "c.yaml", "_base_: [a.yaml, b.yaml]\nz: 4\n")
    c = ConfigRepository(tmp_path).get("c")
    assert c.resolved == {"x": 1, "y": {"p": 1, "q": 2}, "z": 4}
    assert c.provenance["y.q"].source.name == "b.yaml"
    assert c.provenance["z"].origin == Origin.OVERRIDDEN


def test_delete_replaces_subtree(tmp_path):
    write(tmp_path, "a.yaml", "opt: {type: sgd, lr: 0.1, momentum: 0.9}\n")
    write(tmp_path, "b.yaml", "_base_: a.yaml\nopt: {_delete_: true, type: adamw, lr: 0.001}\n")
    b = ConfigRepository(tmp_path).get("b")
    assert b.resolved["opt"] == {"type": "adamw", "lr": 0.001}
    assert "opt.momentum" not in b.provenance


def test_list_replaced_wholesale(cfg):
    write(cfg, "aug.yaml", "_base_: base.yaml\ndata: {augment: [mixup]}\n")
    assert ConfigRepository(cfg).get("aug").resolved["data"]["augment"] == ["mixup"]


def test_cycle_and_missing_base_are_errors(tmp_path):
    write(tmp_path, "x.yaml", "_base_: y.yaml\n")
    write(tmp_path, "y.yaml", "_base_: x.yaml\n")
    write(tmp_path, "z.yaml", "_base_: nope.yaml\n")
    write(tmp_path, "bad.yaml", "a: [1, 2\n")
    repo = ConfigRepository(tmp_path)
    assert "cycle" in repo.get("x").error
    assert "not found" in repo.get("z").error
    assert repo.get("bad").error


def test_editing_inherited_key_writes_child_only(cfg):
    repo = ConfigRepository(cfg)
    child = repo.get("child")
    repo.set_value(child, "optimizer.momentum", 0.95)
    assert child.dirty
    assert child.provenance["optimizer.momentum"].origin == Origin.OVERRIDDEN
    # unsaved edits already flow down to descendants
    assert repo.get("grandchild").resolved["optimizer"]["momentum"] == 0.95
    repo.save(child)
    assert "momentum: 0.95" in (cfg / "child.yaml").read_text()
    assert "momentum: 0.9\n" in (cfg / "base.yaml").read_text()
    assert repo.get("base").resolved["optimizer"]["momentum"] == 0.9


def test_revert_override(cfg):
    repo = ConfigRepository(cfg)
    child = repo.get("child")
    assert repo.remove_override(child, "optimizer.lr")
    assert child.resolved["optimizer"]["lr"] == 0.1
    assert child.provenance["optimizer.lr"].origin == Origin.INHERITED
    assert child.own == {"_base_": "base.yaml"}  # empty parent mapping pruned
    assert not repo.remove_override(child, "optimizer.lr")


def test_derive_writes_only_diff_and_tracks_parent(cfg):
    repo = ConfigRepository(cfg)
    base = repo.get("base")
    kid = repo.derive(base, "base_lr3e-4", {"optimizer.lr": 3e-4})
    text = (cfg / "base_lr3e-4.yaml").read_text()
    assert text.startswith("_base_: base.yaml")
    assert "momentum" not in text and "arch" not in text
    assert kid.resolved["optimizer"]["lr"] == 3e-4
    assert kid in repo.children(base)

    # later change in the parent propagates to the derived config
    repo.set_value(base, "model.arch", "resnet101")
    repo.save(base)
    repo.reload()
    assert repo.get("base_lr3e-4").resolved["model"]["arch"] == "resnet101"

    with pytest.raises(FileExistsError):
        repo.derive(repo.get("base"), "base_lr3e-4")
    with pytest.raises(ValueError):
        repo.derive(repo.get("base"), "../escape")


def test_save_preserves_comments(cfg):
    repo = ConfigRepository(cfg)
    base = repo.get("base")
    repo.set_value(base, "optimizer.lr", 0.2)
    repo.save(base)
    text = (cfg / "base.yaml").read_text()
    assert "# base recipe" in text
    assert "lr: 0.2  # tuned for bs 256" in text


def test_diff(cfg):
    repo = ConfigRepository(cfg)
    rows = {k: (a, b) for k, a, b in repo.diff(repo.get("grandchild"))}
    assert rows["data.batch_size"] == (256, 512)
    assert rows["_scheduler_.gpus"] == (1, 2)
    assert "optimizer.lr" not in rows
    assert repo.diff(repo.get("base")) == []


def test_make_task_from_config(cfg):
    repo = ConfigRepository(cfg)
    task = repo.make_task(repo.get("grandchild"), "python train.py -c {path} --name {name}")
    assert task.required_gpus == 2  # overridden in grandchild
    assert task.estimated_duration is None  # estimates come from run history only
    assert task.fingerprint.startswith("cfg:")
    assert task.command.endswith("grandchild.yaml --name grandchild")
    assert task.config_path == str((cfg / "grandchild.yaml").resolve())


def test_tree_structure(cfg):
    repo = ConfigRepository(cfg)
    assert [n.name for n in repo.roots()] == ["base"]
    assert [n.name for n in repo.children(repo.get("base"))] == ["child"]
    assert {n.name for n in repo.descendants(repo.get("base"))} == {"child", "grandchild"}


@pytest.mark.parametrize("text,value", [
    ("3e-4", 3e-4), ("1.0e-3", 1e-3), ("42", 42), ("true", True), ("null", None),
    ("[1, 2]", [1, 2]), ("adamw", "adamw"), ("'007'", "007"),
])
def test_parse_scalar(text, value):
    assert cl.parse_scalar(text) == value


def test_examples_resolve():
    repo = ConfigRepository(Path(__file__).parent.parent / "examples" / "configs")
    assert repo.nodes and all(n.error is None for n in repo.nodes.values())
    bs512 = repo.get("resnet50_adamw_bs512")
    assert "momentum" not in bs512.resolved["optimizer"]  # removed by _delete_ in parent


def test_fingerprint_follows_parameters_not_names(cfg):
    repo = ConfigRepository(cfg)
    child = repo.get("child")
    fp = repo.params_fingerprint(child, 1)
    twin = repo.derive(repo.get("base"), "twin", {"optimizer.lr": 0.05})  # same params as child
    assert repo.params_fingerprint(twin, 1) == fp
    assert repo.params_fingerprint(repo.get("base"), 1) != fp
    assert repo.params_fingerprint(child, 2) != fp
    # _scheduler_ metadata is not a training parameter
    repo.set_value(child, "_scheduler_.name", "nice-name")
    assert repo.params_fingerprint(child, 1) == fp


def test_rename_updates_children(cfg):
    repo = ConfigRepository(cfg)
    new, children = repo.rename(repo.get("base"), "resnet_base")
    assert new.name == "resnet_base.yaml" and not (cfg / "base.yaml").exists()
    assert children == ["child"]
    assert "_base_: resnet_base.yaml" in (cfg / "child.yaml").read_text()
    gc = repo.get("grandchild")
    assert gc.error is None and gc.resolved["model"]["arch"] == "resnet50"
    with pytest.raises(FileExistsError):
        repo.rename(repo.get("child"), "grandchild")
    repo.set_value(repo.get("child"), "seed", 1)
    with pytest.raises(ValueError, match="save first"):
        repo.rename(repo.get("grandchild"), "gc2")
