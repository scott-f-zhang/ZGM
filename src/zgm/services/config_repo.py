"""ConfigRepository: scan, edit, save and derive experiment YAML configs."""

from __future__ import annotations

import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any

from ruamel.yaml.comments import CommentedMap

from ..models import ConfigNode, LaunchSpec, Task, config_fingerprint
from . import config_loader as cl

DEFAULT_COMMAND_TEMPLATE = "python train.py --config {path}"


def quote_path(path: str) -> str:
    """Quote ``path`` for the local shell (cmd.exe rules on Windows, POSIX elsewhere)."""
    if os.name == "nt":
        return subprocess.list2cmdline([path])
    return shlex.quote(path)


class ConfigRepository:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.nodes: dict[Path, ConfigNode] = {}
        self.reload()

    # loading ---------------------------------------------------------------
    def files(self) -> list[Path]:
        if not self.root.is_dir():
            return []
        return sorted(
            p.resolve() for p in self.root.rglob("*") if p.suffix in (".yaml", ".yml") and p.is_file()
        )

    def reload(self) -> None:
        """Re-read everything from disk, discarding unsaved edits."""
        cache: dict = {}
        self.nodes = {p: cl.load_node(p, cache) for p in self.files()}

    def _recompute(self) -> None:
        """Re-resolve every node from in-memory ``own`` data (keeps unsaved edits)."""
        own = {p: n.own for p, n in self.nodes.items()}
        for path, node in self.nodes.items():
            if node.own is None:
                continue
            node.bases = cl.base_paths(path, node.own)
            try:
                node.resolved, node.provenance = self._resolve_mem(path, own, ())
                node.error = None
            except cl.ConfigError as e:
                node.error = str(e)
                node.resolved, node.provenance = {}, {}

    def _resolve_mem(self, path: Path, own: dict, stack: tuple):
        from ..models import Origin, ProvenanceEntry

        if path in stack:
            raise cl.ConfigError(
                "inheritance cycle: " + " -> ".join(p.name for p in (*stack, path))
            )
        data = own.get(path)
        if data is None:
            if not path.exists():
                raise cl.ConfigError(f"base file not found: {path.name}")
            data = cl.read_yaml(path)
        resolved: dict = {}
        prov: dict = {}
        for bp in cl.base_paths(path, data):
            b_res, b_prov = self._resolve_mem(bp, own, (*stack, path))
            resolved = cl.merge_plain(resolved, b_res)
            prov.update(b_prov)
        prov = {k: ProvenanceEntry(e.source, Origin.INHERITED) for k, e in prov.items()}
        resolved = cl.merge(resolved, data, prov, path)
        return resolved, prov

    # tree ------------------------------------------------------------------
    def roots(self) -> list[ConfigNode]:
        return [n for n in self.nodes.values() if not n.bases or n.bases[0] not in self.nodes]

    def children(self, node: ConfigNode) -> list[ConfigNode]:
        """Nodes whose *first* base is ``node`` (where they appear in the tree)."""
        return [n for n in self.nodes.values() if n.bases and n.bases[0] == node.path]

    def descendants(self, node: ConfigNode) -> list[ConfigNode]:
        out = []
        for n in self.nodes.values():
            if n is not node and node.path in self._ancestors(n):
                out.append(n)
        return out

    def _ancestors(self, node: ConfigNode, seen: set | None = None) -> set[Path]:
        seen = seen if seen is not None else set()
        for b in node.bases:
            if b not in seen:
                seen.add(b)
                if b in self.nodes:
                    self._ancestors(self.nodes[b], seen)
        return seen

    def get(self, name_or_path: str | Path) -> ConfigNode:
        p = Path(name_or_path)
        if not p.is_absolute():
            cand = (self.root / p).resolve()
            if cand in self.nodes:
                return self.nodes[cand]
            for n in self.nodes.values():
                if n.name == str(name_or_path):
                    return n
        p = p.resolve()
        if p in self.nodes:
            return self.nodes[p]
        raise KeyError(name_or_path)

    def rel(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.root))
        except ValueError:
            return str(path)

    # parameters ------------------------------------------------------------
    def params(self, node: ConfigNode) -> list[tuple[str, Any, Any]]:
        """(dotted key, value, provenance) rows; ``_scheduler_.*`` keys are listed last."""
        flat = cl.flatten(node.resolved)
        flat = dict(sorted(flat.items(), key=lambda kv: kv[0].startswith(cl.SCHEDULER_KEY + ".")))
        return [(k, v, node.provenance.get(k)) for k, v in flat.items()]

    def parent_resolved(self, node: ConfigNode) -> dict:
        resolved: dict = {}
        for b in node.bases:
            if b in self.nodes and not self.nodes[b].error:
                resolved = cl.merge_plain(resolved, self.nodes[b].resolved)
        return resolved

    def diff(self, node: ConfigNode) -> list[tuple[str, Any, Any]]:
        """(key, parent value or MISSING, this value) for keys that differ from parents."""
        if not node.bases:
            return []
        mine = cl.flatten(node.resolved)
        theirs = cl.flatten(self.parent_resolved(node))
        rows = []
        for k in sorted(set(mine) | set(theirs)):
            a, b = theirs.get(k, MISSING), mine.get(k, MISSING)
            if a != b:
                rows.append((k, a, b))
        return rows

    # editing ---------------------------------------------------------------
    def set_value(self, node: ConfigNode, dotted: str, value: Any) -> None:
        """Write into *this* file only; an inherited key becomes an override here."""
        if node.own is None:
            node.own = CommentedMap()
        cl.set_dotted(node.own, dotted, value)
        node.dirty = True
        self._recompute()

    def remove_override(self, node: ConfigNode, dotted: str) -> bool:
        if node.own is None or not cl.delete_dotted(node.own, dotted):
            return False
        node.dirty = True
        self._recompute()
        return True

    def is_own_key(self, node: ConfigNode, dotted: str) -> bool:
        return node.own is not None and cl.get_dotted(node.own, dotted, MISSING) is not MISSING

    def save(self, node: ConfigNode) -> None:
        cl.write_yaml(node.path, node.own if node.own is not None else CommentedMap())
        node.dirty = False

    def save_all(self) -> int:
        dirty = [n for n in self.nodes.values() if n.dirty]
        for n in dirty:
            self.save(n)
        return len(dirty)

    @staticmethod
    def _target(directory: Path, new_name: str, keep_suffix: str = ".yaml") -> Path:
        name = new_name.strip()
        if not re.fullmatch(r"[A-Za-z0-9_.\-]+", name) or name.strip(".") == "":
            raise ValueError("config name may only contain letters, digits, '_', '-', '.'")
        if not name.endswith((".yaml", ".yml")):
            name += keep_suffix
        path = (directory / name).resolve()
        if path.exists():
            raise FileExistsError(f"{name} already exists")
        return path

    def derive(
        self, parent: ConfigNode, new_name: str, overrides: dict[str, Any] | None = None
    ) -> ConfigNode:
        """Create ``new_name.yaml`` next to ``parent`` holding ``_base_`` + overrides."""
        path = self._target(parent.path.parent, new_name)
        data = CommentedMap()
        data[cl.BASE_KEY] = parent.path.name
        for k, v in (overrides or {}).items():
            cl.set_dotted(data, k, v)
        cl.write_yaml(path, data)
        node = ConfigNode(path=path, own=data)
        self.nodes[path] = node
        self._recompute()
        return node

    def rename(self, node: ConfigNode, new_name: str) -> tuple[Path, list[str]]:
        """Rename ``node``'s file and repoint every child's ``_base_`` at the new name.

        Returns (new path, names of children that were rewritten). Refuses while
        any config has unsaved edits, so nothing unsaved is written or lost.
        """
        old = node.path
        new = self._target(old.parent, new_name, keep_suffix=old.suffix)
        children = [n for n in self.nodes.values() if old in n.bases]
        dirty = [n.name for n in self.nodes.values() if n.dirty]  # reload() would drop them
        if dirty:
            raise ValueError(f"save first: unsaved edits in {', '.join(dirty)}")
        for child in children:
            raw = child.own.get(cl.BASE_KEY)
            rel = os.path.relpath(new, child.path.parent)
            if isinstance(raw, str):
                child.own[cl.BASE_KEY] = rel
            else:
                for i, item in enumerate(raw):
                    if (child.path.parent / str(item)).resolve() == old:
                        raw[i] = rel
        old.rename(new)
        for child in children:
            self.save(child)
        self.reload()
        return new, [c.name for c in children]

    # tasks -----------------------------------------------------------------
    def params_fingerprint(self, node: ConfigNode, gpus: int) -> str:
        """Same resolved parameters + GPU count => same workload (for estimates)."""
        params = {k: v for k, v in node.resolved.items() if k != cl.SCHEDULER_KEY}
        return config_fingerprint(params, gpus)

    def launch_spec(self, node: ConfigNode,
                    template: str = DEFAULT_COMMAND_TEMPLATE) -> LaunchSpec:
        """How a task for ``node`` starts: its (inherited) ``_scheduler_.run``
        block, or else ``template`` as a shell command. A relative ``run.cwd`` is
        relative to the config file's directory; the default is zgm's own."""
        meta = node.resolved.get(cl.SCHEDULER_KEY) or {}
        run = meta.get("run")
        if run is None:
            return LaunchSpec(command=template.format(path=quote_path(str(node.path)),
                                                      name=node.name))
        try:
            spec = LaunchSpec.from_dict(dict(run) if isinstance(run, dict) else run)
        except ValueError as e:
            raise ValueError(f"{cl.SCHEDULER_KEY}.run: {e}") from None
        cwd = Path(spec.cwd).expanduser() if spec.cwd else Path.cwd()
        spec.cwd = str((node.path.parent / cwd).resolve())
        return spec

    def make_task(self, node: ConfigNode, template: str = DEFAULT_COMMAND_TEMPLATE) -> Task:
        """A task for ``node``. Its estimate comes from past runs of the same
        parameters, never from the config itself."""
        if node.error:
            raise ValueError(node.error)
        meta = node.resolved.get(cl.SCHEDULER_KEY) or {}
        gpus = int(meta.get("gpus", 1))
        return Task(
            name=str(meta.get("name", node.name)),
            launch=self.launch_spec(node, template),
            required_gpus=gpus,
            config_path=str(node.path),
            fingerprint=self.params_fingerprint(node, gpus),
            priority=int(meta.get("priority", 0)),
        )


class _Missing:
    def __repr__(self) -> str:
        return "—"

    def __str__(self) -> str:
        return "—"


MISSING = _Missing()
