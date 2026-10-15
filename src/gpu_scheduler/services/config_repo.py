"""ConfigRepository: scan, edit, save and derive experiment YAML configs."""

from __future__ import annotations

import re
import shlex
from datetime import timedelta
from pathlib import Path
from typing import Any

from ruamel.yaml.comments import CommentedMap

from ..models import ConfigNode, Task, parse_duration
from . import config_loader as cl

DEFAULT_COMMAND_TEMPLATE = "python train.py --config {path}"


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

    def derive(
        self, parent: ConfigNode, new_name: str, overrides: dict[str, Any] | None = None
    ) -> ConfigNode:
        """Create ``new_name.yaml`` next to ``parent`` holding ``_base_`` + overrides."""
        name = new_name.strip()
        if not re.fullmatch(r"[A-Za-z0-9_.\-]+", name):
            raise ValueError("config name may only contain letters, digits, '_', '-', '.'")
        if not name.endswith((".yaml", ".yml")):
            name += ".yaml"
        path = (parent.path.parent / name).resolve()
        if path.exists() or path in self.nodes:
            raise FileExistsError(f"{name} already exists")
        data = CommentedMap()
        data[cl.BASE_KEY] = parent.path.name
        for k, v in (overrides or {}).items():
            cl.set_dotted(data, k, v)
        cl.write_yaml(path, data)
        node = ConfigNode(path=path, own=data)
        self.nodes[path] = node
        self._recompute()
        return node

    # tasks -----------------------------------------------------------------
    def make_task(self, node: ConfigNode, template: str = DEFAULT_COMMAND_TEMPLATE) -> Task:
        if node.error:
            raise ValueError(node.error)
        meta = node.resolved.get(cl.SCHEDULER_KEY) or {}
        gpus = int(meta.get("gpus", 1))
        dur = meta.get("estimated_duration", "1h")
        duration = parse_duration(str(dur)) if not isinstance(dur, timedelta) else dur
        command = template.format(path=shlex.quote(str(node.path)), name=node.name)
        return Task(
            name=str(meta.get("name", node.name)),
            command=command,
            required_gpus=gpus,
            estimated_duration=duration,
            config_path=str(node.path),
        )


class _Missing:
    def __repr__(self) -> str:
        return "—"

    def __str__(self) -> str:
        return "—"


MISSING = _Missing()
