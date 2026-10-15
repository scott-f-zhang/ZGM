"""YAML config inheritance via a ``_base_`` key (mmengine-style).

A child file names one or more parents and stores only what it changes::

    _base_: resnet50_base.yaml     # or a list: [a.yaml, b.yaml]
    optimizer:
      lr: 3.0e-4

Rules:
- ``_base_`` paths resolve relative to the child file; chains and multiple
  bases (merged left to right) are allowed; cycles are an error.
- Mappings merge recursively; anything else (scalars, lists) is replaced.
- ``_delete_: true`` inside a mapping replaces the inherited subtree instead of
  merging into it.
"""

from __future__ import annotations

import copy
import io
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap

from ..models import ConfigNode, Origin, ProvenanceEntry

BASE_KEY = "_base_"
DELETE_KEY = "_delete_"
SCHEDULER_KEY = "_scheduler_"


class ConfigError(Exception):
    pass


def make_yaml() -> YAML:
    y = YAML()  # round-trip: keeps comments and key order
    y.preserve_quotes = True
    y.indent(mapping=2, sequence=4, offset=2)
    y.width = 4096
    return y


def read_yaml(path: Path) -> CommentedMap:
    with open(path, encoding="utf-8") as f:
        data = make_yaml().load(f)
    if data is None:
        data = CommentedMap()
    if not isinstance(data, dict):
        raise ConfigError(f"{path.name}: top level must be a mapping")
    return data


def dump_yaml(data: Any) -> str:
    buf = io.StringIO()
    make_yaml().dump(data, buf)
    return buf.getvalue()


def write_yaml(path: Path, data: Any) -> None:
    path.write_text(dump_yaml(data), encoding="utf-8")


def base_paths(path: Path, data: dict) -> list[Path]:
    raw = data.get(BASE_KEY)
    if raw is None:
        return []
    items = [raw] if isinstance(raw, str) else list(raw)
    return [(path.parent / str(p)).resolve() for p in items]


def to_plain(obj: Any) -> Any:
    """ruamel containers -> plain dict/list (for resolved views)."""
    if isinstance(obj, dict):
        return {str(k): to_plain(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_plain(v) for v in obj]
    return obj


def _strip_meta(d: dict) -> dict:
    return {k: v for k, v in d.items() if k not in (BASE_KEY, DELETE_KEY)}


def merge(
    base: dict,
    override: dict,
    prov: dict[str, ProvenanceEntry],
    source: Path,
    prefix: str = "",
) -> dict:
    """Merge ``override`` (from ``source``) into a copy of ``base``; update provenance."""
    out = dict(base)
    for key, val in _strip_meta(override).items():
        dotted = f"{prefix}{key}"
        if isinstance(val, dict) and not val.get(DELETE_KEY) and isinstance(out.get(key), dict):
            out[key] = merge(out[key], val, prov, source, dotted + ".")
            continue
        existed = key in out
        if isinstance(val, dict) and val.get(DELETE_KEY):
            for k in [k for k in prov if k == dotted or k.startswith(dotted + ".")]:
                del prov[k]
            val = _strip_meta(val)
        elif existed:
            # drop provenance of an inherited subtree being replaced by a scalar/list
            for k in [k for k in prov if k.startswith(dotted + ".")]:
                del prov[k]
        out[key] = copy.deepcopy(to_plain(val))
        _record(out[key], dotted, prov, source, Origin.OVERRIDDEN if existed else Origin.NEW)
    return out


def _record(val: Any, dotted: str, prov: dict, source: Path, origin: Origin) -> None:
    if isinstance(val, dict) and val:
        for k, v in val.items():
            sub = f"{dotted}.{k}"
            o = origin
            if origin == Origin.OVERRIDDEN and sub not in prov:
                o = Origin.NEW
            _record(v, sub, prov, source, o)
    else:
        prev = prov.get(dotted)
        if origin == Origin.OVERRIDDEN and prev is None:
            origin = Origin.NEW
        prov[dotted] = ProvenanceEntry(source, origin)


def resolve(path: Path, _stack: tuple[Path, ...] = (), _cache: dict | None = None):
    """Return (resolved_dict, provenance) for ``path``, following ``_base_``."""
    path = path.resolve()
    cache = _cache if _cache is not None else {}
    if path in _stack:
        chain = " -> ".join(p.name for p in (*_stack, path))
        raise ConfigError(f"inheritance cycle: {chain}")
    if path in cache:
        res, prov = cache[path]
        return copy.deepcopy(res), dict(prov)
    if not path.exists():
        raise ConfigError(f"base file not found: {path.name}")
    data = read_yaml(path)
    resolved: dict = {}
    prov: dict[str, ProvenanceEntry] = {}
    for bp in base_paths(path, data):
        b_res, b_prov = resolve(bp, (*_stack, path), cache)
        # bases merge left to right; the later base's values count as defined by it
        for k, entry in b_prov.items():
            b_prov[k] = ProvenanceEntry(entry.source, Origin.INHERITED)
        resolved = merge_plain(resolved, b_res)
        prov.update(b_prov)
    for k, entry in prov.items():
        prov[k] = ProvenanceEntry(entry.source, Origin.INHERITED)
    resolved = merge(resolved, data, prov, path)
    cache[path] = (copy.deepcopy(resolved), dict(prov))
    return resolved, prov


def merge_plain(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge_plain(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_node(path: Path, cache: dict | None = None) -> ConfigNode:
    path = path.resolve()
    node = ConfigNode(path=path)
    try:
        data = read_yaml(path)
        node.own = data
        node.bases = base_paths(path, data)
        node.resolved, node.provenance = resolve(path, _cache=cache)
    except ConfigError as e:
        node.error = str(e)
    except Exception as e:  # YAML syntax errors etc.
        node.error = f"{type(e).__name__}: {e}"
    return node


def flatten(d: dict, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict) and v:
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


def get_dotted(d: dict, dotted: str, default: Any = None) -> Any:
    cur: Any = d
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def set_dotted(d: dict, dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    cur = d
    for part in parts[:-1]:
        nxt = cur.get(part)
        if not isinstance(nxt, dict):
            nxt = CommentedMap()
            cur[part] = nxt
        cur = nxt
    cur[parts[-1]] = value


def delete_dotted(d: dict, dotted: str) -> bool:
    parts = dotted.split(".")
    chain = [d]
    cur = d
    for part in parts[:-1]:
        nxt = cur.get(part)
        if not isinstance(nxt, dict):
            return False
        chain.append(nxt)
        cur = nxt
    if parts[-1] not in cur:
        return False
    del cur[parts[-1]]
    # prune now-empty parent mappings
    for parent, part in zip(reversed(chain[:-1]), reversed(parts[:-1])):
        child = parent[part]
        if isinstance(child, dict) and not child:
            del parent[part]
        else:
            break
    return True


def parse_scalar(text: str) -> Any:
    """Interpret what the user typed as a YAML value (``3e-4`` -> float, ...)."""
    text = text.strip()
    if text == "":
        return ""
    y = YAML(typ="safe")
    try:
        val = y.load(text)
    except Exception as e:
        raise ConfigError(f"not valid YAML: {e}") from e
    # YAML 1.2 parses 3e-4 as a string; be friendlier about unquoted floats
    if isinstance(val, str) and text[0] not in "'\"":
        try:
            return float(val) if any(c in val.lower() for c in ".e") else int(val)
        except ValueError:
            return val
    return val


def format_value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return "null"
    if isinstance(v, (list, dict)):
        import json

        return json.dumps(v, default=str)
    return str(v)
