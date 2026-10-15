"""ConfigNode: one experiment YAML file and its resolved (inherited) view."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class Origin(str, Enum):
    INHERITED = "inherited"  # value comes from an ancestor
    OVERRIDDEN = "override"  # this file redefines a value an ancestor set
    NEW = "new"  # this file introduces the key


@dataclass
class ProvenanceEntry:
    source: Path  # file whose value wins
    origin: Origin


@dataclass
class ConfigNode:
    path: Path
    bases: list[Path] = field(default_factory=list)
    own: Any = None  # ruamel CommentedMap of this file (minus _base_)
    resolved: dict = field(default_factory=dict)
    provenance: dict[str, ProvenanceEntry] = field(default_factory=dict)
    error: str | None = None
    dirty: bool = False

    @property
    def name(self) -> str:
        return self.path.stem
