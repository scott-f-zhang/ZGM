"""ConfigChain: a task's config as the Configs view's tree shows it, cut down to
the path that leads to it: each parent (``_base_``, the first one, where the
tree files a config) above its child, without siblings."""

from __future__ import annotations

from pathlib import Path

from rich.text import Text
from textual.widgets import Tree

from ..models import ConfigNode
from ..services import config_loader as cl
from . import theme

MAX_DEPTH = 20
MIN_WIDTH = 24


def config_label(node: ConfigNode) -> Text:
    """A config's label in a tree: name, ⧉N when it has N parents, * if unsaved."""
    t = Text()
    if node.error:
        t.append("⚠ ", style=f"bold {theme.BAD}")
    t.append(node.name, style=theme.BAD if node.error else theme.MAIN_FG)
    if len(node.bases) > 1:
        t.append(f" ⧉{len(node.bases)}", style=theme.CACHED[1])
    if node.dirty:
        t.append(" *", style=f"bold {theme.WARN}")
    return t


def chain(path: Path) -> list[ConfigNode]:
    """``path`` and its first parents, the topmost first."""
    out: list[ConfigNode] = []
    cache: dict = {}
    seen = set()
    while path not in seen and len(out) < MAX_DEPTH:
        seen.add(path)
        node = cl.load_node(path, cache)
        out.append(node)
        if not node.bases or not node.bases[0].exists():
            break
        path = node.bases[0]
    return out[::-1]


class ConfigChain(Tree, can_focus=False):
    def __init__(self, **kw) -> None:
        super().__init__("config", **kw)
        self.show_root = False
        self.path: str | None = None

    def show(self, config_path: str | None) -> None:
        if config_path == self.path:
            return
        self.path = config_path
        self.clear()
        self.styles.width = MIN_WIDTH
        if not config_path:
            self.root.add_leaf(Text("no config", style=theme.SUBTLE))
            return
        path = Path(config_path)
        if not path.exists():
            self.root.add_leaf(Text(f"⚠ {path.name} is gone", style=theme.BAD))
            return
        parent = self.root
        nodes = chain(path.resolve())
        widest = 0
        for i, node in enumerate(nodes):
            label = config_label(node)
            widest = max(widest, (i + 1) * self.guide_depth + label.cell_len)
            if i == len(nodes) - 1:
                label.stylize("bold")
                parent.add_leaf(label, data=node.path)
            else:
                parent = parent.add(label, data=node.path, expand=True)
        self.styles.width = max(MIN_WIDTH, widest + 4)  # + the divider, padding and scrollbar
