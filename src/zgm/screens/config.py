"""Configs view: browse YAML experiment configs, edit values, derive children."""

from __future__ import annotations

import statistics
from datetime import datetime, timedelta
from pathlib import Path

from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Input, Static, Tree

from ..models import AssignmentState, ConfigNode, Origin, format_duration
from ..services import config_loader as cl
from ..services.config_repo import MISSING
from ..widgets.config_chain import config_label
from ..widgets.dialogs import FormScreen
from ..widgets import theme
from ..widgets.theme import box_title
from ..widgets.view_tabs import ViewTabs

ORIGIN_STYLE = {
    Origin.INHERITED: ("○", theme.SUBTLE),
    Origin.OVERRIDDEN: ("●", theme.WARN),
    Origin.NEW: ("✚", theme.OK),
}

TREE_MIN_WIDTH = 36  # the tree grows to fit its deepest label, up to max-width (app.tcss)


class ConfigScreen(Screen):
    BINDINGS = [
        Binding("e", "edit", "edit"),
        Binding("x", "revert", "revert to inherited"),
        Binding("n", "add_key", "add key"),
        Binding("D", "derive", "derive"),
        Binding("r", "rename", "rename"),
        Binding("ctrl+s", "save", "save"),
        Binding("R", "reload", "reload"),
        Binding("T", "to_task", "→ task"),
        Binding("v", "toggle_preview", "diff/yaml"),
        Binding("slash", "filter", "filter"),
        Binding("escape", "clear_filter", "clear filter", show=False),
    ]

    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        self.current: Path | None = None
        self.preview_mode = "diff"
        self.filter_text = ""

    @property
    def repo(self):
        return self.app.config_repo

    def compose(self) -> ComposeResult:
        yield ViewTabs("configs", id="view-tabs")
        with Horizontal(id="config-main"):
            tree: Tree = Tree("configs", id="config-tree")
            tree.border_title = box_title("configs")
            tree.show_root = False
            yield tree
            with Vertical(id="config-right"):
                filt = Input(placeholder="filter keys…", id="config-filter")
                filt.display = False
                yield filt
                table: DataTable = DataTable(id="config-table", cursor_type="row", zebra_stripes=True)
                table.border_title = box_title("parameters")
                yield table
                preview = Static(id="config-preview")
                preview.border_title = box_title("diff vs parent")
                yield preview
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.add_column("key", key="key")
        table.add_column("value", key="value")
        table.add_column("source", key="source")
        self.rebuild_tree()
        self.query_one(Tree).focus()
        self.app.store.subscribe(self.refresh_last_run)

    def on_unmount(self) -> None:
        self.app.store.unsubscribe(self.refresh_last_run)

    def on_screen_resume(self) -> None:
        self.refresh_last_run()

    # helpers ---------------------------------------------------------------
    def node(self) -> ConfigNode | None:
        if self.current and self.current in self.repo.nodes:
            return self.repo.nodes[self.current]
        return None

    def selected_key(self) -> str | None:
        table = self.query_one(DataTable)
        if table.row_count == 0:
            return None
        try:
            row_key, _ = table.coordinate_to_cell_key(table.cursor_coordinate)
        except Exception:
            return None
        return row_key.value

    def rebuild_tree(self) -> None:
        tree = self.query_one(Tree)
        tree.clear()
        tree.border_title = box_title("configs", info=f"{self.repo.root.name}/")
        target = None
        widest = 0

        def add(parent, node: ConfigNode, depth: int = 0) -> None:
            nonlocal target, widest
            kids = self.repo.children(node)
            label = self.tree_label(node)
            widest = max(widest, (depth + 1) * tree.guide_depth + label.cell_len)
            if kids and depth < 20:
                tn = parent.add(label, data=node.path, expand=True)
                for k in sorted(kids, key=lambda n: n.name):
                    add(tn, k, depth + 1)
            else:
                tn = parent.add_leaf(label, data=node.path)
            if node.path == self.current:
                target = tn

        for root in sorted(self.repo.roots(), key=lambda n: n.name):
            add(tree.root, root)
        # nodes in a cycle have no root; list them flat so they stay visible
        shown = set()

        def collect(n):
            shown.add(n.data)
            for c in n.children:
                collect(c)

        collect(tree.root)
        for n in self.repo.nodes.values():
            if n.path not in shown:
                tn = tree.root.add_leaf(self.tree_label(n), data=n.path)
                if n.path == self.current:
                    target = tn
        # + borders and the scrollbar
        tree.styles.width = max(TREE_MIN_WIDTH, widest + 4)
        if not self.repo.nodes:
            tree.root.add_leaf(Text(f"no *.yaml in {self.repo.root}", style="dim"))
        if target is None and tree.root.children and tree.root.children[0].data:
            target = tree.root.children[0]
        if target is not None:
            self.current = target.data
            self.call_after_refresh(tree.move_cursor, target)
        self.refresh_params()

    def tree_label(self, node: ConfigNode) -> Text:
        return config_label(node)

    def refresh_tree_labels(self) -> None:
        tree = self.query_one(Tree)

        def walk(n):
            if n.data in self.repo.nodes:
                n.set_label(self.tree_label(self.repo.nodes[n.data]))
            for c in n.children:
                walk(c)

        walk(tree.root)

    def refresh_params(self, keep_key: str | None = None) -> None:
        table = self.query_one(DataTable)
        node = self.node()
        keep_key = keep_key or self.selected_key()
        table.clear()
        if node is None:
            table.border_title = box_title("parameters")
            table.border_subtitle = ""
            self.refresh_preview()
            return
        dirty = " *" if node.dirty else ""
        table.border_title = box_title(
            "parameters", info=f"{self.repo.rel(node.path)}{dirty} · {self.estimate_text(node)}")
        table.border_subtitle = self.last_run_text(node)
        if node.error:
            table.add_row(Text("error", style="red"), Text(node.error, style="red"), "", key="__error__")
        for key, value, prov in self.repo.params(node):
            if self.filter_text and self.filter_text.lower() not in key.lower():
                continue
            mark, color = ORIGIN_STYLE.get(prov.origin if prov else Origin.NEW)
            if prov and prov.source == node.path:
                src = Text(f"{mark} {prov.origin.value} (this)", style=color)
            elif prov:
                src = Text(f"{mark} {prov.source.stem}", style=color)
            else:
                src = Text("?", style="dim")
            val_style = "bold" if prov and prov.source == node.path else ""
            table.add_row(Text(key, style=color if val_style else ""),
                          Text(cl.format_value(value), style=val_style), src, key=key)
        if keep_key:
            try:
                table.move_cursor(row=table.get_row_index(keep_key))
            except Exception:
                pass
        self.refresh_preview()

    def refresh_preview(self) -> None:
        preview = self.query_one("#config-preview", Static)
        node = self.node()
        if node is None:
            preview.update("")
            return
        if self.preview_mode == "yaml":
            preview.border_title = box_title("yaml", info=node.path.name)
            preview.update(Syntax(cl.dump_yaml(node.own) if node.own is not None else "",
                                  "yaml", theme="monokai", background_color="default"))
            return
        bases = ", ".join(b.name for b in node.bases) or "none"
        preview.border_title = box_title("diff vs parent", info=bases)
        rows = self.repo.diff(node)
        if not rows:
            msg = "root config — nothing inherited" if not node.bases else "identical to parent"
            preview.update(Text(msg, style="dim"))
            return
        grid = Table.grid(padding=(0, 2))
        grid.add_column(style="bold")
        grid.add_column(style=theme.USED[1])
        grid.add_column(style=theme.SUBTLE)
        grid.add_column(style=theme.OK)
        for key, old, new in rows:
            grid.add_row(
                key,
                "—" if old is MISSING else cl.format_value(old),
                "→",
                "(removed)" if new is MISSING else cl.format_value(new),
            )
        preview.update(grid)

    # events ----------------------------------------------------------------
    def on_tree_node_highlighted(self, event: Tree.NodeHighlighted) -> None:
        if event.node.data and event.node.data != self.current:
            self.current = event.node.data
            self.refresh_params()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        self.action_edit()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "config-filter":
            self.filter_text = event.value
            self.refresh_params()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "config-filter":
            self.query_one(DataTable).focus()

    # actions ---------------------------------------------------------------
    def action_edit(self) -> None:
        node = self.node()
        key = self.selected_key()
        if node is None or key is None or key == "__error__":
            return
        current = cl.get_dotted(node.resolved, key)
        inherited = not self.repo.is_own_key(node, key)

        def done(value) -> None:
            if value is None:
                return
            self.repo.set_value(node, key, value[0])
            self.refresh_params(keep_key=key)
            self.refresh_tree_labels()

        hint = (f"Inherited from {node.provenance[key].source.name} — saving adds an override "
                f"to {node.path.name}." if inherited and key in node.provenance else
                f"Defined in {node.path.name}.")
        self.app.push_screen(
            FormScreen(
                f"Edit {key}",
                [("value", "Value", cl.format_value(current))],
                lambda v: (cl.parse_scalar(v["value"]),),
                hint=hint + "  YAML syntax: 3e-4, true, [1, 2], null",
            ),
            done,
        )

    def action_add_key(self) -> None:
        node = self.node()
        if node is None:
            return

        def build(v):
            key = v["key"].strip()
            if not key or any(not part for part in key.split(".")):
                raise ValueError("key like optimizer.lr is required")
            if key.split(".")[0] in (cl.BASE_KEY,):
                raise ValueError("use Derive to change _base_")
            return key, cl.parse_scalar(v["value"])

        def done(res) -> None:
            if res:
                self.repo.set_value(node, res[0], res[1])
                self.refresh_params(keep_key=res[0])
                self.refresh_tree_labels()

        self.app.push_screen(
            FormScreen(f"Add key to {node.path.name}",
                       [("key", "Key (dotted)", ""), ("value", "Value", "")], build),
            done,
        )

    def action_revert(self) -> None:
        node = self.node()
        key = self.selected_key()
        if node is None or key is None:
            return
        if not node.bases:
            self.notify("Root config: nothing to inherit — edit the value instead.",
                        severity="warning")
            return
        if self.repo.remove_override(node, key):
            self.notify(f"{key} now inherits from parent")
            self.refresh_params(keep_key=key)
            self.refresh_tree_labels()
        else:
            self.notify(f"{key} is not overridden in {node.path.name}", severity="warning")

    def action_derive(self) -> None:
        parent = self.node()
        if parent is None:
            return
        if parent.dirty:
            self.notify("Save the parent first (ctrl+s) so the child inherits saved values.",
                        severity="warning")
            return

        def build(v):
            overrides = {}
            for item in filter(None, (s.strip() for s in v["overrides"].split(";"))):
                if "=" not in item:
                    raise ValueError(f"override {item!r} must look like key=value")
                k, val = item.split("=", 1)
                overrides[k.strip()] = cl.parse_scalar(val)
            return self.repo.derive(parent, v["name"], overrides)

        def done(child) -> None:
            if child is None:
                return
            self.current = child.path
            self.rebuild_tree()
            self.notify(f"Created {child.path.name} (inherits {parent.path.name})",
                        title="Derived")

        self.app.push_screen(
            FormScreen(
                f"Derive from {parent.path.name}",
                [("name", "New name", f"{parent.name}_v2"),
                 ("overrides", "Overrides", "")],
                build,
                hint="Overrides (optional): optimizer.lr=3e-4; data.batch_size=512\n"
                     "The new file stores only _base_ + overrides.",
            ),
            done,
        )

    def action_save(self) -> None:
        n = self.repo.save_all()
        self.notify(f"Saved {n} file(s)" if n else "Nothing to save")
        self.refresh_tree_labels()
        self.refresh_params()

    def estimate_text(self, node: ConfigNode) -> str:
        """Run-time estimate for these exact parameters, from completed runs."""
        if node.error:
            return "no estimate"
        gpus = int((node.resolved.get(cl.SCHEDULER_KEY) or {}).get("gpus", 1))
        past = self.app.store.run_durations().get(self.repo.params_fingerprint(node, gpus), [])
        if not past:
            return "est. no run history"
        median = timedelta(seconds=round(statistics.median(d.total_seconds() for d in past)))
        n = len(past)
        return f"est. ~{format_duration(median)} ({n} run{'s' if n != 1 else ''})"

    def last_run_text(self, node: ConfigNode) -> str:
        """When this config last ran: the latest run of a task made from this file,
        with the same parameters (a config that was moved or renamed), or with the
        config's task name (parameters edited since)."""
        if node.error:
            return ""
        store = self.app.store
        meta = node.resolved.get(cl.SCHEDULER_KEY) or {}
        fp = self.repo.params_fingerprint(node, int(meta.get("gpus", 1)))
        name = str(meta.get("name", node.name))
        tasks = {t.id: t for t in store.tasks.values()
                 if t.config_path == str(node.path) or (t.fingerprint and t.fingerprint == fp)
                 or t.name == name}
        runs = [a for a in store.assignments.values() if a.task_id in tasks and a.actual_start]
        if not runs:
            queued = any(a.task_id in tasks and a.state == AssignmentState.PLANNED
                         for a in store.assignments.values())
            return "never run" + (" · queued" if queued else "")
        a = max(runs, key=lambda a: a.actual_start)
        when = a.actual_start.strftime("%m-%d %H:%M")
        if a.state == AssignmentState.RUNNING:
            text = f"running since {when} ({format_duration(datetime.now() - a.actual_start)})"
        else:
            took = f", {format_duration(a.actual_end - a.actual_start)}" if a.actual_end else ""
            text = f"last run {when} · {a.state.value}{took}"
        task = tasks[a.task_id]
        if task.fingerprint != fp:
            text += " · older parameters"
        elif task.config_path != str(node.path) and task.name != name:
            text += f" · as {task.name}"
        if len(runs) > 1:
            text += f" · {len(runs)} runs"
        return text

    def refresh_last_run(self) -> None:
        """The store changed (a run started or ended): update the line, not the table."""
        node = self.node()
        if self.is_mounted and node is not None:
            self.query_one(DataTable).border_subtitle = self.last_run_text(node)

    def action_rename(self) -> None:
        node = self.node()
        if node is None:
            return

        def done(new_name) -> None:
            if not new_name:
                return
            try:
                old = node.path
                new, children = self.repo.rename(node, new_name)
            except Exception as e:
                self.notify(str(e), severity="error")
                return
            tasks = self.app.config_renamed(old, new)
            self.current = new
            self.rebuild_tree()
            extra = []
            if children:
                extra.append(f"updated _base_ in {', '.join(children)}")
            if tasks:
                extra.append(f"repointed {tasks} task(s)")
            self.notify(f"Renamed to {new.name}" + (f" ({'; '.join(extra)})" if extra else ""),
                        title="Renamed")

        self.app.push_screen(
            FormScreen(f"Rename {node.path.name}", [("name", "New name", node.name)],
                       lambda v: v["name"].strip() or None,
                       hint="Children that inherit from it are updated to the new name."),
            done,
        )

    def action_reload(self) -> None:
        dirty = [n.name for n in self.repo.nodes.values() if n.dirty]
        self.repo.reload()
        self.rebuild_tree()
        msg = "Reloaded from disk"
        if dirty:
            msg += f" (discarded edits in {', '.join(dirty)})"
        self.notify(msg, severity="warning" if dirty else "information")

    def action_to_task(self) -> None:
        node = self.node()
        if node is None:
            return
        if node.dirty:
            self.notify("Save first (ctrl+s): the task runs the file on disk.", severity="warning")
            return
        try:
            task = self.repo.make_task(node, self.app.command_template)
        except Exception as e:
            self.notify(str(e), severity="error")
            return
        self.app.store.add_task(task)
        self.notify(f"Task {task.name} added (GPU×{task.required_gpus}). Press 1 to schedule it.",
                    title="Task created")

    def action_toggle_preview(self) -> None:
        self.preview_mode = "yaml" if self.preview_mode == "diff" else "diff"
        self.refresh_preview()

    def action_filter(self) -> None:
        f = self.query_one("#config-filter", Input)
        f.display = True
        f.focus()

    def action_clear_filter(self) -> None:
        f = self.query_one("#config-filter", Input)
        if f.display:
            f.value = ""
            f.display = False
            self.filter_text = ""
            self.refresh_params()
            self.query_one(DataTable).focus()
        else:
            self.app.action_menu()
