"""Small modal forms: new task, edit a value, name a derived config."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Grid, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Static

from ..models import LaunchSpec, Task, launch_fingerprint
from .theme import box_title


class FormScreen(ModalScreen[Any]):
    """Generic labelled-input form; ``validate`` turns values into a result or raises."""

    BINDINGS = [Binding("escape", "cancel", "cancel"), Binding("ctrl+s", "submit", "ok")]
    DEFAULT_CSS = """
    FormScreen { align: center middle; }
    FormScreen > Vertical {
        width: 84; max-width: 100%; height: auto; max-height: 100%; overflow-y: auto;
        padding: 1 2;
        border: round #556d59; border-title-color: #556d59; background: #000000;
    }
    FormScreen Grid { grid-size: 2; grid-columns: 16 1fr; grid-rows: 3; height: auto; }
    FormScreen Grid Label { padding: 1 0 0 0; }
    FormScreen .form-error { color: $error; height: auto; }
    FormScreen .form-hint { color: #606060; height: auto; }
    FormScreen Horizontal { height: auto; align-horizontal: right; margin-top: 1; }
    FormScreen Button { margin-left: 1; }
    """

    def __init__(
        self,
        title: str,
        fields: list[tuple[str, str, str]],  # (key, label, initial)
        validate: Callable[[dict[str, str]], Any],
        hint: str = "",
    ) -> None:
        super().__init__()
        self.title_text = title
        self.fields = fields
        self.validate_fn = validate
        self.hint = hint

    def compose(self) -> ComposeResult:
        with Vertical() as box:
            box.border_title = box_title(self.title_text.lower())
            with Grid():
                for key, label, initial in self.fields:
                    yield Label(label)
                    yield Input(value=initial, id=f"f-{key}")
            if self.hint:
                yield Static(self.hint, classes="form-hint")
            yield Static("", classes="form-error", id="form-error")
            with Horizontal():
                yield Button("Cancel", id="cancel")
                yield Button("OK", variant="primary", id="ok")

    def on_mount(self) -> None:
        first = self.query(Input).first()
        first.focus()
        first.action_end()

    def values(self) -> dict[str, str]:
        return {key: self.query_one(f"#f-{key}", Input).value for key, _, _ in self.fields}

    def submit(self) -> None:
        try:
            result = self.validate_fn(self.values())
        except Exception as e:  # show validation problems inline
            self.query_one("#form-error", Static).update(f"✗ {e}")
            return
        self.dismiss(result)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        inputs = list(self.query(Input))
        idx = inputs.index(event.input)
        if idx + 1 < len(inputs):
            inputs[idx + 1].focus()
        else:
            self.submit()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "ok":
            self.submit()
        else:
            self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)

    def action_submit(self) -> None:
        self.submit()


def _task_fields(v: dict[str, str]) -> tuple[str, LaunchSpec, int]:
    name = v["name"].strip()
    if not name:
        raise ValueError("name is required")
    try:
        gpus = int(v["gpus"])
    except ValueError:
        raise ValueError("GPUs must be an integer") from None
    if gpus < 1:
        raise ValueError("GPUs must be at least 1")
    cwd = v["cwd"].strip()
    if cwd:
        path = Path(cwd).expanduser()
        if not path.is_dir():
            raise ValueError(f"working directory not found: {cwd}")
        cwd = str(path.resolve())
    spec = LaunchSpec.from_text(v["run"], v["environment"], cwd, v["env"])
    return name, spec, gpus


TASK_HINT = ("Run: 'train.py ...' / '-m pkg ...' use the environment's Python; anything "
             "else is a shell command. Environment: python path, 'conda:NAME', or empty "
             "(python on PATH). Env vars: 'A=1; B=2'. Ctrl+S saves.")


def _task_form_fields(task: Task | None) -> list[tuple[str, str, str]]:
    spec = task.launch if task else None
    return [
        ("name", "Name", task.name if task else ""),
        ("gpus", "GPUs", str(task.required_gpus) if task else "1"),
        ("run", "Run", spec.target if spec else "train.py"),
        ("environment", "Environment", spec.environment if spec else ""),
        ("cwd", "Working dir", (spec.cwd or "") if spec else str(Path.cwd())),
        ("env", "Env vars", spec.env_text if spec else ""),
    ]


def new_task_form() -> FormScreen:
    def build(v: dict[str, str]) -> Task:
        name, spec, gpus = _task_fields(v)
        return Task(name=name, launch=spec, required_gpus=gpus)

    return FormScreen("New task", _task_form_fields(None), build, hint=TASK_HINT)


def edit_task_form(task: Task, name_only: bool = False) -> FormScreen:
    """Rename/edit a task. Returns (name, launch spec, gpus) or None."""

    def build(v: dict[str, str]) -> tuple[str, LaunchSpec, int]:
        name, spec, gpus = _task_fields(v)
        if name_only and (spec != task.launch or gpus != task.required_gpus):
            raise ValueError("the task is running: only its name can change")
        return name, spec, gpus

    hint = "Running: only the name can change." if name_only else TASK_HINT
    return FormScreen(f"Edit {task.name}", _task_form_fields(task), build, hint=hint)


def apply_task_edit(task: Task, name: str, spec: LaunchSpec, gpus: int) -> None:
    task.name = name
    if spec != task.launch or gpus != task.required_gpus:
        task.launch, task.required_gpus = spec, gpus
        task.command = spec.describe()
        if not task.config_path:  # config tasks are keyed by their parameters instead
            task.fingerprint = launch_fingerprint(spec, gpus)


class ChoiceScreen(ModalScreen[str | None]):
    """A question with a few buttons; dismisses with the chosen key (None = Esc)."""

    BINDINGS = [Binding("escape", "cancel", "cancel")]
    DEFAULT_CSS = """
    ChoiceScreen { align: center middle; }
    ChoiceScreen > Vertical {
        width: 64; height: auto; padding: 1 2;
        border: round #805252; border-title-color: #805252; background: #000000;
    }
    ChoiceScreen .choice-body { color: #606060; height: auto; margin-bottom: 1; }
    ChoiceScreen Vertical Button { width: 100%; margin-top: 1; }
    """

    def __init__(self, title: str, body: str, choices: list[tuple[str, str, str]]) -> None:
        """``choices``: (key, label, button variant)."""
        super().__init__()
        self.title_text, self.body, self.choices = title, body, choices

    def compose(self) -> ComposeResult:
        with Vertical() as box:
            box.border_title = box_title(self.title_text)
            if self.body:
                yield Static(self.body, classes="choice-body")
            for key, label, variant in self.choices:
                yield Button(label, id=f"choice-{key}", variant=variant)

    def on_mount(self) -> None:
        self.query(Button).first().focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id.removeprefix("choice-"))

    def action_cancel(self) -> None:
        self.dismiss(None)
