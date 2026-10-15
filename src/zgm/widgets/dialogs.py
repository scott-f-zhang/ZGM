"""Small modal forms: new task, edit a value, name a derived config."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Grid, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Static

from ..models import Task, command_fingerprint


class FormScreen(ModalScreen[Any]):
    """Generic labelled-input form; ``validate`` turns values into a result or raises."""

    BINDINGS = [Binding("escape", "cancel", "Cancel")]
    DEFAULT_CSS = """
    FormScreen { align: center middle; }
    FormScreen > Vertical {
        width: 72; height: auto; padding: 1 2;
        border: round $accent; background: $panel;
    }
    FormScreen .form-title { text-style: bold; margin-bottom: 1; }
    FormScreen Grid { grid-size: 2; grid-columns: 16 1fr; grid-rows: 3; height: auto; }
    FormScreen Grid Label { padding: 1 0 0 0; }
    FormScreen .form-error { color: $error; height: auto; }
    FormScreen .form-hint { color: $text-muted; height: auto; }
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
        with Vertical():
            yield Static(self.title_text, classes="form-title")
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


def _task_fields(v: dict[str, str]) -> tuple[str, str, int]:
    name = v["name"].strip()
    if not name:
        raise ValueError("name is required")
    command = v["command"].strip()
    if not command:
        raise ValueError("command is required")
    try:
        gpus = int(v["gpus"])
    except ValueError:
        raise ValueError("GPUs must be an integer") from None
    if gpus < 1:
        raise ValueError("GPUs must be at least 1")
    return name, command, gpus


ESTIMATE_HINT = ("No duration to enter: the estimate comes from past runs of the same "
                 "command/config, or shows 'no data'.")


def new_task_form() -> FormScreen:
    def build(v: dict[str, str]) -> Task:
        name, command, gpus = _task_fields(v)
        return Task(name=name, command=command, required_gpus=gpus)

    return FormScreen(
        "New task",
        [("name", "Name", ""), ("command", "Command", "python train.py"), ("gpus", "GPUs", "1")],
        build,
        hint=ESTIMATE_HINT,
    )


def edit_task_form(task: Task, name_only: bool = False) -> FormScreen:
    """Rename/edit a task. Returns (name, command, gpus) or None."""

    def build(v: dict[str, str]) -> tuple[str, str, int]:
        name, command, gpus = _task_fields(v)
        if name_only and (command != task.command or gpus != task.required_gpus):
            raise ValueError("the task is running: only its name can change")
        return name, command, gpus

    hint = "Running: only the name can change." if name_only else ESTIMATE_HINT
    return FormScreen(
        f"Edit {task.name}",
        [("name", "Name", task.name), ("command", "Command", task.command),
         ("gpus", "GPUs", str(task.required_gpus))],
        build,
        hint=hint,
    )


def apply_task_edit(task: Task, name: str, command: str, gpus: int) -> None:
    task.name = name
    if command != task.command or gpus != task.required_gpus:
        task.command, task.required_gpus = command, gpus
        if not task.config_path:  # config tasks are keyed by their parameters instead
            task.fingerprint = command_fingerprint(command, gpus)


class ChoiceScreen(ModalScreen[str | None]):
    """A question with a few buttons; dismisses with the chosen key (None = Esc)."""

    BINDINGS = [Binding("escape", "cancel", "Cancel")]
    DEFAULT_CSS = """
    ChoiceScreen { align: center middle; }
    ChoiceScreen > Vertical {
        width: 64; height: auto; padding: 1 2;
        border: round $warning; background: $panel;
    }
    ChoiceScreen .choice-title { text-style: bold; margin-bottom: 1; }
    ChoiceScreen .choice-body { color: $text-muted; height: auto; margin-bottom: 1; }
    ChoiceScreen Vertical Button { width: 100%; margin-top: 1; }
    """

    def __init__(self, title: str, body: str, choices: list[tuple[str, str, str]]) -> None:
        """``choices``: (key, label, button variant)."""
        super().__init__()
        self.title_text, self.body, self.choices = title, body, choices

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static(self.title_text, classes="choice-title")
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
