"""TextToggle: a yes/no setting shown as the words ``True  False``, the chosen
one highlighted. Easier to read on a dark terminal than a switch."""

from __future__ import annotations

from rich.text import Text
from textual import events
from textual.binding import Binding
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Static

from . import theme


class TextToggle(Static, can_focus=True):
    BINDINGS = [
        Binding("enter,space", "toggle", "toggle", show=False),
        Binding("left,right,h,l", "toggle", "toggle", show=False),
    ]

    value: reactive[bool] = reactive(False)

    class Changed(Message):
        def __init__(self, toggle: TextToggle, value: bool) -> None:
            super().__init__()
            self.toggle, self.value = toggle, value

        @property
        def control(self) -> TextToggle:
            return self.toggle

    def __init__(self, value: bool = False, **kw) -> None:
        super().__init__(**kw)
        self.set_reactive(TextToggle.value, value)

    def render(self) -> Text:
        out = Text(no_wrap=True)
        for word, on in (("True", True), ("False", False)):
            if on == self.value:
                bg = theme.OK if on else theme.WARN
                out.append(f" {word} ", style=f"bold {theme.MAIN_BG} on {bg}")
            else:
                out.append(f" {word} ", style=theme.SUBTLE)
            out.append(" ")
        if self.has_focus:
            out.append("◀ ▶ enter", style=theme.HI_FG)
        return out

    def action_toggle(self) -> None:
        self.value = not self.value

    def watch_value(self, value: bool) -> None:
        self.post_message(self.Changed(self, value))

    def on_click(self, event: events.Click) -> None:
        self.action_toggle()

    def on_focus(self) -> None:
        self.refresh()

    def on_blur(self) -> None:
        self.refresh()
