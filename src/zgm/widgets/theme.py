"""Colours and box drawing after btop's default theme.

Values come from btop's ``Default_theme`` (src/btop_theme.cpp in
github.com/aristocratos/btop). The way they are used follows btop's drawing code:
rounded boxes, each in its own muted colour, with the title set into the top
line between ``┐`` and ``┌`` (``╭─┐title┌───╮``), key numbers as red
superscripts, and ``■`` meters whose cells take a three-stop gradient.

Fonts and font size belong to the terminal, not to zgm: set them there (btop's
README recommends the same).
"""

from __future__ import annotations

from rich.segment import Segment
from rich.style import Style

# btop Default_theme --------------------------------------------------------------------
MAIN_BG = "#000000"
MAIN_FG = "#cccccc"
TITLE = "#eeeeee"
HI_FG = "#b54040"  # key hints
SELECTED_BG = "#6a2f2f"
SELECTED_FG = "#eeeeee"
INACTIVE_FG = "#404040"
GRAPH_TEXT = "#606060"
METER_BG = "#404040"
DIV_LINE = "#303030"
PROC_MISC = "#0de756"

CPU_BOX = "#556d59"
MEM_BOX = "#6c6c4b"
NET_BOX = "#5c588d"
PROC_BOX = "#805252"

CPU = ("#77ca9b", "#cbc06c", "#dc4c4c")
USED = ("#592b26", "#d9626d", "#ff4769")
FREE = ("#384f21", "#b5e685", "#dcff85")
CACHED = ("#163350", "#74e6fc", "#26c5ff")
AVAILABLE = ("#4e3f0e", "#ffd77a", "#ffb814")
TEMP = ("#4897d4", "#5474e8", "#ff40b6")
DOWNLOAD = ("#291f75", "#4f43a3", "#b0a9de")
UPLOAD = ("#620665", "#7d4180", "#dcafde")
PROCESS = ("#80d0a3", "#dcd179", "#d45454")

# zgm's uses of them -----------------------------------------------------------------------
# a box's colour, and the brighter one it takes while it has focus
BOX_QUEUE = (CPU_BOX, CPU[0])
BOX_DETAILS = (MEM_BOX, FREE[1])
BOX_TIMELINE = (PROC_BOX, PROCESS[2])
BOX_LIST = (NET_BOX, DOWNLOAD[2])

OK = CPU[0]  # green: running, ready
WARN = CPU[1]  # yellow: waiting, paused
BAD = CPU[2]  # red: failed, in use elsewhere
SUBTLE = GRAPH_TEXT  # secondary text
PAST_BG = "#0a0a0a"  # the history half of the timeline
NOW = CPU[2]

SUPERSCRIPT = "⁰¹²³⁴⁵⁶⁷⁸⁹"


def _rgb(color: str) -> tuple[int, int, int]:
    c = color.lstrip("#")
    return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)


def gradient(stops: tuple[str, ...], fraction: float) -> str:
    """The colour at ``fraction`` (0..1) of a start→mid→end gradient, as btop
    builds it: start→mid over the first half, mid→end over the second."""
    f = min(1.0, max(0.0, fraction))
    if len(stops) == 3:
        a, b, f = (stops[0], stops[1], f * 2) if f <= 0.5 else (stops[1], stops[2], f * 2 - 1)
    else:
        a, b = stops[0], stops[-1]
    (r1, g1, b1), (r2, g2, b2) = _rgb(a), _rgb(b)
    return "#{:02x}{:02x}{:02x}".format(round(r1 + (r2 - r1) * f), round(g1 + (g2 - g1) * f),
                                        round(b1 + (b2 - b1) * f))


def meter(fraction: float, width: int, stops: tuple[str, ...],
          base: Style | None = None) -> list[Segment]:
    """btop's meter: ``width`` ■ cells; filled ones take the gradient colour of
    their position, the rest are dark."""
    base = base or Style()
    filled = round(min(1.0, max(0.0, fraction)) * width)
    segs = [Segment("■", base + Style(color=gradient(stops, (i + 1) / width)))
            for i in range(filled)]
    if filled < width:
        segs.append(Segment("■" * (width - filled), base + Style(color=METER_BG)))
    return segs


def badge(bg: str, fg: str = MAIN_BG) -> str:
    """Rich style for a status badge (" AUTO ▶ ")."""
    return f"bold {fg} on {bg}"


def hints(pairs: list[tuple[str, str]]):
    """Key hints as btop writes them: the key in red, what it does dim."""
    from rich.text import Text

    out = Text(no_wrap=True)
    for i, (key, what) in enumerate(pairs):
        if i:
            out.append("  ")
        out.append(key, style=f"bold {HI_FG}")
        out.append(f" {what}", style=SUBTLE)
    return out


def sup(n: int) -> str:
    return "".join(SUPERSCRIPT[int(d)] for d in str(n))


def box_title(name: str, key: int | str | None = None, info: str = "") -> str:
    """Markup for a btop-style title in a box's top line: ``─┐¹name┌``.
    ``info`` follows in dim text inside the same notch. A letter ``key`` that
    starts ``name`` is shown as that letter in the key colour (``─┐queue┌``, q red)."""
    from textual.markup import escape

    if isinstance(key, str) and name.startswith(key):
        extra = f" [{SUBTLE}]{escape(info)}[/]" if info else ""
        return (f"─┐[b {HI_FG}]{escape(key)}[/][b {TITLE}]{escape(name[len(key):])}[/]"
                f"{extra}┌")
    num = f"[{HI_FG}]{sup(key) if isinstance(key, int) else key}[/]" if key is not None else ""
    extra = f" [{SUBTLE}]{escape(info)}[/]" if info else ""
    return f"─┐{num}[b {TITLE}]{escape(name)}[/]{extra}┌"


def box_footer(text: str) -> str:
    """Markup for the bottom line (right-aligned): ``┘text└─``."""
    from textual.markup import escape

    return f"┘[{SUBTLE}]{escape(text)}[/]└─"


def install_box_titles() -> None:
    """Let ``┐title┌`` / ``┘text└`` labels touch the box line, like btop.

    Textual pads every border label with a space on both sides
    (``╭─ title ─╮``). For labels in btop's notch form only, ask Textual's
    label renderer to skip that padding. If Textual's internals change, this
    does nothing and the titles keep the padding.
    """
    try:
        from textual import _styles_cache
    except ImportError:
        return
    original = getattr(_styles_cache, "render_border_label", None)
    if original is None or getattr(original, "_zgm_btop", False):
        return

    def render_border_label(label, is_title, name, width, inner_style, outer_style, style,
                            has_left_corner, has_right_corner):
        plain = getattr(label[0], "plain", "").lstrip("─")
        if plain[:1] in ("┐", "┘"):
            has_left_corner = has_right_corner = False
        return original(label, is_title, name, width, inner_style, outer_style, style,
                        has_left_corner, has_right_corner)

    render_border_label._zgm_btop = True
    _styles_cache.render_border_label = render_border_label


def textual_theme():
    """The same palette for Textual's own widgets (footer, inputs, tables, trees)."""
    from textual.theme import Theme

    return Theme(
        name="btop",
        primary=CPU_BOX,
        secondary=NET_BOX,
        accent=HI_FG,
        foreground=MAIN_FG,
        background=MAIN_BG,
        surface=MAIN_BG,
        panel="#141414",
        success=OK,
        warning=WARN,
        error=BAD,
        dark=True,
        variables={
            "footer-background": MAIN_BG,
            "footer-key-foreground": HI_FG,
            "footer-key-background": MAIN_BG,
            "footer-description-foreground": SUBTLE,
            "footer-description-background": MAIN_BG,
            "footer-item-background": MAIN_BG,
            "block-cursor-background": SELECTED_BG,
            "block-cursor-foreground": SELECTED_FG,
            "block-cursor-text-style": "bold",
            "block-cursor-blurred-background": "#2a1a1a",
            "block-cursor-blurred-foreground": MAIN_FG,
            "input-cursor-background": MAIN_FG,
            "input-selection-background": SELECTED_BG,
            "border": CPU[0],
            "border-blurred": INACTIVE_FG,
            "scrollbar": DIV_LINE,
            "scrollbar-hover": INACTIVE_FG,
            "scrollbar-active": GRAPH_TEXT,
            "scrollbar-background": MAIN_BG,
            "scrollbar-background-hover": MAIN_BG,
            "scrollbar-background-active": MAIN_BG,
            "scrollbar-corner-color": MAIN_BG,
        },
    )
