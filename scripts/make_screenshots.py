"""Make the README screenshots: docs/images/*.png.

    python scripts/make_screenshots.py [--browser PATH]

Builds a demo scene (made-up tasks, GPUs and host name, no real processes),
renders the TUI to SVG with Textual, and turns each SVG into a sharp 2x PNG
with a headless Chromium browser (Edge or Chrome; found automatically, or pass
--browser), or with rsvg-convert where no Chromium is installed.
Re-run it after changing the UI.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "images"
SIZE = (190, 52)

os.environ["ZGM_MOCK_EXTERNAL"] = "3:30500:4242"  # another program on GPU 3
socket.gethostname = lambda: "node07"  # the host panel's title: not this machine's name

from zgm.app import ZGMApp  # noqa: E402
from zgm.models import Assignment, AssignmentState, Task, TaskStatus  # noqa: E402
from zgm.runtime.base import RunHandle, RunState  # noqa: E402
from zgm.services.scheduler import AssignmentService  # noqa: E402
from zgm.services.task_store import TaskStore, add_completed_run  # noqa: E402
from zgm.storage.sqlite import SqliteStorage  # noqa: E402


class StillRunning:
    """A runtime whose runs never end: the demo's running tasks stay running."""

    name = "local"
    log_dir = None

    def available(self):
        return True

    def attach(self, assignment):
        return RunHandle(assignment.id, None)

    def status(self, handle):
        return RunState.RUNNING

    def submit(self, task, assignment):
        raise RuntimeError("demo only")

    def cancel(self, handle):
        pass


def scene(db: Path) -> None:
    """Some history, two jobs running, a queue, and tasks waiting to be placed."""
    store = TaskStore(SqliteStorage(db))
    now = datetime.now().replace(second=0, microsecond=0)

    def task(name, command, gpus=1):
        t = Task(name, command, gpus)
        t.created_at = now - timedelta(minutes=30)
        return t

    resnet = task("resnet50-baseline", "python train.py --config configs/resnet50_base.yaml")
    vit = task("vit-b16-pretrain", "python train.py --model vit_b16 --epochs 90", 2)
    bert = task("bert-finetune-sst2", "python finetune.py --task sst2")
    evalc = task("eval-imagenet-c", "python eval.py --suite imagenet-c")
    diff = task("diffusion-ablation", "python ddpm.py --ablate attn", 2)
    llama = task("llama-lora-sweep", "python lora.py --sweep configs/lora.yaml", 4)
    clip = task("clip-zero-shot", "python clip_eval.py --datasets all")

    # earlier runs: they make the estimates, and the recent ones show as history
    add_completed_run(store, resnet, (0,), now - timedelta(days=2), timedelta(hours=2, minutes=50))
    add_completed_run(store, resnet, (1,), now - timedelta(days=1), timedelta(hours=3, minutes=5))
    add_completed_run(store, bert, (2,), now - timedelta(days=1), timedelta(minutes=85))
    add_completed_run(store, diff, (1, 2), now - timedelta(days=3), timedelta(hours=2))
    add_completed_run(store, task("data-preprocess", "python prep.py --shards 64"), (3,),
                      now - timedelta(hours=6, minutes=30), timedelta(hours=1, minutes=30))
    add_completed_run(store, task("tokenize-corpus", "python tok.py"), (0,),
                      now - timedelta(hours=5, minutes=40), timedelta(hours=1, minutes=10))
    # short runs (minutes): visible in the time view, readable cards in the task view
    for name, command, gpus, ago, minutes in (
            ("eval-cifar10-c", "python eval.py --suite cifar10-c", (0,), 4 * 60 + 10, 4),
            ("eval-cifar10-c", "python eval.py --suite cifar10-c", (0,), 3 * 60 + 50, 3),
            ("ckpt-export", "python export.py --fmt safetensors", (1,), 3 * 60, 2),
            ("vit-smoke-test", "python train.py --model vit_b16 --steps 200", (1, 2),
             5 * 60 + 30, 35),
            ("data-shard-check", "python prep.py --verify", (3,), 4 * 60, 6)):
        add_completed_run(store, task(name, command, len(gpus)), gpus,
                          now - timedelta(minutes=ago), timedelta(minutes=minutes))
    failed = store.add_task(task("bert-large-lr1e-3", "python finetune.py --lr 1e-3"))
    failed.status = TaskStatus.FAILED
    store.update_task(failed, quiet=True)
    start = now - timedelta(hours=4, minutes=20)
    store.save_assignment(Assignment(failed.id, (2,), start, start + timedelta(minutes=50),
                                     actual_start=start, actual_end=start + timedelta(minutes=50),
                                     state=AssignmentState.FAILED), "failed")

    for t in (resnet, vit, bert, evalc, diff, llama, clip):
        store.add_task(t)
    for t, gpus, minutes in ((resnet, (0,), 95), (vit, (1, 2), 150)):
        began = now - timedelta(minutes=minutes)
        a = Assignment(t.id, gpus, began, None, actual_start=began, actual_gpu_ids=gpus,
                       state=AssignmentState.RUNNING)
        store.save_assignment(a, "started")
        t.status = TaskStatus.RUNNING
        store.update_task(t, quiet=True)
    svc = AssignmentService(store, [0, 1, 2, 3])
    svc.assign(bert.id, (0,))
    svc.assign(diff.id, (1, 2))
    svc.assign(evalc.id, (3,))
    store.notify()


def make_app(db: Path, configs: Path) -> ZGMApp:
    app = ZGMApp(storage=SqliteStorage(db), mock_gpus=4, config_dir=configs,
                 stats_interval=0, runtime=StillRunning(), card_width=18)
    app.gpu_source = "NVML"
    for g, (used, util) in zip(app.gpus, [(61, 92), (71, 88), (70, 87), (31, 46)]):
        g.name = "NVIDIA A100-SXM4-80GB"
        g.memory_used, g.utilization = used * 1024**3, util
    for proc in app.provider.fake_processes.get(3, []):
        proc.name = "python"
    return app


async def shoot(svg_dir: Path, configs: Path) -> list[Path]:
    db = svg_dir / "demo.db"
    scene(db)
    app = make_app(db, configs)
    shots = []
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause()
        await pilot.press("t", "down")  # the timeline (time view), GPU 1
        await pilot.press("left", "left", "left")  # run by run, back to vit-smoke-test
        await pilot.pause()
        shots.append(Path(app.save_screenshot("schedule.svg", str(svg_dir))))
        await pilot.press("t")  # the task view: one card per run
        await pilot.pause()
        shots.append(Path(app.save_screenshot("tasks.svg", str(svg_dir))))
        app.screen.queue.focus()
        await pilot.press("right", "right")  # a waiting task (clip-zero-shot)
        await pilot.press("a", "down", "down")  # place it: right after now, then on GPU 2
        await pilot.pause()
        shots.append(Path(app.save_screenshot("placing.svg", str(svg_dir))))
        await pilot.press("escape", "2")
        await pilot.pause()
        screen = app.screen
        screen.current = app.config_repo.get("resnet50_adamw_bs512").path
        screen.rebuild_tree()
        screen.refresh_params()
        await pilot.pause()
        shots.append(Path(app.save_screenshot("configs.svg", str(svg_dir))))
    return shots


def find_browser(given: str | None) -> str | None:
    """A Chromium browser, or None to fall back to rsvg-convert."""
    candidates = [given, shutil.which("msedge"), shutil.which("chrome"),
                  shutil.which("google-chrome"), shutil.which("chromium"),
                  r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                  r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"]
    for c in candidates:
        if c and Path(c).exists():
            return c
    if shutil.which("rsvg-convert"):
        return None
    sys.exit("no Chromium browser and no rsvg-convert found: pass --browser PATH")


def to_png(browser: str | None, svg: Path, png: Path) -> None:
    if browser is None:  # rsvg-convert: the terminal font comes from the system
        subprocess.run(["rsvg-convert", "-z", "2", "-b", "black", str(svg), "-o", str(png)],
                       check=True, capture_output=True, timeout=120)
        if shutil.which("convert"):  # a terminal has few colours: a palette PNG is ~4x smaller
            subprocess.run(["convert", str(png), "+dither", "-colors", "256", f"PNG8:{png}"],
                           check=True, capture_output=True, timeout=120)
        return
    text = svg.read_text(encoding="utf-8")
    width, height = (float(v) for v in re.search(
        r'viewBox="0 0 ([\d.]+) ([\d.]+)"', text).groups())
    page = svg.with_suffix(".html")
    page.write_text("<!doctype html><meta charset=utf-8><style>html,body{margin:0;"
                    "background:#000}svg{display:block}</style>" + text, encoding="utf-8")
    png.unlink(missing_ok=True)
    # its own profile: otherwise the call may be handed to a browser already open
    profile = svg.parent / f"profile-{svg.stem}"
    subprocess.run([browser, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                    f"--user-data-dir={profile}", "--no-first-run",
                    "--force-device-scale-factor=2", "--virtual-time-budget=5000",
                    f"--window-size={int(width)},{int(height)}", f"--screenshot={png}",
                    page.as_uri()], check=True, capture_output=True, timeout=120)
    # Edge's launcher returns before the browser has written the file
    deadline, size = time.monotonic() + 60, -1
    while time.monotonic() < deadline:
        if png.exists() and png.stat().st_size == size > 0:
            return
        size = png.stat().st_size if png.exists() else -1
        time.sleep(0.5)
    sys.exit(f"{browser} did not write {png}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--browser", help="Chromium-based browser to render with")
    args = parser.parse_args()
    browser = find_browser(args.browser)
    OUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:  # Windows: db still open
        tmp = Path(tmp)
        configs = tmp / "configs"
        configs.mkdir()
        tracked = subprocess.run(["git", "ls-files", "examples/configs"], cwd=ROOT,
                                 capture_output=True, text=True, check=True).stdout.split()
        for name in tracked:  # only the examples in the repository
            shutil.copy(ROOT / name, configs)
        for svg in asyncio.run(shoot(tmp, configs)):
            png = OUT / svg.with_suffix(".png").name
            to_png(browser, svg, png)
            print(f"{png.relative_to(ROOT)}  {png.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
