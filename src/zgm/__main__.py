"""Command-line entry point: ``zgm`` or ``python -m zgm``."""

from __future__ import annotations

import argparse

from .services.config_repo import DEFAULT_COMMAND_TEMPLATE
from .storage.sqlite import SqliteStorage, default_db_path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="zgm", description=__doc__)
    p.add_argument("--db", default=str(default_db_path()),
                   help="SQLite file for tasks/plans/history (default: %(default)s)")
    p.add_argument("--mock-gpus", type=int, default=None, metavar="N",
                   help="use N fake GPUs instead of NVML (also: ZGM_MOCK_GPUS)")
    p.add_argument("--config-dir", default="configs",
                   help="directory of experiment YAML configs (default: ./configs)")
    p.add_argument("--command-template", default=DEFAULT_COMMAND_TEMPLATE,
                   help="command for tasks created from a config; {path} and {name} are "
                        "substituted (default: %(default)r)")
    p.add_argument("--demo", action="store_true",
                   help="seed example tasks when the database is empty")
    p.add_argument("--runtime", choices=["none", "local", "mock", "ray"], default="none",
                   help="execution backend for the 'r' key: local = child processes on "
                        "this machine (default: none = planning only)")
    p.add_argument("--no-autostart", dest="autostart", action="store_false",
                   help="with a runtime, don't start queued tasks when their GPUs free up; "
                        "start them with r, or turn auto-start on with p")
    p.add_argument("--log-dir", default=None,
                   help="where the local runtime writes task logs "
                        "(default: logs/ next to the database)")
    p.add_argument("--card-width", type=int, default=24, metavar="N",
                   help="width of queued task cards on the timeline (min 12, default 24)")
    p.add_argument("--ray-address", default=None, help="Ray cluster address (default: local)")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    runtime = None
    if args.runtime == "local":
        from pathlib import Path

        from .runtime.local import LocalRuntime

        runtime = LocalRuntime(args.log_dir or Path(args.db).expanduser().parent / "logs")
    elif args.runtime == "mock":
        from .runtime.mock import MockRuntime

        runtime = MockRuntime()
    elif args.runtime == "ray":
        from .runtime.ray_adapter import RayRuntime

        runtime = RayRuntime(address=args.ray_address)
        if not runtime.available():
            raise SystemExit("ray is not installed: pip install 'zgm[ray]'")

    from .app import ZGMApp

    app = ZGMApp(
        storage=SqliteStorage(args.db),
        mock_gpus=args.mock_gpus,
        config_dir=args.config_dir,
        demo=args.demo,
        runtime=runtime,
        command_template=args.command_template,
        card_width=args.card_width,
        autostart=args.autostart,
    )
    app.run()


if __name__ == "__main__":
    main()
