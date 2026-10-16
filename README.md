# ZGM

A btop-style terminal UI for planning GPU work on one machine. Task cards sit at the top, details in the middle, and a per-GPU Gantt timeline at the bottom. You place tasks on GPUs and time slots with the keyboard or the mouse. A second view manages experiment YAML configs, which can inherit from each other like classes.

On its own, the TUI only **plans**: an assignment is a planned slot, and moving it never starts or kills a process. To **run** the queue, start `zgm daemon start`, or open the TUI with `--runtime local`. Each task then starts on its planned GPUs as soon as they are free, and keeps running after zgm exits (see [Running your code](#running-your-code), [Auto-start](#auto-start) and [In the background](#in-the-background-zgm-daemon)). Execution goes through a `Runtime` interface; local, mock and Ray backends are included.

```bash
pip install -e '.[dev]'            # add [ray] for the Ray runtime
zgm --demo --config-dir examples/configs
zgm daemon start                                    # run the queue in the background
zgm --config-dir examples/configs --runtime local   # ...or from the TUI, while it is open
zgm submit -g 2 -- train.py --lr 3e-4                # queue a task from your project dir
# or: python -m zgm ...
```

Without NVIDIA drivers or NVML, the app falls back to 4 mock GPUs. You can force a count with `--mock-gpus N` or by setting `ZGM_MOCK_GPUS=N`.

| flag | meaning |
|---|---|
| `--db PATH` | SQLite file for tasks, plans and history (default `~/.local/share/zgm/state.db`) |
| `--config-dir DIR` | experiment YAML configs (default `./configs`) |
| `--command-template T` | shell command for tasks made from a config that has no `_scheduler_.run` block; `{path}` and `{name}` are substituted |
| `--demo` | seed example tasks into an empty database |
| `--runtime none\|local\|mock\|ray` | run the queue from this TUI when no `zgm daemon` does (`local`: processes on this machine). With a daemon running, the TUI is its client either way |
| `--no-autostart` | pause auto-start: tasks only start with `r` until `p` turns it on (the setting is shared with the daemon) |
| `--log-dir DIR` | where the local runtime writes task logs (default `logs/` next to the database) |

## Views (btop-style)

`1` / `F1` opens **Schedule**. `2` / `F2` opens **Configs**. You can also click the tabs. Each view keeps its own state while you switch.

### Schedule

The **task queue** at the top holds only tasks that are waiting: new ones, and ones that were stopped, failed or unassigned. Once a task is planned or running it lives on the timeline and leaves the queue; unassigning or stopping it with "put back" returns it.

The timeline is split at **now** (red `┃`):

- **Left of now: history, drawn to scale.** These tasks really ran, so their blocks use actual start and end times (grey = finished, brown = stopped, red = failed). Use `h`/`l` to pan back in time and `+`/`-` to zoom; the mouse wheel does the same over this zone.
- **Right of now: the queue, as fixed-width cards.** Each planned assignment gets one column, in queue order. The card shows only the full task name (centred, wrapped if needed), plus the GPU ids for multi-GPU tasks — no times, since they are only estimates (the Task Details pane still lists the estimated start/end). A multi-GPU task takes the same column on each of its GPU rows. The card width is set by `--card-width` (default 24).

Planned work is an **ordered queue**: you choose the GPUs and the position, and the start time is derived. A task starts as soon as all of its GPUs are free, after the running work and the queued work ahead of it. Placements therefore never overlap. Moving, inserting or removing a card re-times everything after it.

| key | action |
|---|---|
| `Tab` | switch focus between the task queue and the timeline |
| `←/→` (queue) | select a card |
| `a` | place the selected task; the ghost starts at the end of the queue |
| `←/→` / `↑/↓` while placing | move the ghost's queue position / anchor GPU (multi-GPU tasks take the following GPUs) |
| `Enter` | confirm the placement, or pick up the card under the cursor to move it |
| `Esc` | cancel the placement |
| `[` / `]` (or `Shift+←/→`) | move the selected card one place earlier / later |
| `u` / `Del` | unassign |
| `t` | reset the view to now |
| `e` | edit/rename the task (queue card, or the card under the timeline cursor). A running task can only be renamed |
| `s` | stop the running task under the cursor. You are asked whether to put it back in the queue or delete it |
| `n` / `d` | new task / delete a waiting task |
| `p` | with a runtime: pause / resume auto-start |
| `r` | with a runtime: start the selected planned task now, ahead of the queue (its GPUs must be free) |

Mouse: drag a task card onto a GPU row. Dropping on the left half of a queued card inserts before it; dropping on the right half inserts after it. Drag a queued card to reorder it or move it to another GPU. A drop on the history zone cancels the drag. A placement that can't be made (wrong GPU count, task already running) shows a red ghost and is refused.

The axis header numbers the queue positions (`#1`, `#2`, …) and shows how many columns are scrolled out of view (`◀2 3▶`).

### Estimates come only from run history

You never type in how long a task takes. A task's estimate is the **median duration of earlier completed runs of the same workload**:

- a task made from a config matches on the config's **resolved parameters** (everything except `_scheduler_`) plus its GPU count. The name doesn't matter, and two differently named configs with identical parameters share history;
- a plain-command task matches on its command plus its GPU count.

With no matching run, the task shows **"no data"** and its end is unknown. Anything queued behind it on the same GPUs then has an unknown start too. Stopped and failed runs are kept in history but are not used for estimates. The Configs view shows the estimate for the selected config's exact parameters in the table title.

### Configs: YAML inheritance

YAML anchors (`<<: *x`) only work inside a single file. Configs here inherit across files through a `_base_` key, mmengine-style:

```yaml
# resnet50_adamw_bs512.yaml   — "subclass"
_base_: resnet50_adamw.yaml    # "parent"; a list means multiple bases, merged left→right
data:
  batch_size: 512              # only the overrides are stored
optimizer:
  lr: 6.0e-4
```

- Mappings merge recursively. Scalars and lists are replaced.
- `_delete_: true` inside a mapping replaces the inherited subtree instead of merging into it.
- `_scheduler_: {gpus, name, run}` sets the GPU count, task name and launch settings (see [Running your code](#running-your-code)) for a task created from the config. There is deliberately no duration field (see above).
- Cycles, missing bases and YAML errors are flagged in the tree. They don't crash the app.

| key | action |
|---|---|
| `Enter` / `e` | edit a value. An inherited key becomes an override in *this* file; the parent is never touched |
| `x` | remove the override so the key inherits again |
| `n` | add a key (`optimizer.betas`, YAML value) |
| `D` | derive a child config. It writes `_base_` plus any `key=value; …` overrides, so later changes to the parent flow through |
| `Ctrl+S` / `R` | save edited files (comments and order are kept) / reload from disk |
| `r` | rename the config file; children's `_base_` and tasks created from it are updated to the new name |
| `T` | create a task from the config and add it to the queue |
| `v` | toggle the bottom pane between the diff vs the parent and the raw YAML |
| `/` | filter keys |

The source column shows where each value comes from: `○` inherited (with the file it comes from), `●` overridden here, `✚` new here.

## Running your code

A task says **where**, **in which environment** and **what** to run. Nothing in your code has to import zgm.

| field | meaning |
|---|---|
| Run | `train.py --lr 3e-4` or `-m pkg.train ...` runs with the environment's Python. Anything else, e.g. `torchrun --nproc_per_node=$ZGM_NUM_GPUS train.py`, is a shell command |
| Environment | path to a Python interpreter (`.venv/bin/python`, relative to the working dir; a venv/conda prefix directory also works), `conda:NAME` (or `conda:/path/to/env`), or empty for `python` on PATH |
| Working dir | where the process starts; relative paths in your code resolve from here |
| Env vars | extra variables, `A=1; B=2` |

You fill these in with `n` (new task) or `e` (edit). `Ctrl+S` saves the form from any field. A config can carry them in `_scheduler_.run`, which is inherited like everything else, so a base config can set the project and environment once:

```yaml
_scheduler_:
  gpus: 2
  run:
    cwd: ../project              # relative to this config file (default: zgm's working dir)
    conda: torch                 # or python: ~/project/.venv/bin/python
    script: train.py             # or module: pkg.train, or command: "torchrun ..."
    args: [--config, "{config}"] # quote placeholders: {..} is a mapping in YAML
    env: {WANDB_MODE: offline}
```

Placeholders in `args`, `command` and `env`, substituted when the task starts:

- `{config}`: a snapshot of the config with `_base_` inheritance resolved and `_scheduler_` removed. Your code reads a plain YAML file, and the snapshot records exactly what the run used. The local runtime keeps it next to the log as `<assignment id>.config.yaml`.
- `{path}`: the config file itself, with `_base_` intact.
- `{name}`: the task name. `{num_gpus}`: its GPU count.

**What the process gets.** `CUDA_VISIBLE_DEVICES` is set to the GPUs the task was planned on, so your code just uses `cuda:0..n-1`. `CUDA_DEVICE_ORDER=PCI_BUS_ID` makes CUDA number GPUs the way `nvidia-smi` and zgm do. CUDA's default is fastest first, which on a machine with mixed GPUs puts another card behind "GPU 2". It also gets `ZGM_NUM_GPUS`, `ZGM_GPU_IDS`, `ZGM_TASK_ID`, `ZGM_TASK_NAME`, and `PYTHONUNBUFFERED=1` so logs appear as they are written. Exit code 0 means success, and only successful runs feed the duration estimates. stdout and stderr go to `<log-dir>/<assignment id>.log`; the Task Details pane shows the path. Stopping a task (`s`) kills its whole process tree, from any zgm on the database.

**Environments.** With a Python path, its directory goes first on PATH, so tools like `torchrun` come from the same environment. With conda, zgm runs `conda run` once per env to capture the activated variables (activation hooks included) and caches them. The process then starts directly with those variables: no wrapper process, and output is live. zgm finds conda through `CONDA_EXE`, PATH, or the usual install folders (`~/anaconda3`, `~/miniconda3`, `~/miniforge3`, `~/mambaforge`).

`examples/project/train.py` is a stand-in training script. The example configs' `run` blocks point at it, so `zgm --config-dir examples/configs --runtime local` runs end to end.

## Submitting from the command line

From your project directory, in the environment you train in:

```bash
zgm submit -g 2 -- train.py --lr 3e-4            # script + args
zgm submit -- -m pkg.train --bs 64               # a module
zgm submit -g 2 --cmd "torchrun --nproc_per_node={num_gpus} train.py"   # any shell command
zgm submit --config configs/rn50_adamw.yaml      # a config, with its _scheduler_ settings
```

The task gets the same fields as in the TUI, filled in from where you are:

- **Working dir**: the current directory (`-C DIR` to change it).
- **Environment**: the active venv (`VIRTUAL_ENV`), else the active conda env (`CONDA_PREFIX`), else `python` on PATH. `--python PATH` or `--conda ENV` set it explicitly. The report says which one was used.
- **What to run**: `script.py ...` and `-m module ...` run with that environment's Python. Anything else becomes a shell command. Put `--` before RUN when it starts with `-` or uses options zgm also has. Use `--cmd "..."` for pipes, `&&` or variables. Write `{num_gpus}` instead of `$ZGM_NUM_GPUS` there, or quote it, so your shell does not expand it at submit time.
- `-e KEY=VALUE` (repeatable) adds environment variables, `-n NAME` names the task, and `-g N` sets the GPU count.

Before queueing, zgm builds the process once, so a missing script, working dir, interpreter or conda env is reported right away (exit code 2) instead of when the task would start. `--no-check` skips this.

**Placement.** By default the task goes to the end of the queue, on the GPUs where it would start first: zgm simulates the queue for every window of adjacent GPUs. A known start beats an unknown one; ties go to the GPUs with less queued work. `--gpu-ids 0,1` picks the GPUs yourself, `--first` puts the task at the front of the queue, and `--hold` only adds it to the waiting queue, for placing in the TUI.

`--config FILE` takes the GPU count, name and `_scheduler_.run` block from the config. If you also give RUN, that runs instead, from the current directory, and can use `{config}` / `{path}`.

A running zgm picks submitted tasks up within a couple of seconds: it reloads when another process writes to the database (SQLite `data_version`). Whoever runs the queue (the daemon, or a TUI with `--runtime local`) then starts them as their GPUs free up. The report's last line says who that is, or that nothing runs the queue yet. `zgm submit` and the TUI must use the same `--db` (the default is the same for both). Without NVML, both default to 4 mock GPUs; `ZGM_MOCK_GPUS` or `--mock-gpus` change that for both.

## Auto-start

Whoever runs the queue (see [In the background](#in-the-background-zgm-daemon)) works through it by itself. Every couple of seconds it starts each planned task that is ready, using the same rule that draws the timeline: none of the task's GPUs is running anything, and no task queued before it is waiting for any of them. So the queue starts in exactly the order you see, a later task never jumps ahead on a GPU that an earlier task is waiting for, and tasks on separate GPUs run side by side.

- The status line shows **AUTO ▶** or **PAUSED ‖**, and who runs the queue, or **PLAN ONLY** if nothing does. `p` toggles auto-start for every zgm on the database; `--no-autostart` pauses it.
- `r` starts the selected task ahead of its place in the queue: now if its GPUs are free, otherwise as soon as they are. This works while auto-start is paused.
- A task that cannot start (missing working dir, script, python or conda env) goes back to the waiting queue as *failed* with the reason, and the queue moves on.
- Nothing starts while you are placing or dragging a card.
- Nothing starts on a GPU that another program is using; see below.

## GPUs used by other programs

Putting a task on a GPU that something else already fills is how training runs out of memory, so zgm checks before it starts anything. It reads each GPU's **compute processes** from NVML: the CUDA contexts and MPS clients, the same list `nvidia-smi` prints. It then sets aside the processes that belong to zgm's own running tasks:

- the runner and the task process of every running task, and all their descendants: torchrun workers, data loader processes, shells;
- any process whose environment carries the `ZGM_TASK_ID` of a running task, which also catches processes whose parent has already exited. Ray workers' children are caught this way too.

Every other compute process makes its GPU **in use outside zgm**:

- Nothing new starts there, not even with `r`. In queue order, tasks queued behind one that waits for that GPU wait as well.
- The timeline shows starts on that GPU as unknown, and the GPU's label shows **! other: N procs, X G**. A notification, or a line in the daemon log, says which process it is, and says so again when the GPU is free.
- `zgm submit` picks other GPUs if it can, and warns if you pin a task to such a GPU with `--gpu-ids`. `zgm daemon status` lists them.

Limits, from what NVML can see:

- **Graphics-only processes are not counted**: the desktop, browsers, games. Every display GPU has some, and on Windows (WDDM) NVML can't say how much memory each one uses. A game holding 10 GB of VRAM is therefore not detected.
- On Windows (WDDM), per-process memory is not available. The label then shows the process count without a size; the GPU is blocked either way.
- If NVML can't list a GPU's processes (old driver, unsupported device), zgm says so once (**? procs unknown** on the label) and does not block that GPU, since it has nothing to go on.
- In a container without the host's PID namespace, NVML reports host PIDs, which zgm can't match to its own tasks. Those tasks' GPUs are already busy in zgm's queue anyway, but the label may count them as *other*.
- A program can still grab a GPU right after zgm checked. Nothing short of exclusive compute mode (`nvidia-smi -c EXCLUSIVE_PROCESS`) can reserve a GPU.

With mock GPUs, `ZGM_MOCK_EXTERNAL="1:9000,2:500:4321"` simulates other programs: 9000 MiB on GPU 1, and pid 4321 using 500 MiB on GPU 2.

## In the background: zgm daemon

```bash
zgm daemon start     # start it in the background (log: daemon.log next to the database)
zgm daemon status    # who runs the queue, what is running and queued
zgm daemon stop      # stop it; running tasks keep running
zgm daemon run       # the same in the foreground (Ctrl+C stops it)
```

**One executor per database.** Exactly one process starts tasks and records how they end: the *executor*. That is `zgm daemon`, or a TUI with `--runtime` when no daemon runs. It holds an OS lock on `<db>.executor.lock`, and describes itself in `<db>.executor.json`. Every other zgm on the database is a client: TUIs show the queue and send `p`, `r` and `s` through the database, and `zgm submit` adds tasks. When the executor exits or dies, the OS drops its lock and the next one takes over: a TUI with `--runtime` takes over within one tick, and a waiting `zgm daemon` takes over as soon as the lock is free.

**Tasks outlive zgm.** The local runtime starts each task through a small detached runner (`zgm.runtime.runner`). On Windows it gets a new process group, no console window, and leaves the parent's job where that is allowed; on POSIX it gets a new session. Closing the TUI, stopping the daemon or closing the terminal leaves the task running. The runner leaves the run's state next to its log:

| file | contents |
|---|---|
| `<id>.log` | the task's output, plus a header and footer from zgm |
| `<id>.run.json` | runner and task pids, GPUs, start time; then exit code and end time |
| `<id>.lock` | held by the runner while it lives, so liveness doesn't rely on pids, which get reused |
| `<id>.cancel` | written by `s`, so a stopped run reads as stopped, not failed |
| `<id>.config.yaml` | the `{config}` snapshot |

The executor that starts next follows every run still marked running from these files. It records the exit code and the **real** end time, even if the run ended while no executor was watching. A run that left no trace, because it was killed outright or the machine rebooted, is marked failed (`lost` in the history), and its GPUs are freed.

The environment, which can hold secrets, reaches the runner through `<id>.spec.json` (mode 0600), which the runner deletes as soon as it has read it. Mock and Ray runs live inside the executor's process, so they can't be followed after it exits; zgm says so, and `s` frees their GPUs.

## Architecture

```
widgets/ + screens/    Textual UI (TaskQueue, TaskCard, TaskDetails, GPUTimeline/GPURow, Config*)
services/              TaskStore · AssignmentService · ConflictValidator · GPU discovery · ConfigRepository
submit.py              zgm submit: CLI -> Task, environment detection, best GPUs
services/launcher.py   LaunchSpec -> process: cwd, interpreter, conda env, CUDA_VISIBLE_DEVICES
services/executor.py   start / follow / stop runs: shared by the daemon and the TUI
services/lease.py      one executor per database (OS file lock)
services/gpu_usage.py  GPUs used by other programs (NVML compute processes minus zgm's)
daemon.py              zgm daemon start|status|stop|run
runtime/               Runtime protocol · LocalRuntime (+ runner.py) · MockRuntime · RayRuntime
storage/sqlite.py      tasks · assignments · history · settings
```

Keyboard and mouse placement both go through `AssignmentService.assign/move`. The UI never imports Ray.

**The local runtime pins exactly**: the planned GPUs are the GPUs used. **Ray** (`--runtime ray`, also for `zgm daemon`) runs each task as a Ray task. Stopping one kills the task's processes too: on POSIX the task is interrupted and kills its own process tree; on Windows, where Ray can't interrupt a running task, the worker is killed and then every process carrying the task's `ZGM_TASK_ID`. Ray runs can't be followed after the executor exits. Ray only counts GPUs it detects, so on a machine without NVIDIA GPUs a task that needs GPUs never starts under Ray. **Ray GPUs are logical resources.** `num_gpus=n` reserves n GPUs, but Ray picks the physical devices. That is why an assignment keeps `preferred_gpu_ids` (your plan) separate from `actual_gpu_ids` (reported by the worker through `ray.get_gpu_ids()`). Strict pinning to physical GPUs would need custom per-device resources.

## Tests

```bash
pytest -q        # models, scheduler, storage, config inheritance, Textual Pilot UI tests,
                 # real processes (runner, daemon, cancel); NVML is faked where no GPU exists
                 # tests/test_ray_adapter.py runs only when ray is installed ([ray] extra)
```
