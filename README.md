# ZGM

A btop-style terminal UI for planning GPU work on one machine. Task cards sit at the top, details in the middle, and a per-GPU Gantt timeline at the bottom. You place tasks on GPUs and time slots with the keyboard or the mouse. A second view manages experiment YAML configs, which can inherit from each other like classes.

By default zgm only **plans**: an assignment is a planned slot, and moving it never starts or kills a process. With `--runtime local`, zgm also **runs** the queue: each task starts as a child process on its planned GPUs as soon as they are free (see [Running your code](#running-your-code) and [Auto-start](#auto-start)). Execution goes through a `Runtime` interface; local, mock and Ray backends are included.

```bash
pip install -e '.[dev]'            # add [ray] for the Ray runtime
zgm --demo --config-dir examples/configs
zgm --config-dir examples/configs --runtime local   # T on a config, place it, then r
# or: python -m zgm ...
```

Without NVIDIA drivers or NVML, the app falls back to 4 mock GPUs. You can force a count with `--mock-gpus N` or by setting `ZGM_MOCK_GPUS=N`.

| flag | meaning |
|---|---|
| `--db PATH` | SQLite file for tasks, plans and history (default `~/.local/share/zgm/state.db`) |
| `--config-dir DIR` | experiment YAML configs (default `./configs`) |
| `--command-template T` | shell command for tasks made from a config that has no `_scheduler_.run` block; `{path}` and `{name}` are substituted |
| `--demo` | seed example tasks into an empty database |
| `--runtime none\|local\|mock\|ray` | execute planned tasks: `local` runs child processes on this machine; queued tasks auto-start as GPUs free up |
| `--no-autostart` | with a runtime, start paused: tasks only start with `r` until `p` turns auto-start on |
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

**What the process gets.** `CUDA_VISIBLE_DEVICES` is set to the GPUs the task was planned on, so your code just uses `cuda:0..n-1`. It also gets `ZGM_NUM_GPUS`, `ZGM_GPU_IDS`, `ZGM_TASK_ID`, `ZGM_TASK_NAME`, and `PYTHONUNBUFFERED=1` so logs appear as they are written. Exit code 0 means success, and only successful runs feed the duration estimates. stdout and stderr go to `<log-dir>/<assignment id>.log`; the Task Details pane shows the path. Stopping a task (`s`) kills its whole process tree.

**Environments.** With a Python path, its directory goes first on PATH, so tools like `torchrun` come from the same environment. With conda, zgm runs `conda run` once per env to capture the activated variables (activation hooks included) and caches them. The process then starts directly with those variables: no wrapper process, and output is live. zgm finds conda through `CONDA_EXE`, PATH, or the usual install folders (`~/anaconda3`, `~/miniconda3`, `~/miniforge3`, `~/mambaforge`).

`examples/project/train.py` is a stand-in training script. The example configs' `run` blocks point at it, so `zgm --config-dir examples/configs --runtime local` runs end to end.

## Auto-start

With a runtime (`--runtime local|mock|ray`), zgm works through the queue by itself. Every couple of seconds it starts each planned task that is ready, using the same rule that draws the timeline: none of the task's GPUs is running anything, and no task queued before it is waiting for any of them. So the queue starts in exactly the order you see, a later task never jumps ahead on a GPU that an earlier task is waiting for, and tasks on separate GPUs run side by side.

- The status line shows **AUTO ▶** or **PAUSED ‖**. `p` toggles it; `--no-autostart` starts zgm paused.
- `r` starts the selected task now, ahead of its place in the queue. It is refused if one of its GPUs is busy.
- A task that cannot start (missing working dir, script, python or conda env) goes back to the waiting queue as *failed* with the reason, and the queue moves on.
- Nothing starts while you are placing or dragging a card.
- zgm only tracks processes it started in this session. A task still marked running from an earlier session is reported at startup; stop it with `s` to free its GPUs. GPUs used by programs outside zgm are not detected.

## Architecture

```
widgets/ + screens/    Textual UI (TaskQueue, TaskCard, TaskDetails, GPUTimeline/GPURow, Config*)
services/              TaskStore · AssignmentService · ConflictValidator · GPU discovery · ConfigRepository
services/launcher.py   LaunchSpec -> process: cwd, interpreter, conda env, CUDA_VISIBLE_DEVICES
runtime/               Runtime protocol · LocalRuntime · MockRuntime · RayRuntime
storage/sqlite.py      tasks · assignments · history
```

Keyboard and mouse placement both go through `AssignmentService.assign/move`. The UI never imports Ray.

**The local runtime pins exactly**: the planned GPUs are the GPUs used. **Ray GPUs are logical resources.** `num_gpus=n` reserves n GPUs, but Ray picks the physical devices. That is why an assignment keeps `preferred_gpu_ids` (your plan) separate from `actual_gpu_ids` (reported by the worker through `ray.get_gpu_ids()`). Strict pinning to physical GPUs would need custom per-device resources and is planned for phase 2, along with status sync, cancel and retry.

## Tests

```bash
pytest -q        # models, scheduler, storage, config inheritance, Textual Pilot UI tests
                 # tests/test_ray_adapter.py runs only when ray is installed
```
