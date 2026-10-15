# ZGM

A btop-style terminal UI for planning GPU work on one machine. Task cards sit at the top, details in the middle, and a per-GPU Gantt timeline at the bottom. You place tasks on GPUs and time slots with the keyboard or the mouse. A second view manages experiment YAML configs, which can inherit from each other like classes.

Phase 1 is **planning only**: an assignment is a planned slot, and moving it never starts or kills a process. Execution goes through a `Runtime` interface (mock and Ray adapters are included) and is off by default.

```bash
pip install -e '.[dev]'            # add [ray] for the Ray runtime
zgm --demo --config-dir examples/configs
# or: python -m zgm ...
```

Without NVIDIA drivers or NVML, the app falls back to 4 mock GPUs. You can force a count with `--mock-gpus N` or by setting `ZGM_MOCK_GPUS=N`.

| flag | meaning |
|---|---|
| `--db PATH` | SQLite file for tasks, plans and history (default `~/.local/share/zgm/state.db`) |
| `--config-dir DIR` | experiment YAML configs (default `./configs`) |
| `--command-template T` | command for tasks made from a config; `{path}` and `{name}` are substituted |
| `--demo` | seed example tasks into an empty database |
| `--runtime none\|mock\|ray` | enables the `r` key to execute a planned task |

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
- `_scheduler_: {gpus, name}` sets the GPU count and task name for a task created from the config. There is deliberately no duration field (see above).
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

## Architecture

```
widgets/ + screens/    Textual UI (TaskQueue, TaskCard, TaskDetails, GPUTimeline/GPURow, Config*)
services/              TaskStore · AssignmentService · ConflictValidator · GPU discovery · ConfigRepository
runtime/               Runtime protocol · MockRuntime · RayRuntime
storage/sqlite.py      tasks · assignments · history
```

Keyboard and mouse placement both go through `AssignmentService.assign/move`. The UI never imports Ray.

**Ray GPUs are logical resources.** `num_gpus=n` reserves n GPUs, but Ray picks the physical devices. That is why an assignment keeps `preferred_gpu_ids` (your plan) separate from `actual_gpu_ids` (reported by the worker through `ray.get_gpu_ids()`). Strict pinning to physical GPUs would need custom per-device resources and is planned for phase 2, along with status sync, cancel and retry.

## Tests

```bash
pytest -q        # models, scheduler, storage, config inheritance, Textual Pilot UI tests
                 # tests/test_ray_adapter.py runs only when ray is installed
```
