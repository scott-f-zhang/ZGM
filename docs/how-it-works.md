# How zgm works

The details behind the [README](../README.md): how the queue decides, how jobs are started and followed, how GPUs used by other programs are detected, and how the code is laid out. For using zgm from scripts or AI agents, see [AGENTS.md](../AGENTS.md).

## The queue: when a job starts

Whoever runs the queue (see [Running in the background](#running-in-the-background-the-executor-and-detached-runs)) works through it by itself. Every couple of seconds it starts each planned task that is ready, using the same rule that draws the timeline: none of the task's GPUs is running anything, and no task queued before it is waiting for any of them. So the queue starts in exactly the order you see, a later task never jumps ahead on a GPU that an earlier task is waiting for, and tasks on separate GPUs run side by side.

- The status line shows **AUTO ▶** or **PAUSED ‖**, and who runs the queue, or **PLAN ONLY** if nothing does. `p` toggles auto-start for every zgm on the database; `--no-autostart` pauses it.
- `r` starts the selected task ahead of its place in the queue: now if its GPUs are free, otherwise as soon as they are. This works while auto-start is paused.
- A task that cannot start (missing working dir, script, python or conda env) goes back to the waiting queue as *failed* with the reason, and the queue moves on.
- **Auto-fill** (switched in the settings view, `3`) also uses GPUs the queue leaves idle: a GPU with nothing running or queued, and no other program on it, gets the task of the waiting queue's *auto-fill* section with the highest priority that fits there (the oldest among equals) (on adjacent GPUs of one host, like a placement by hand). Tasks whose last run failed or was stopped stay where they are, so a broken task is tried once, not over and over. Nothing is filled while auto-start is paused.
- Nothing starts while you are placing or dragging a card.
- Nothing starts on a GPU that another program is using; see [GPUs used by other programs](#gpus-used-by-other-programs).

## Estimates come only from run history

You never type in how long a task takes. A task's estimate is the **median duration of earlier completed runs of the same workload**:

- a task made from a config matches on the config's **resolved parameters** (everything except `_scheduler_`) plus its GPU count. The name doesn't matter, and two differently named configs with identical parameters share history;
- a plain-command task matches on its command plus its GPU count.

With no matching run, the task shows **"no data"** and its end is unknown. Anything queued behind it on the same GPUs then has an unknown start too. Stopped and failed runs are kept in history but are not used for estimates. The Configs view shows the estimate for the selected config's exact parameters in the table title.

## Launching a job

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

## zgm submit in detail

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

## Running in the background: the executor and detached runs

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

## Several hosts: cluster mode

One zgm queue can span several machines. It is built in layers, so that supporting another kind of setup means adding one module:

| layer | what it does | today | to add |
|---|---|---|---|
| **backend** (`cluster/<name>.py`) | which groups of GPUs exist now, and on which host | `slurm`: your running Slurm jobs | a class implementing `cluster.base.Backend`, one line in `cluster.BACKENDS` |
| **transport** (`cluster/transport.py`) | how a shell command runs on a host, non-interactively | `ssh`, `local` | a class with `host`, `argv(script)` and `run(script, timeout)` |
| **GPU provider** (`services/gpu_discovery.py`) | a machine's GPUs and their processes, read on that machine | NVML, mock | a provider class (e.g. for non-NVIDIA GPUs) |
| **core** (`cluster/core.py`) | probing, zgm GPU ids, placement, groups that go away; the same for every backend | | nothing |

`zgm daemon start --cluster slurm` turns it on for the database (`--cluster none` turns it off). Then:

- **Groups.** The executor asks the backend for its groups (`backend.interval`; Slurm: every 30 s) and keeps them in the `gpu_groups` table. The dashboard shows them in a box under the timeline, titled with the backend's name, and `zgm daemon status` in its `cluster:` block.
- **Probes.** Every 10 seconds the executor runs `python -m zgm.probe` on the host of each usable group, through its transport. The probe reads the GPUs there, and their processes, each with its `ZGM_TASK_ID` if it has one. The group the executor runs in is read in-process. Results go into the table, so TUIs and `zgm submit` never reach other hosts themselves.
- **GPU ids.** The first time a group's probe works, its GPUs get the lowest free zgm ids, which they keep while the group is usable. A multi-GPU task always gets GPUs of one host; placement windows (`a`, `best_gpus`) stay within a host, and the timeline labels each row with its host.
- **Tasks on another host** start there: the transport runs `setsid python -m zgm.runtime.runner --remote ...` on the host. A runner on another host can't be watched through a lock or pids, so it writes a heartbeat into `run.json` every 5 seconds (none for 60 seconds = gone), and stops the task itself when `<id>.cancel` appears. All of this goes through the state dir, which every host must see. The task gets `CUDA_VISIBLE_DEVICES` in that host's numbering, and none of the variables the backend says describe where zgm runs.
- **Groups that go away.** Work queued on their GPUs goes back to the waiting queue; runs still there end as failed once their heartbeat stops; then the ids are free for the next group.

### Before you turn it on: `zgm check`

`zgm check --cluster slurm` (or `zgm check --host HOST`) tests what cluster mode needs, host by host, and says how to fix what fails:

- **reach**: the host accepts a login without a password or a prompt. For ssh that usually means your public key in `~/.ssh/authorized_keys` on the host (with a shared home, the same file), and maybe options in `ZGM_SSH_OPTIONS` (a key file, a jump host).
- **shared**: the host sees the state dir at the same path. zgm writes a file there and reads the host's answer back.
- **zgm**: `python -m zgm.probe` runs on the host with the same Python path, and sees GPUs. Install zgm somewhere every host shares, e.g. a venv in a shared home.
- **state dir**: it is yours and not writable by others. On storage several users share, use a directory named after you: `export ZGM_HOME=/shared/projects/$USER/zgm`.

### The Slurm backend

A group is one of your running, single-node Slurm jobs with GPUs; pending jobs are listed with Slurm's expected start and the reason they wait. Slurm clusters let users ssh to the nodes where they have a job through pam_slurm_adopt, which puts the ssh session into that job: the probe sees exactly the job's GPUs, numbered from 0, and a task started there lives as long as that job, even after the job zgm runs in has ended. Not used, with a note: jobs on several nodes, and jobs on a node where you have another running job (ssh would land in the newest of them, whose GPUs may be other ones).

Limits: logs from another host reach `zgm logs -f` with the shared filesystem's delay (NFS: a few seconds).

## Code layout

```
widgets/ + screens/    Textual UI (TaskQueue, TaskCard, TaskDetails, GPUTimeline/GPURow, Config*)
services/              TaskStore · AssignmentService · ConflictValidator · GPU discovery · ConfigRepository
submit.py              zgm submit: CLI -> Task, environment detection, best GPUs
services/launcher.py   LaunchSpec -> process: cwd, interpreter, conda env, CUDA_VISIBLE_DEVICES
services/executor.py   start / follow / stop runs: shared by the daemon and the TUI
services/lease.py      one executor per database (OS file lock)
services/gpu_usage.py  GPUs used by other programs (NVML compute processes minus zgm's)
cluster/               cluster mode: base (Backend, Group) · core (ClusterWatch, ClusterProvider)
                       · transport (local, ssh) · slurm (a backend) · check (zgm check)
probe.py               python -m zgm.probe: a host's GPUs and processes, as JSON
daemon.py              zgm daemon start|status|stop|run
runtime/               Runtime protocol · LocalRuntime (+ runner.py) · MockRuntime · RayRuntime
storage/sqlite.py      tasks · assignments · history · settings
```

Keyboard and mouse placement both go through `AssignmentService.assign/move`. The UI never imports Ray.

**The local runtime pins exactly**: the planned GPUs are the GPUs used. **Ray** (`--runtime ray`, also for `zgm daemon`) runs each task as a Ray task. Stopping one kills the task's processes too: on POSIX the task is interrupted and kills its own process tree; on Windows, where Ray can't interrupt a running task, the worker is killed and then every process carrying the task's `ZGM_TASK_ID`. Ray runs can't be followed after the executor exits. Ray only counts GPUs it detects, so on a machine without NVIDIA GPUs a task that needs GPUs never starts under Ray. **Ray GPUs are logical resources.** `num_gpus=n` reserves n GPUs, but Ray picks the physical devices. That is why an assignment keeps `preferred_gpu_ids` (your plan) separate from `actual_gpu_ids` (reported by the worker through `ray.get_gpu_ids()`). Strict pinning to physical GPUs would need custom per-device resources.
