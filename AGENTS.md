# AGENTS.md: using and working on zgm

This file is for AI agents (and scripts). Read it top to bottom before running anything. `README.md` describes the same tool for people, including the interactive TUI. Everything an agent needs is here.

zgm queues jobs on the GPUs of **one machine**. It starts each job on the GPUs it was queued on (`CUDA_VISIBLE_DEVICES`) as soon as those GPUs are free, in queue order. It never starts a job on a GPU that another program is using. Jobs run detached: they keep running when zgm exits.

## Rules

1. **Never run bare `zgm` (or `python -m zgm`) without a subcommand.** That opens a full-screen interactive TUI that waits for keyboard input. Use the subcommands: `zgm submit`, `zgm list`, `zgm logs`, `zgm stop`, `zgm daemon`, `zgm check`.
2. **Use the same `--db` in every command**, or leave it out everywhere. The default is `$ZGM_HOME/state.db`, and `ZGM_HOME` defaults to `~/.local/share/zgm`. Two databases are two separate queues that know nothing about each other.
3. **Pass `--python ABSOLUTE_PATH` or `--conda ENV` on every `zgm submit`.** Without them, zgm uses the environment the command runs in: an activated venv, else an activated conda env, else `python` on PATH, else zgm's own Python. An agent's shell rarely has the right one activated.
4. **Use absolute paths** for `--python`, `-C` and `--config`. A relative `--python` is resolved against the task's working directory (`-C`), not the current directory.
5. **Put `--` between zgm's options and the job's command line**: `zgm submit -g 2 -- train.py --lr 3e-4`.
6. **Check `zgm daemon status` before relying on GPUs.** On a GPU machine, its `gpus:` line must end in `(NVML)`. If it says `mock GPUs`, the NVIDIA driver or NVML is not usable: jobs would get fake GPU ids. Stop and report this; don't submit.
7. **Don't delete or edit** `state.db`, `*.executor.lock`, `*.executor.json` or the files in `logs/` while anything is running. Stop jobs through zgm (see [Stop or remove a task](#stop-or-remove-a-task)), not by killing processes.
8. `zgm daemon stop` stops the scheduler, **not** the jobs. Running jobs continue, and are picked up again by the next `zgm daemon start`.

## Install

Requires Python ≥ 3.10. Real GPUs need an NVIDIA driver; zgm reads the GPUs through NVML (`nvidia-ml-py`, NVIDIA's binding to the driver's `libnvidia-ml`).

```bash
git clone <repo-url> zgm && cd zgm
python3 -m venv .venv
.venv/bin/pip install -e .            # add ".[ray]" for the Ray runtime, ".[dev]" for tests
.venv/bin/zgm daemon status           # check the "gpus:" line (rule 6)
```

On Windows the executables are `.venv\Scripts\zgm.exe` and `.venv\Scripts\python.exe`. The examples below use `zgm` for whichever path applies. zgm's own venv only needs to hold zgm: jobs run in the environment you pass with `--python` or `--conda`.

## Workflow: run a job and get its result

```bash
# 1. Start the scheduler once. Idempotent: prints "already running: ..." if it is.
zgm daemon start
#    -> started: zgm daemon (pid 12345, local runtime)

# 2. Check the machine (rule 6). It also shows running/queued jobs, and GPUs used by other programs.
zgm daemon status

# 3. Queue the job. Exit code 0 = queued; 2 = rejected ({"error": ...} on stdout, reason on stderr).
zgm submit --json --python /home/me/proj/.venv/bin/python -C /home/me/proj -g 2 -n rn50-lr3e-4 \
    -- train.py --lr 3e-4 --out runs/rn50
#    -> {"id": "1a2b3c4d", "name": "rn50-lr3e-4", "status": "planned", "gpus": [0, 1], ...,
#        "executor": "zgm daemon (pid 12345, local runtime)", "warnings": []}

# 4. Wait for it and stream its output: waits while it is queued, prints the log as it is
#    written, and exits when the run ends: 0 if it completed, 1 if it failed or was stopped.
zgm logs -f 1a2b3c4d

# 5. Read the outcome as data.
zgm list --json --status completed,failed
```

Wherever a command takes a task ID, a unique prefix of the id or the task's (unique) name works too.

Check `warnings` in the `submit --json` output: a non-empty list means the job won't start soon. The usual reasons are "nothing runs the queue yet", "auto-start is paused", and "GPU N is in use outside zgm".

### Task records

`zgm list --json` prints a JSON array, oldest task first. `zgm submit --json` prints one record, plus `env_source`, `executor`, `autostart` and `warnings`. `zgm stop --json` prints one record plus `action`.

```json
{"id": "1a2b3c4d", "name": "rn50-lr3e-4", "status": "completed", "last_run": "completed",
 "gpus": [0, 1], "required_gpus": 2, "queue_position": null, "planned_start": null,
 "estimate_seconds": 9360, "started": "2026-10-15T14:05:02", "ended": "2026-10-15T16:41:10",
 "exit_code": 0, "log": "/home/me/.local/share/zgm/logs/9f8e7d6c.log",
 "command": "python train.py --lr 3e-4 --out runs/rn50", "cwd": "/home/me/proj",
 "environment": "/home/me/proj/.venv/bin/python", "config": null, "created": "2026-10-15T14:04:58"}
```

| `status` | meaning | final? |
|---|---|---|
| `planned` | queued on GPUs (`queue_position` 1 = next), waiting for them to be free | no |
| `running` | its process is running | no |
| `completed` | exited with code 0 | yes |
| `failed` | exited non-zero (`exit_code`); or lost (no trace left: killed outright, reboot); or **could not start** (`last_run` is `null`; the reason is in the daemon log) | yes |
| `pending` | in the waiting list, not queued on GPUs: submitted with `--hold`, or stopped (`last_run` = `cancelled`). It will not start on its own | yes, for an agent |
| `cancelled` | its run was stopped, and it was not put back in the waiting list | yes |

`estimate_seconds` is the median of earlier successful runs of the same job, or `null` if there are none. `planned_start` is derived from those estimates and is `null` when unknown.

The log holds the job's stdout and stderr, written as they come, between a zgm header (cwd, command, `CUDA_VISIBLE_DEVICES`) and footer (`# zgm: exited with code N at ...`). `zgm logs ID` prints it at any time, and `zgm logs --path ID` prints where it is.

### Stop or remove a task

```bash
zgm stop 1a2b3c4d             # running: kill its process tree; queued: take it off the queue
zgm stop 1a2b3c4d --remove    # ...and delete the task and its history
```

A stopped task stays listed with `status` `pending` and `last_run` `cancelled`, unless you pass `--remove`. Exit code 1 means there was nothing to stop: the task is neither running nor queued. This works whether or not the daemon runs, and the daemon sees the change on its next check (every 2 s).

## Commands

### `zgm submit [options] -- RUN...`

RUN is one of:

- `script.py ARGS...`: run with the environment's Python;
- `-m package.module ARGS...`: likewise;
- anything else, e.g. `torchrun --nproc_per_node=2 train.py`: run by the shell. For pipes, `&&` or `$VARS`, use `--cmd "STRING"` instead of RUN.

| option | meaning |
|---|---|
| `--python PATH` / `--conda ENV` | the job's environment (rule 3). `PATH` may also be a venv or conda prefix directory. `ENV` is a conda env name or prefix path |
| `-C DIR` | working directory (default: the current directory) |
| `-g N` | number of GPUs (default 1, or the config's `_scheduler_.gpus`) |
| `--gpu-ids 0,1` | exact GPUs. Default: the adjacent GPUs where the job would start first, avoiding GPUs used by other programs |
| `-n NAME` | task name (default: the script/module/program name) |
| `-e KEY=VALUE` | environment variable for the job (repeatable) |
| `--config FILE` | build the task from a zgm YAML config (see README, "Experiment settings") |
| `--first` | put it at the front of the queue |
| `--hold` | only add it to the waiting list; it won't start until placed in the TUI |
| `--json` | print the queued task as JSON (see [Task records](#task-records)); on error `{"error": ...}` |
| `--no-check` | skip the pre-check below |
| `--db PATH`, `--mock-gpus N` | as everywhere |

Before queueing, zgm builds the job's process once. It checks that the working directory, the script, the interpreter and the conda env exist (a conda env is activated once to check; that takes a few seconds). On failure it exits **2**, prints `zgm submit: error: <reason>` on stderr, and queues nothing. Exit **0** means queued.

Placeholders in RUN, `--cmd` and `-e` values are filled in when the job starts: `{num_gpus}`, `{name}`, `{path}` (the config file), and `{config}` (the config with `_base_` inheritance resolved, written to a file). Quote them in the shell. Prefer `{num_gpus}` to `$ZGM_NUM_GPUS` inside `--cmd`, because the shell would expand `$ZGM_NUM_GPUS` at submit time.

### `zgm list [--json] [--status S,S]`

All tasks. `--json` prints the records described above; without it, a table: running tasks first, then the queue in order, then waiting tasks, then finished tasks, newest first. `--status` filters, e.g. `--status running,planned`. If the database doesn't exist yet, it lists nothing and creates nothing.

### `zgm logs ID [-f] [-n N] [--path]`

The log of the task's current or latest run. `-f` waits while the task is queued, follows the output, and exits when the run ends: **0** if it completed, **1** if it failed or was stopped. `-n N` prints only the last N lines (before following). `--path` prints the file's path. Exit 1 also means the task has no log yet (it never started).

### `zgm stop ID [--remove] [--json]`

See [Stop or remove a task](#stop-or-remove-a-task). Exit codes: 0 done, 1 nothing to stop, 2 unknown or ambiguous ID.

### `zgm daemon start | status | stop | run`

| command | prints | exit |
|---|---|---|
| `start [--runtime local\|ray\|mock] [--no-autostart] [--cluster slurm\|none] [--log-dir DIR]` | `started: ...` / `already running: ...` / `started (pid N); it takes over when a zgm TUI ... exits` | 0; 1 if it didn't come up (see `daemon.log`) |
| `status` | queue, executor, `gpus:`, `auto-start:`, running jobs (with log paths), queued count, GPUs used by other programs | 0 |
| `stop` | `stopped (pid N); running tasks keep running` / `not running` | 0; 1 if a TUI (not a daemon) runs the queue, or the daemon didn't exit |
| `run` | the daemon in the foreground, logging to stdout | 0 on Ctrl+C / SIGTERM |

`status` output looks like:

```
queue:      /home/me/.local/share/zgm/state.db
executor:   zgm daemon (pid 12345, local runtime) since 2026-10-15T14:00:00
gpus:       8 x NVIDIA H100 80GB HBM3 (NVML)
auto-start: on
running:    1
  rn50-lr3e-4  GPU 0,1  since 14:05  log /home/me/.local/share/zgm/logs/9f8e7d6c.log
queued:     3
GPU 5 in use outside zgm: python pid 4242, 31.0 GiB
task logs:  /home/me/.local/share/zgm/logs
daemon log: /home/me/.local/share/zgm/daemon.log
```

`auto-start: paused` means nothing new starts on its own. The setting is stored in the database, so it stays paused across daemon restarts: `zgm daemon start` does not resume it. Someone paused it on purpose (the TUI's `p` key, or `--no-autostart`); ask before resuming. To resume without the TUI:

```bash
.venv/bin/python -c "import sys; from zgm.storage.sqlite import SqliteStorage; SqliteStorage(sys.argv[1]).set_setting('autostart', '1')" ~/.local/share/zgm/state.db
```

## Several hosts (cluster mode)

One zgm queue can span several machines. A **backend** says which groups of GPUs exist (today: `slurm`, every running Slurm job of the user); zgm does the rest the same way for every backend. Off by default; it is a setting of the database:

```bash
zgm check --cluster slurm            # 1. must end with "result: ready" (see below)
zgm daemon start --cluster slurm     # 2. stays set for this database; --cluster none turns it off
zgm daemon status                    # 3. the "cluster:" block lists the groups
```

Requirements, all checked by `zgm check` (exit 0 = ready, 1 = something failed; each `FAIL` line says how to fix it):

| requirement | why | `zgm check` line |
|---|---|---|
| the state dir (`ZGM_HOME`) is on a filesystem every host mounts **at the same path** | runs on other hosts are followed through their files there | `shared` |
| every host accepts a login **without a password or prompt** (ssh: a key in `~/.ssh/authorized_keys`; options in `ZGM_SSH_OPTIONS`) | zgm probes hosts and starts runners non-interactively | `reach` |
| zgm runs on every host **with the same Python path** (e.g. a venv in a shared home) | the probe and the runner are `python -m zgm...` | `zgm` |
| the state dir is yours and private; on shared storage, a path with your user name | other users must not read or change your queue and logs | `state dir` |

Do not change ssh keys or permissions on the user's behalf without asking. `zgm check --host HOST` checks any host, with or without a backend.

How it behaves:

- The executor asks the backend for its groups (Slurm: `squeue` every 30 s) and probes each usable group's host every 10 s (`python -m zgm.probe` there). A group's GPUs appear as zgm GPU ids the first time its probe works (`daemon.log`: `node07 (Slurm job 1001): 4 x H100 80GB HBM3 added as GPU 1-4`), and keep those ids while it is usable.
- A task's GPUs are always on one host; `--gpu-ids` takes zgm's ids. A task on another host is started there and followed through the shared state dir. Its `CUDA_VISIBLE_DEVICES` is in that host's numbering, and it gets none of the variables that describe where zgm runs (Slurm: `SLURM_*`).
- When a group goes away, the work queued on its GPUs goes back to the waiting list (`pending`), and runs still there end as `failed`.
- A task that needs more GPUs than any usable host has is rejected (exit 2). `--hold` keeps it in the waiting list, to place later.

Slurm specifics: a group is a running single-node job with GPUs; ssh to its node lands in that job (pam_slurm_adopt, which most clusters use to let users reach their nodes at all), so a task started there lives as long as that job, even after the daemon's own job ends. Not used, with a note: jobs on several nodes, and jobs on a node where the user has another running job.

## What the job's process gets

- `CUDA_VISIBLE_DEVICES`: the GPUs it was queued on, so the job uses `cuda:0` … `cuda:N-1`. `CUDA_DEVICE_ORDER=PCI_BUS_ID`, so zgm's GPU numbers are `nvidia-smi`'s.
- `ZGM_NUM_GPUS`, `ZGM_GPU_IDS`, `ZGM_TASK_ID`, `ZGM_TASK_NAME`, and `PYTHONUNBUFFERED=1`.
- With `--python`, that interpreter's directory first on `PATH` (so `torchrun` etc. come from the same env). With `--conda`, the env's fully activated variables.
- `PYTHONIOENCODING=utf-8`, so Python jobs can print any character (the log is a file, where Python would otherwise use the locale's encoding, e.g. GBK on Chinese Windows).
- stdin is closed. stdout and stderr go to the log file.

Exit code 0 counts as success. Only successful runs feed the duration estimates.

## How the queue decides

- A job starts when **all** its GPUs are free **and** no job queued before it is waiting for any of them. Later jobs never jump ahead on a GPU an earlier job waits for. Jobs on different GPUs run in parallel.
- A GPU with a compute process that zgm didn't start (from NVML, the list `nvidia-smi` prints) counts as **in use outside zgm**: nothing starts there until that process is gone. `zgm daemon status` names the process. Graphics-only processes (desktop, browsers) are not counted.
- Durations are estimated from earlier successful runs of the same command (or config parameters) and GPU count. Until one exists, start times behind a job are "unknown"; that never blocks anything.

## Files and locations

Everything lives next to the database, in the state dir: `$ZGM_HOME`, default `~/.local/share/zgm/`. zgm creates it private (mode 700). On storage several users share, give each user a directory of their own, e.g. `export ZGM_HOME=/shared/projects/$USER/zgm`; `zgm check` warns about a shared path that doesn't name the user.

| path | contents |
|---|---|
| `state.db` | SQLite: tasks, queue, run history, settings. Safe to read; write only through zgm |
| `state.db.executor.lock` / `.executor.json` | who runs the queue (an OS lock, released when that process dies) |
| `daemon.log` | the daemon's log: starts, ends, failures to start, GPUs used by other programs |
| `logs/<run id>.log` | a job's output |
| `logs/<run id>.run.json` | pids, GPUs, start; then `exit_code` and `ended` |
| `logs/<run id>.config.yaml` | the `{config}` snapshot, if used |

## Environment variables zgm reads

| variable | effect |
|---|---|
| `ZGM_HOME` | the state dir: database, logs, daemon log (default `~/.local/share/zgm`) |
| `ZGM_SSH_OPTIONS` | extra `ssh` options for cluster mode, e.g. `-i ~/.ssh/cluster_key -o ProxyJump=gateway` |
| `ZGM_MOCK_GPUS=N` | use N fake GPUs instead of NVML (testing only) |
| `ZGM_MOCK_EXTERNAL="1:9000,2:500:4321"` | with mock GPUs: pretend other programs use GPU 1 (9000 MiB) and GPU 2 (pid 4321, 500 MiB) |
| `CONDA_EXE` | where conda is, if it is not on PATH or in `~/anaconda3`, `~/miniconda3`, `~/miniforge3` or `~/mambaforge` |
| `VIRTUAL_ENV`, `CONDA_PREFIX` | the environment `zgm submit` uses when neither `--python` nor `--conda` is given |

## Troubleshooting

| symptom | cause | fix |
|---|---|---|
| `gpus: N mock GPUs` on a GPU machine | NVML not usable: no driver, or not visible in this container | fix the driver/container (`nvidia-smi` must work); don't submit |
| submit exits 2: `python not found` | relative or wrong `--python` | absolute path to the interpreter (or the env directory) |
| submit exits 2: `conda env 'x' could not be activated` | no such env, or conda not found | `conda env list`; set `CONDA_EXE` |
| last line `nothing runs the queue yet` | no daemon | `zgm daemon start` |
| job stays `planned` | its GPUs are busy (by zgm jobs ahead of it or by other programs), auto-start is paused, or nothing runs the queue | `zgm daemon status`: look at `executor:`, `running:`, `GPU N in use outside zgm`, `auto-start:` |
| `failed` with `last_run` `null` | it could not start (e.g. the working dir was deleted after submit) | the reason is in `daemon.log` |
| `failed`, log ends without `# zgm: exited` | lost: killed outside zgm or machine restarted | resubmit |
| `zgm logs -f` exits 1 | the run failed or was stopped | `zgm list --json`: `exit_code`, `last_run` |
| cluster mode: a group shows `probe failed: HOST: Permission denied` | no non-interactive login to that host | `zgm check --host HOST` says what to fix; ask before changing ssh keys |
| cluster mode: `the runner did not start ... is .../logs shared with HOST?` | the host doesn't see the state dir | `zgm check --host HOST`; put `ZGM_HOME` on shared storage |
| cluster mode: `gpus: no GPUs yet` | the daemon has not probed a group yet (a few seconds after start), or no daemon runs | wait, then `zgm daemon status` again; check `executor:` |
| `zgm daemon stop` exits 1: `... runs the queue, not a daemon; close that TUI instead` | a person's TUI runs the queue | leave it; jobs still run, and submitted jobs still start. Ask before closing someone's TUI |

## Working on zgm's code

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev,ray]"
.venv/bin/python -m pytest -q        # ~225 tests, ~1.5 min; Ray tests skip without [ray]
```

Layout (`src/zgm/`):

| path | role |
|---|---|
| `__main__.py` | CLI entry: TUI, or dispatch to `submit` / `daemon` |
| `submit.py`, `commands.py`, `daemon.py` | the non-interactive commands (`commands.task_record` is the JSON record) |
| `models/` | `Task`, `LaunchSpec` (cwd/env/what to run), `Assignment` (a task's place on GPUs), `GPUDevice` |
| `services/scheduler.py` | the queue: ordered placement, derived start times, `ready_to_start`, `best_gpus` |
| `services/executor.py` | start / follow / stop runs. Shared by the daemon and the TUI. Only the lease holder calls `tick` |
| `services/lease.py`, `filelock.py` | one executor per database |
| `services/launcher.py` | `LaunchSpec` → argv/cwd/env (conda activation, `CUDA_VISIBLE_DEVICES`) |
| `services/gpu_usage.py`, `gpu_discovery.py` | NVML/mock GPUs, and processes not started by zgm |
| `cluster/` | cluster mode, optional: `base.py` (the `Backend` protocol and `Group`), `core.py` (backend-independent: `ClusterWatch`, `ClusterProvider`, `locate`), `transport.py` (local / ssh), `slurm.py` (the Slurm backend), `check.py` (`zgm check`) |
| `probe.py` | `python -m zgm.probe`: a host's GPUs and processes, as JSON |
| `runtime/local.py` + `runner.py` | detached runs and their files; `ray_adapter.py`, `mock.py` |
| `storage/sqlite.py` | schema, with automatic migration when `TABLES` changes (bump `SCHEMA_VERSION`) |
| `app.py`, `screens/`, `widgets/` | the Textual TUI |

Conventions:

- All UI and CLI text is English. Match the surrounding code's comment density and naming.
- Placement goes through `AssignmentService.assign/move` (keyboard, mouse and `zgm submit` alike). The scheduling rule lives in `simulate()` and `ready_to_start()`; keep them consistent.
- Execution logic belongs in `services/executor.py`, not in the TUI. The UI never imports Ray.
- State shared between processes goes in SQLite (other zgm processes notice through `PRAGMA data_version`), or in a run's files. Never only in memory.
- Tests that start real processes must clean them up. Windows matters: venv `python.exe` is a launcher shim that starts the real interpreter as a child, so the pid you spawn is not the pid that runs, and a file can't be replaced while another process reads it.
- No NVIDIA GPU is needed for tests: NVML is faked (`tests/test_gpu_usage.py`), and `--mock-gpus` / `ZGM_MOCK_GPUS` give fake devices. No cluster either: `tests/test_cluster.py` fakes a backend and a transport, and the remote-run tests put a fake `ssh` on PATH.
- Nothing outside `cluster/<backend>.py` knows a backend. A new one (a list of PCs, another scheduler) is a class implementing `cluster.base.Backend`, one entry in `cluster.BACKENDS`, and tests like `tests/test_slurm_backend.py`; a new way to reach hosts is a class in `cluster/transport.py`; a new kind of GPU (not NVML) is a provider in `services/gpu_discovery.py`.
- No site-specific names (hosts, job ids, paths, accounts) in code, docs or tests: use made-up ones like `node07` and job `1001`.
