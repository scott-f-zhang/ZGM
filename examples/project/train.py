"""A stand-in training script for trying zgm's local runtime.

zgm starts it as configured in the example configs' ``_scheduler_.run`` block:
in this directory, with ``--config`` pointing at a snapshot of the config with
``_base_`` inheritance already resolved, and with CUDA_VISIBLE_DEVICES set to
the GPUs the task was planned on. A real script would just use ``cuda:0..n-1``.
"""

import argparse
import os
import time

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
parser.add_argument("--epochs", type=int, default=5)
args = parser.parse_args()

print(f"task     {os.environ.get('ZGM_TASK_NAME')}")
print(f"GPUs     CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}"
      f" ({os.environ.get('ZGM_NUM_GPUS')} visible)")
print(f"config   {args.config}")
with open(args.config, encoding="utf-8") as f:
    print(f.read())
for epoch in range(1, args.epochs + 1):
    time.sleep(1)
    print(f"epoch {epoch}/{args.epochs} done")
