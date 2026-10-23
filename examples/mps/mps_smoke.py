"""A small PyTorch MPS smoke test, run by zgm's example config.

It checks that PyTorch can see the Apple GPU, runs a few matrix multiplications
on it, and prints what it used. A real training script would look much the same:
put the model and the batches on ``mps`` and go.
"""

import argparse
import os
import sys
import time

import torch

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=20)
parser.add_argument("--size", type=int, default=1024)
args = parser.parse_args()

print(f"task     {os.environ.get('ZGM_TASK_NAME')}")
print(f"python   {sys.executable}")
print(f"torch    {torch.__version__}")
print(f"zgm      GPUs={os.environ.get('ZGM_GPU_IDS')}"
      f" CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")

if not torch.backends.mps.is_available():
    raise SystemExit("MPS is not available: use an Apple Silicon Mac and a PyTorch "
                     "build with MPS support")

device = torch.device("mps")
torch.manual_seed(0)
a = torch.randn(args.size, args.size, device=device)
b = torch.randn(args.size, args.size, device=device)
c = a @ b  # warm up
torch.mps.synchronize()

started = time.perf_counter()
for _ in range(args.steps):
    c = a @ b
torch.mps.synchronize()
elapsed = time.perf_counter() - started

print(f"device   {c.device}")
print(f"result   {c[0, 0].item():.6f}")
print(f"matmul   {args.steps} x {args.size}x{args.size} in {elapsed:.2f} s"
      f" ({args.steps / elapsed:.1f} it/s)")
print(f"memory   {torch.mps.current_allocated_memory() / 1024**3:.2f} GiB allocated")
print("MPS OK")
