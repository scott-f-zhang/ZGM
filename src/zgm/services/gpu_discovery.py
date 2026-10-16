"""Discover NVIDIA GPUs via NVML, falling back to mock devices."""

from __future__ import annotations

import os
import random

from ..models import GPUDevice, GPUProcess

DEFAULT_MOCK_GPUS = 4


class NVMLProvider:
    def __init__(self) -> None:
        import pynvml  # from nvidia-ml-py

        self._nvml = pynvml
        pynvml.nvmlInit()

    def discover(self) -> list[GPUDevice]:
        nv = self._nvml
        out = []
        for i in range(nv.nvmlDeviceGetCount()):
            h = nv.nvmlDeviceGetHandleByIndex(i)
            name = nv.nvmlDeviceGetName(h)
            uuid = nv.nvmlDeviceGetUUID(h)
            mem = nv.nvmlDeviceGetMemoryInfo(h)
            try:
                util = nv.nvmlDeviceGetUtilizationRates(h).gpu
            except nv.NVMLError:
                util = 0
            out.append(
                GPUDevice(
                    id=i,
                    uuid=uuid.decode() if isinstance(uuid, bytes) else uuid,
                    name=name.decode() if isinstance(name, bytes) else name,
                    memory_total=int(mem.total),
                    memory_used=int(mem.used),
                    utilization=int(util),
                )
            )
        return out

    def processes(self, gpu_id: int) -> list[GPUProcess]:
        """Compute processes on the GPU (CUDA contexts and MPS clients), as
        ``nvidia-smi`` lists them. Raises if NVML can't list them."""
        from .gpu_usage import process_name

        nv = self._nvml
        h = nv.nvmlDeviceGetHandleByIndex(gpu_id)
        found: dict[int, GPUProcess] = {}
        for query in (nv.nvmlDeviceGetComputeRunningProcesses,
                      nv.nvmlDeviceGetMPSComputeRunningProcesses):
            try:
                procs = query(h)
            except nv.NVMLError_NotSupported:
                if query is nv.nvmlDeviceGetComputeRunningProcesses:
                    raise
                continue  # MPS isn't supported everywhere; the compute list is
            for p in procs:
                mem = p.usedGpuMemory
                prev = found.get(p.pid)
                if prev is None or (mem or 0) > (prev.used_memory or 0):
                    found[p.pid] = GPUProcess(p.pid, process_name(p.pid), mem)
        return list(found.values())

    def refresh(self, devices: list[GPUDevice]) -> None:
        nv = self._nvml
        for d in devices:
            try:
                h = nv.nvmlDeviceGetHandleByIndex(d.id)
                d.memory_used = int(nv.nvmlDeviceGetMemoryInfo(h).used)
                d.utilization = int(nv.nvmlDeviceGetUtilizationRates(h).gpu)
            except nv.NVMLError:
                pass


class MockGPUProvider:
    """Fake GPUs so the UI can be developed and tested without hardware."""

    def __init__(self, count: int = DEFAULT_MOCK_GPUS, seed: int = 0) -> None:
        self.count = count
        self._rng = random.Random(seed)
        # processes "on" the fake GPUs; ZGM_MOCK_EXTERNAL="1:8000,2:500:4321" puts a
        # process using 8000 MiB on GPU 1 and one (pid 4321) using 500 MiB on GPU 2
        self.fake_processes: dict[int, list[GPUProcess]] = {}
        for item in filter(None, os.environ.get("ZGM_MOCK_EXTERNAL", "").split(",")):
            gpu, mib, *pid = item.split(":")
            self.fake_processes.setdefault(int(gpu), []).append(GPUProcess(
                int(pid[0]) if pid else 999_000 + int(gpu), "mock-external",
                int(mib) * 1024**2))

    def processes(self, gpu_id: int) -> list[GPUProcess]:
        return list(self.fake_processes.get(gpu_id, []))

    def discover(self) -> list[GPUDevice]:
        return [
            GPUDevice(
                id=i,
                uuid=f"GPU-mock-{i:04d}",
                name="Mock A100-SXM4-80GB",
                memory_total=80 * 1024**3,
                memory_used=self._rng.randint(1, 20) * 1024**3,
                utilization=self._rng.randint(0, 40),
            )
            for i in range(self.count)
        ]

    def refresh(self, devices: list[GPUDevice]) -> None:
        for d in devices:
            d.utilization = max(0, min(100, d.utilization + self._rng.randint(-8, 8)))
            delta = self._rng.randint(-1, 1) * 1024**3
            d.memory_used = max(0, min(d.memory_total, d.memory_used + delta))


def make_provider(mock_count: int | None = None) -> NVMLProvider | MockGPUProvider:
    """NVML if available and not overridden; otherwise mock GPUs.

    ``mock_count`` (or env ZGM_MOCK_GPUS) forces the mock provider.
    """
    env = os.environ.get("ZGM_MOCK_GPUS")
    if mock_count is None and env:
        mock_count = int(env)
    if mock_count is not None:
        return MockGPUProvider(mock_count)
    try:
        provider = NVMLProvider()
        if provider.discover():
            return provider
    except Exception:  # ImportError, NVMLError (no driver / no GPU), ...
        pass
    return MockGPUProvider(DEFAULT_MOCK_GPUS)


def describe_gpus(provider, gpus: list[GPUDevice]) -> str:
    """One line saying what the GPUs are, and whether they are real."""
    if isinstance(provider, MockGPUProvider):
        return (f"{len(gpus)} mock GPUs, not real devices (no NVIDIA driver/NVML found, "
                "or --mock-gpus / ZGM_MOCK_GPUS set)")
    counts: dict[str, int] = {}
    for g in gpus:
        counts[g.name] = counts.get(g.name, 0) + 1
    return ", ".join(f"{n} x {name}" for name, n in counts.items()) + " (NVML)"


def discover_gpus(mock_count: int | None = None) -> list[GPUDevice]:
    return make_provider(mock_count).discover()
