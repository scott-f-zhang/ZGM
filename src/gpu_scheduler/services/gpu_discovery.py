"""Discover NVIDIA GPUs via NVML, falling back to mock devices."""

from __future__ import annotations

import os
import random

from ..models import GPUDevice

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

    ``mock_count`` (or env GPU_SCHEDULER_MOCK_GPUS) forces the mock provider.
    """
    env = os.environ.get("GPU_SCHEDULER_MOCK_GPUS")
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


def discover_gpus(mock_count: int | None = None) -> list[GPUDevice]:
    return make_provider(mock_count).discover()
