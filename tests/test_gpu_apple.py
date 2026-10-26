"""The Apple Silicon GPU provider: one unified-memory GPU, read from
system_profiler/ioreg, with no per-process visibility."""

from __future__ import annotations

import subprocess

import pytest

from zgm.models import GPUDevice
from zgm.services import gpu_discovery
from zgm.services.gpu_discovery import AppleGPUProvider, describe_gpus, make_provider

DISPLAYS = """
{"SPDisplaysDataType": [
  {"_name": "Apple M5 Max", "spdisplays_mtlgpufamilysupport": "spdisplays_metal4",
   "spdisplays_ndrvs": [{"_name": "DELL U2723QX"}]}
]}
"""

IOREG = '''
+-o AGXAcceleratorG16X  <class IOAccelerator>
  {
    "PerformanceStatistics" = {"Alloc system memory"=6253461504,
      "Device Utilization %"=12,"In use system memory"=1673297920,
      "Renderer Utilization %"=9}
    "model" = "Apple M5 Max"
  }
'''


class Run:
    def __init__(self, displays=DISPLAYS, ioreg=IOREG, memsize="68719476736") -> None:
        self.outputs = {"system_profiler": displays, "ioreg": ioreg, "sysctl": memsize}
        self.calls: list[list[str]] = []

    def __call__(self, argv, **kw):
        self.calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, self.outputs[argv[0]], "")


def test_apple_gpu_is_discovered_with_unified_memory():
    run = Run()
    gpus = AppleGPUProvider(run=run).discover()
    assert [(g.id, g.name) for g in gpus] == [(0, "Apple M5 Max")]
    assert gpus[0].memory_total == 64 * 1024**3
    assert gpus[0].memory_used == 1673297920
    assert gpus[0].utilization == 12
    assert ["system_profiler", "SPDisplaysDataType", "-json"] in run.calls
    assert run.calls[-1][:4] == ["ioreg", "-r", "-d", "1"]


def test_apple_gpu_refresh_re_reads_ioreg():
    run = Run()
    p = AppleGPUProvider(run=run)
    gpus = p.discover()
    run.outputs["ioreg"] = (
        '"PerformanceStatistics" = {"Alloc system memory"=1,'
        '"Device Utilization %"=77,"In use system memory"=2147483648}')
    assert p.sync(gpus) is False
    assert gpus[0].memory_used == 2 * 1024**3
    assert gpus[0].utilization == 77


def test_apple_gpu_without_statistics_is_not_blocked():
    p = AppleGPUProvider(run=Run(ioreg="no statistics here"))
    gpus = p.discover()
    assert (gpus[0].memory_used, gpus[0].utilization) == (0, 0)


def test_apple_gpu_cannot_list_processes():
    with pytest.raises(NotImplementedError):
        AppleGPUProvider(run=Run()).processes(0)


def test_apple_gpu_without_apple_silicon_gpu_is_empty():
    p = AppleGPUProvider(run=Run(displays='{"SPDisplaysDataType": [{"_name": '
                                         '"AMD Radeon Pro 5700"}]}'))
    assert p.discover() == []


def test_describe_says_apple():
    p = AppleGPUProvider(run=Run())
    assert describe_gpus(p, p.discover()) == "1 x Apple M5 Max (Apple GPU)"


def test_make_provider_prefers_apple_on_macos(monkeypatch):
    class FakeApple:
        def discover(self):
            return [GPUDevice(0, "Apple-GPU-0000", "Apple M5 Max", 64 * 1024**3)]

    def boom(*a, **k):
        raise RuntimeError("no NVML")

    monkeypatch.delenv("ZGM_MOCK_GPUS", raising=False)
    monkeypatch.setattr(gpu_discovery, "NVMLProvider", boom)
    monkeypatch.setattr(gpu_discovery, "AppleGPUProvider", FakeApple)
    monkeypatch.setattr(gpu_discovery.sys, "platform", "darwin")
    assert isinstance(make_provider(), FakeApple)


def test_gpu_watch_marks_the_apple_gpu_as_unwatched(store):
    # no per-process view: nothing counts as external, and it is said once
    from zgm.services.gpu_usage import GPUWatch
    provider = AppleGPUProvider(run=Run())
    devices = provider.discover()
    watch = GPUWatch(provider, devices)
    busy, changes = watch.refresh(store)
    assert busy == set() and devices[0].processes_visible is False
    assert devices[0].external == []
    assert len(changes) == 1 and "can't list its processes" in changes[0]
    assert watch.refresh(store) == (set(), [])
