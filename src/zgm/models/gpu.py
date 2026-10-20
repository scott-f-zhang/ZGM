"""GPUDevice: one physical GPU, on this machine or (cluster mode) on another host."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GPUProcess:
    """A compute process NVML lists on a GPU."""

    pid: int
    name: str = "?"
    used_memory: int | None = None  # bytes; None where NVML can't tell (Windows WDDM)
    host: str | None = None  # set for a process on another host (seen by zgm.probe)
    zgm_task: str | None = None  # its ZGM_TASK_ID, read there (host is set)


@dataclass
class GPUDevice:
    id: int
    uuid: str
    name: str
    memory_total: int  # bytes
    memory_used: int = 0  # bytes
    utilization: int = 0  # percent
    external: list[GPUProcess] = field(default_factory=list)  # used outside zgm
    processes_visible: bool = True  # False: NVML can't list its processes
    # cluster mode: the group whose GPU this is (e.g. a Slurm job), its host, and
    # the GPU's index there (what CUDA_VISIBLE_DEVICES means there); else None
    group: str | None = None
    host: str | None = None
    local_index: int | None = None

    @property
    def memory_fraction(self) -> float:
        return self.memory_used / self.memory_total if self.memory_total else 0.0

    @property
    def short_name(self) -> str:
        return self.name.replace("NVIDIA ", "").replace("GeForce ", "")


def format_bytes_gib(n: int) -> str:
    return f"{n / 1024**3:.0f}G"
