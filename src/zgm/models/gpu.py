"""GPUDevice: one physical GPU on this machine."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GPUDevice:
    id: int
    uuid: str
    name: str
    memory_total: int  # bytes
    memory_used: int = 0  # bytes
    utilization: int = 0  # percent

    @property
    def memory_fraction(self) -> float:
        return self.memory_used / self.memory_total if self.memory_total else 0.0

    @property
    def short_name(self) -> str:
        return self.name.replace("NVIDIA ", "").replace("GeForce ", "")


def format_bytes_gib(n: int) -> str:
    return f"{n / 1024**3:.0f}G"
