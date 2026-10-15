from __future__ import annotations

import time
from datetime import datetime, timedelta

import pytest

from gpu_scheduler.models import Assignment, Task
from gpu_scheduler.runtime.base import RunState
from gpu_scheduler.runtime.ray_adapter import RayRuntime

ray = pytest.importorskip("ray")
pytestmark = pytest.mark.ray


@pytest.fixture(scope="module")
def runtime():
    rt = RayRuntime(num_cpus=1, include_dashboard=False, log_to_driver=False)
    yield rt
    ray.shutdown()


def wait(rt, handle, timeout=60):
    deadline = time.time() + timeout
    while time.time() < deadline:
        state = rt.status(handle)
        if state not in (RunState.PENDING, RunState.RUNNING):
            return state
        time.sleep(0.2)
    raise TimeoutError


def test_submit_and_status(runtime):
    task = Task("echo", "echo hello-from-ray", 1, timedelta(minutes=1))
    now = datetime.now()
    a = Assignment(task.id, (0,), now, now + task.estimated_duration)
    # num_gpus=0: this container has no GPU; the GPU path is identical apart from the resource
    handle = runtime.submit(task, a, num_gpus=0)
    assert wait(runtime, handle) == RunState.SUCCEEDED
    result = runtime.result(handle)
    assert "hello-from-ray" in result["stdout_tail"]
    assert result["gpu_ids"] == []


def test_failed_command(runtime):
    task = Task("fail", "exit 3", 1, timedelta(minutes=1))
    now = datetime.now()
    a = Assignment(task.id, (0,), now, now + task.estimated_duration)
    handle = runtime.submit(task, a, num_gpus=0)
    assert wait(runtime, handle) == RunState.FAILED
