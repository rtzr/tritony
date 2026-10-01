"""Async task count selection is backward compatible at import time."""

import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "legacy,preferred,expected",
    [(None, None, 4), ("2", None, 2), (None, "3", 3), ("2", "3", 3), ("invalid", "3", 3)],
)
def test_async_task_environment(legacy, preferred, expected):
    env = os.environ.copy()
    for name, value in [("ASYNC_TASKS", legacy), ("TRITON_ASYNC_TASKS", preferred)]:
        env.pop(name, None)
        if value is not None:
            env[name] = value
    result = subprocess.run(
        [sys.executable, "-c", "from tritony import ASYNC_TASKS; print(ASYNC_TASKS)"],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert int(result.stdout) == expected
