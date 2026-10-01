import os

from .version import __version__

TRITON_ASYNC_TASKS = int(os.environ.get("TRITON_ASYNC_TASKS", os.environ.get("ASYNC_TASKS", 4)))
ASYNC_TASKS = TRITON_ASYNC_TASKS

from .tools import InferenceClient

__all__ = ["InferenceClient", "TRITON_ASYNC_TASKS", "ASYNC_TASKS", "__version__"]
