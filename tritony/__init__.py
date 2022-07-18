import os

from .version import __version__

ASYNC_TASKS = int(os.environ.get("ASYNC_TASKS", 6))

from .tools import InferenceClient

__all__ = [InferenceClient, __version__]
