import os

from .version import __version__
from .tools import InferenceClient

__all__ = [InferenceClient]

ASYNC_TASKS = int(os.environ.get("ASYNC_TASKS", 4))

