import os

import pytest

MODEL_NAME = os.environ.get("MODEL_NAME", "sample")
TRITON_HOST = os.environ.get("TRITON_HOST", "localhost")
TRITON_HTTP = os.environ.get("TRITON_HTTP", "8100")
TRITON_GRPC = os.environ.get("TRITON_GRPC", "8101")


@pytest.fixture(params=[("http", TRITON_HTTP, True), ("grpc", TRITON_GRPC, True), ("grpc", TRITON_GRPC, False)])
def config(request):
    """
    Returns a tuple of (protocol, port, run_async)
    """
    return request.param
