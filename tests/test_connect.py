import os

import grpc
import numpy as np
import pytest

from tritony import InferenceClient

MODEL_NAME = os.environ.get("MODEL_NAME", "sample")
TRITON_HOST = os.environ.get("TRITON_HOST", "localhost")
TRITON_HTTP = os.environ.get("TRITON_HTTP", "8100")
TRITON_GRPC = os.environ.get("TRITON_GRPC", "8101")


@pytest.fixture(params=[("http", TRITON_HTTP, True), ("grpc", TRITON_GRPC, True), ("grpc", TRITON_GRPC, False)])
def protocol_and_port(request):
    return request.param


def test_basics(protocol_and_port):
    protocol, port, run_async = protocol_and_port
    print(f"Testing {protocol} with run_async={run_async}")

    client = InferenceClient.create_with(MODEL_NAME, f"{TRITON_HOST}:{port}", protocol=protocol, run_async=run_async)

    sample = np.random.rand(1, 100).astype(np.float32)
    result = client(sample)
    assert np.isclose(result, sample).all()

    result = client({"model_in": sample})
    assert np.isclose(result, sample).all()


def test_batching(protocol_and_port):
    protocol, port, run_async = protocol_and_port
    print(f"{__name__}, Testing {protocol} with run_async={run_async}")

    client = InferenceClient.create_with(
        "sample_autobatching", f"{TRITON_HOST}:{port}", protocol=protocol, run_async=run_async
    )

    sample = np.random.rand(100, 100).astype(np.float32)
    # client automatically makes sub batches with (50, 2, 100)
    result = client(sample)
    assert np.isclose(result, sample).all()


def test_exception(protocol_and_port):
    protocol, port, run_async = protocol_and_port
    print(f"{__name__}, Testing {protocol} with run_async={run_async}")

    client = InferenceClient.create_with(
        "sample_autobatching", f"{TRITON_HOST}:{port}", protocol=protocol, run_async=run_async
    )

    sample = np.random.rand(100, 100, 100).astype(np.float32)
    # client automatically makes sub batches with (50, 2, 100)

    try:
        result = client(sample)
    except RuntimeError as e:
        print(type(e))
    except grpc._channel._InactiveRpcError as e:
        print(f"\n\n\n\ndetails: {e.details()}")
