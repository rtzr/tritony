import os

import grpc
import numpy as np
import pytest

from tritony import InferenceClient

MODEL_NAME = os.environ.get("MODEL_NAME", "sample")
TRITON_HOST = os.environ.get("TRITON_HOST", "localhost")
TRITON_HTTP = os.environ.get("TRITON_HTTP", "8000")
TRITON_GRPC = os.environ.get("TRITON_GRPC", "8001")


@pytest.fixture(params=[("http", TRITON_HTTP), ("grpc", TRITON_GRPC)])
def protocol_and_port(request):
    return request.param


def test_basics(protocol_and_port):
    protocol, port = protocol_and_port
    print(f"Testing {protocol}")

    client = InferenceClient.create_with(MODEL_NAME, f"{TRITON_HOST}:{port}", protocol=protocol)

    sample = np.random.rand(1, 100).astype(np.float32)
    result = client(sample)
    print(f"Result: {np.isclose(result, sample).all()}")

    result = client({"model_in": sample})
    print(f"Dict Result: {np.isclose(result, sample).all()}")


def test_batching(protocol_and_port):
    protocol, port = protocol_and_port
    print(f"{__name__}, Testing {protocol}")

    client = InferenceClient.create_with("sample_autobatching", f"{TRITON_HOST}:{port}", protocol=protocol)

    sample = np.random.rand(100, 100).astype(np.float32)
    # client automatically makes sub batches with (50, 2, 100)
    result = client(sample)
    print(f"Result: {np.isclose(result, sample).all()}")


def test_exception(protocol_and_port):
    protocol, port = protocol_and_port
    print(f"{__name__}, Testing {protocol}")

    client = InferenceClient.create_with("sample_autobatching", f"{TRITON_HOST}:{port}", protocol=protocol)

    sample = np.random.rand(100, 100, 100).astype(np.float32)
    # client automatically makes sub batches with (50, 2, 100)

    try:
        result = client(sample)
    except RuntimeError as e:
        print(type(e))
    except grpc._channel._InactiveRpcError as e:
        print(f"\n\n\n\ndetails: {e.details()}")
