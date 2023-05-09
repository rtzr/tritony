import os

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


def test_swithcing(protocol_and_port):
    protocol, port = protocol_and_port
    print(f"Testing {protocol}")

    client = InferenceClient.create_with(MODEL_NAME, f"{TRITON_HOST}:{port}", protocol=protocol)

    sample = np.random.rand(1, 100).astype(np.float32)
    result = client(sample)
    print(f"Result: {np.isclose(result, sample).all()}")

    sample_batched = np.random.rand(100, 100).astype(np.float32)
    client(sample_batched, model_name="sample_autobatching")
    print(f"Result: {np.isclose(result, sample).all()}")


def test_with_input_name(protocol_and_port):
    protocol, port = protocol_and_port
    print(f"Testing {protocol}")

    client = InferenceClient.create_with(MODEL_NAME, f"{TRITON_HOST}:{port}", protocol=protocol)

    sample = np.random.rand(1, 100).astype(np.float32)
    result = client({client.input_name_list[0]: sample})
    print(f"Result: {np.isclose(result, sample).all()}")

    sample = np.random.rand(100, 100).astype(np.float32)
    result = client({client.default_model_spec.input_name[0]: sample})

    print(f"Result: {np.isclose(result, sample).all()}")
