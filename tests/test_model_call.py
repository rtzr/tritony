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


def get_client(protocol, port):
    print(f"Testing {protocol}", flush=True)
    return InferenceClient.create_with(MODEL_NAME, f"{TRITON_HOST}:{port}", protocol=protocol)


def test_swithcing(protocol_and_port):
    client = get_client(*protocol_and_port)

    sample = np.random.rand(1, 100).astype(np.float32)
    result = client(sample)
    assert {np.isclose(result, sample).all()}

    sample_batched = np.random.rand(100, 100).astype(np.float32)
    client(sample_batched, model_name="sample_autobatching")
    assert {np.isclose(result, sample).all()}


def test_with_input_name(protocol_and_port):
    client = get_client(*protocol_and_port)

    sample = np.random.rand(100, 100).astype(np.float32)
    result = client({client.default_model_spec.model_input[0].name: sample})
    assert {np.isclose(result, sample).all()}


def test_with_parameters(protocol_and_port):
    client = get_client(*protocol_and_port)

    sample = np.random.rand(1, 100).astype(np.float32)
    ADD_VALUE = 1
    result = client({client.default_model_spec.model_input[0].name: sample}, parameters={"add": f"{ADD_VALUE}"})

    assert {np.isclose(result[0], sample[0] + ADD_VALUE).all()}
