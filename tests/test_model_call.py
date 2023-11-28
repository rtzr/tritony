import os

import numpy as np
import pytest

from tritony import InferenceClient

TRITON_HOST = os.environ.get("TRITON_HOST", "localhost")
TRITON_HTTP = os.environ.get("TRITON_HTTP", "8000")
TRITON_GRPC = os.environ.get("TRITON_GRPC", "8001")


EPSILON = 1e-8


@pytest.fixture(params=[("http", TRITON_HTTP), ("grpc", TRITON_GRPC)])
def protocol_and_port(request):
    return request.param


def get_client(protocol, port, model_name):
    print(f"Testing {protocol}", flush=True)
    return InferenceClient.create_with(model_name, f"{TRITON_HOST}:{port}", protocol=protocol)


def test_swithcing(protocol_and_port):
    client = get_client(*protocol_and_port, model_name="sample")

    sample = np.random.rand(1, 100).astype(np.float32)
    result = client(sample)
    assert {np.isclose(result, sample).all()}

    sample_batched = np.random.rand(100, 100).astype(np.float32)
    client(sample_batched, model_name="sample_autobatching")
    assert np.isclose(result, sample).all()


def test_with_input_name(protocol_and_port):
    client = get_client(*protocol_and_port, model_name="sample")

    sample = np.random.rand(100, 100).astype(np.float32)
    result = client({client.default_model_spec.model_input[0].name: sample})
    assert np.isclose(result, sample).all()


def test_with_parameters(protocol_and_port):
    client = get_client(*protocol_and_port, model_name="sample")

    sample = np.random.rand(1, 100).astype(np.float32)
    ADD_VALUE = 1
    result = client({client.default_model_spec.model_input[0].name: sample}, parameters={"add": f"{ADD_VALUE}"})

    assert np.isclose(result[0], sample[0] + ADD_VALUE).all()


def test_with_optional(protocol_and_port):
    client = get_client(*protocol_and_port, model_name="sample_optional")

    sample = np.random.rand(1, 100).astype(np.float32)

    result = client({client.default_model_spec.model_input[0].name: sample})
    assert np.isclose(result[0], sample[0], rtol=EPSILON).all()

    OPTIONAL_SUB_VALUE = np.zeros_like(sample) + 3
    result = client(
        {
            client.default_model_spec.model_input[0].name: sample,
            "optional_model_sub": OPTIONAL_SUB_VALUE,
        }
    )
    assert np.isclose(result[0], sample[0] - OPTIONAL_SUB_VALUE, rtol=EPSILON).all()


def test_reload_model_spec(protocol_and_port):
    client = get_client(*protocol_and_port, model_name="sample_autobatching")
    # force to change max_batch_size
    client.default_model_spec.max_batch_size = 4

    sample = np.random.rand(8, 100).astype(np.float32)
    result = client(sample)
    assert np.isclose(result, sample).all()


if __name__ == "__main__":
    test_with_parameters(("grpc", "8101"))
    test_with_optional(("grpc", "8101"))
