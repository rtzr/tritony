import os

import numpy as np
import pytest

from tritony import InferenceClient

MODEL_NAME = os.environ.get("MODEL_NAME", "sample")
TRITON_HOST = os.environ.get("TRITON_HOST", "localhost")
TRITON_HTTP = os.environ.get("TRITON_HTTP", "8000")
TRITON_GRPC = os.environ.get("TRITON_GRPC", "8001")


@pytest.mark.parametrize("protocol_and_port", [("http", TRITON_HTTP), ("grpc", TRITON_GRPC)])
def test_basics(protocol_and_port):
    protocol, port = protocol_and_port
    print(f"Testing {protocol}")

    client = InferenceClient.create_with(MODEL_NAME, f"{TRITON_HOST}:{port}", protocol=protocol)

    sample = np.random.rand(1, 100).astype(np.float32)
    result = client(sample)
    print(f"Result: {np.isclose(result, sample).all()}")
