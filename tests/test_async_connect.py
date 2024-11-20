import numpy as np
import pytest

from tritony import InferenceClient

from .common_fixtures import MODEL_NAME, TRITON_HOST, async_config

__all__ = ["async_config"]
EPSILON = 1e-8


def get_client(protocol, port, model_name):
    print(f"Testing {protocol}", flush=True)
    return InferenceClient.create_with_asyncio(model_name, f"{TRITON_HOST}:{port}", protocol=protocol)


@pytest.mark.asyncio
async def test_basics(async_config):
    protocol, port = async_config
    print(f"Testing {protocol}:{port}")

    client = get_client(*async_config, model_name=MODEL_NAME)
    sample = np.random.rand(1, 100).astype(np.float32)

    result = await client.aio_infer(sample)
    assert np.isclose(result, sample).all()

    result = await client.aio_infer({"model_in": sample})
    assert np.isclose(result, sample).all()
