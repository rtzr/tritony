import asyncio
import logging

import numpy as np
import pytest

from tritony import InferenceClient

from .common_fixtures import MODEL_NAME, TRITON_HOST, async_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)

__all__ = ["async_config"]
EPSILON = 1e-8


def get_client(protocol, port, model_name):
    print(f"Testing {protocol}", flush=True)
    return InferenceClient.create_with_asyncio(model_name, f"{TRITON_HOST}:{port}", protocol=protocol)


@pytest.mark.asyncio
async def test_basics(async_config):
    protocol, port = async_config

    client = get_client(*async_config, model_name=MODEL_NAME)
    sample = np.random.rand(1, 100).astype(np.float32)

    result = await client.aio_infer(sample)
    assert np.isclose(result, sample).all()

    result = await client.aio_infer({"model_in": sample})
    assert np.isclose(result, sample).all()


@pytest.mark.asyncio
async def test_multiple_tasks(async_config):
    n_multiple_tasks = 10
    protocol, port = async_config
    print(f"Testing {protocol}:{port}")

    client_list = [get_client(*async_config, model_name="sample_sleep_1sec") for _ in range(n_multiple_tasks)]

    sample = np.random.rand(1, 100).astype(np.float32)
    tasks = [client.aio_infer(sample) for client in client_list]

    start_time = asyncio.get_event_loop().time()
    results = await asyncio.gather(*tasks)
    end_time = asyncio.get_event_loop().time()

    for result in results:
        assert np.isclose(result, sample).all()

    assert (end_time - start_time) < 2, f"Time taken: {end_time - start_time}"
