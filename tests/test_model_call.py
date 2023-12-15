import numpy as np

from tritony import InferenceClient

from .common_fixtures import TRITON_HOST, config

EPSILON = 1e-8
__all__ = ["config"]


def get_client(protocol, port, run_async, model_name):
    print(f"Testing {protocol} with run_async={run_async}", flush=True)
    return InferenceClient.create_with(model_name, f"{TRITON_HOST}:{port}", protocol=protocol, run_async=run_async)


def test_swithcing(config):
    client = get_client(*config, model_name="sample")

    sample = np.random.rand(1, 100).astype(np.float32)
    result = client(sample)
    assert {np.isclose(result, sample).all()}

    sample_batched = np.random.rand(100, 100).astype(np.float32)
    client(sample_batched, model_name="sample_autobatching")
    assert np.isclose(result, sample).all()


def test_with_input_name(config):
    client = get_client(*config, model_name="sample")

    sample = np.random.rand(100, 100).astype(np.float32)
    result = client({client.default_model_spec.model_input[0].name: sample})
    assert np.isclose(result, sample).all()


def test_with_parameters(config):
    client = get_client(*config, model_name="sample")

    sample = np.random.rand(1, 100).astype(np.float32)
    ADD_VALUE = 1
    result = client({client.default_model_spec.model_input[0].name: sample}, parameters={"add": f"{ADD_VALUE}"})

    assert np.isclose(result[0], sample[0] + ADD_VALUE).all()


def test_with_optional(config):
    client = get_client(*config, model_name="sample_optional")

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


def test_reload_model_spec(config):
    client = get_client(*config, model_name="sample_autobatching")
    # force to change max_batch_size
    client.default_model_spec.max_batch_size = 4

    sample = np.random.rand(8, 100).astype(np.float32)
    result = client(sample)
    assert np.isclose(result, sample).all()
