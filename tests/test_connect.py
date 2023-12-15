import grpc
import numpy as np

from tritony import InferenceClient

from .common_fixtures import MODEL_NAME, TRITON_HOST, config

__all__ = ["config"]


def test_basics(config):
    protocol, port, run_async = config
    print(f"Testing {protocol} with run_async={run_async}")

    client = InferenceClient.create_with(MODEL_NAME, f"{TRITON_HOST}:{port}", protocol=protocol, run_async=run_async)

    sample = np.random.rand(1, 100).astype(np.float32)
    result = client(sample)
    assert np.isclose(result, sample).all()

    result = client({"model_in": sample})
    assert np.isclose(result, sample).all()


def test_batching(config):
    protocol, port, run_async = config
    print(f"{__name__}, Testing {protocol} with run_async={run_async}")

    client = InferenceClient.create_with(
        "sample_autobatching", f"{TRITON_HOST}:{port}", protocol=protocol, run_async=run_async
    )

    sample = np.random.rand(100, 100).astype(np.float32)
    # client automatically makes sub batches with (50, 2, 100)
    result = client(sample)
    assert np.isclose(result, sample).all()


def test_exception(config):
    protocol, port, run_async = config
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
