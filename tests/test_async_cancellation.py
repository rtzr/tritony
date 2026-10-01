"""Async inference owns its tasks through cancellation and failure."""

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

from tritony import tools


def pending_tritony_tasks():
    return [task for task in asyncio.all_tasks() if task.get_name().startswith("tritony.") and not task.done()]


def test_cancellation_cleans_queue_waiters():
    async def run():
        task = asyncio.create_task(tools.send_request_async(None, asyncio.Queue(), asyncio.Event(), None, None))
        try:
            while len(pending_tritony_tasks()) < 2:
                await asyncio.sleep(0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert pending_tritony_tasks() == []
        finally:
            leftovers = pending_tritony_tasks()
            for leftover in leftovers:
                leftover.cancel()
            await asyncio.gather(*leftovers, return_exceptions=True)

    asyncio.run(run())


@pytest.mark.parametrize("outcome", ["success", "failure", "cancel"])
@pytest.mark.parametrize("workers", [1, 4])
def test_inference_cleans_owned_tasks(monkeypatch, outcome, workers):
    monkeypatch.setattr(tools, "ASYNC_TASKS", workers)

    async def run():
        started, cleaning, release, finished = (asyncio.Event() for _ in range(4))

        async def request_async(protocol, model_input, client, **kwargs):
            started.set()
            try:
                if outcome == "cancel":
                    await asyncio.Event().wait()
                if outcome == "failure":
                    raise ValueError("RPC failed")
                return [model_input["audio"]]
            finally:
                cleaning.set()
                if outcome == "cancel":
                    await release.wait()
                finished.set()

        monkeypatch.setattr(tools, "request_async", request_async)
        client = SimpleNamespace(
            flag=SimpleNamespace(protocol="grpc", compression_algorithm=None, use_aio_tritonclient=True),
            triton_client=None,
            client_timeout=1,
            build_triton_input=lambda data, *args, **kwargs: {"audio": data[0]},
        )
        model_spec = SimpleNamespace(max_batch_size=1)
        data = np.arange(4).reshape(4, 1)
        task = asyncio.create_task(tools.InferenceClient._call_async_item(client, [data], model_spec))
        try:
            if outcome == "cancel":
                await asyncio.wait_for(started.wait(), timeout=1)
                task.cancel()
                await asyncio.wait_for(cleaning.wait(), timeout=1)
                task.cancel()
                await asyncio.sleep(0)
                assert not task.done() and not finished.is_set()
                release.set()
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(task, timeout=1)
            else:
                result = await asyncio.wait_for(task, timeout=1)
                if outcome == "failure":
                    assert isinstance(result, ValueError) and str(result) == "RPC failed"
                else:
                    np.testing.assert_array_equal(result, data)
            assert finished.is_set()
            assert pending_tritony_tasks() == []
        finally:
            release.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            leftovers = pending_tritony_tasks()
            for leftover in leftovers:
                leftover.cancel()
            await asyncio.gather(*leftovers, return_exceptions=True)

    asyncio.run(run())
