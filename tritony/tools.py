from __future__ import annotations

import asyncio
import functools
import itertools
import logging
import os
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import grpc
import numpy as np
from reretry import retry
from tritonclient import grpc as grpcclient
from tritonclient import http as httpclient
from tritonclient.grpc import InferResult
from tritonclient.grpc import aio as aio_grpcclient
from tritonclient.grpc._utils import _get_inference_request as grpc_get_inference_request
from tritonclient.http import aio as aio_httpclient
from tritonclient.utils import InferenceServerException

from tritony import ASYNC_TASKS
from tritony.helpers import (
    COMPRESSION_ALGORITHM_MAP,
    TritonClientFlag,
    TritonModelSpec,
    TritonProtocol,
    async_get_triton_client,
    get_triton_client,
    init_triton_client,
)

logger = logging.getLogger(__name__)

TRITON_LOAD_DELAY = float(os.environ.get("TRITON_LOAD_DELAY", 2))
TRITON_BACKOFF_COEFF = float(os.environ.get("TRITON_BACKOFF_COEFF", 2))
TRITON_RETRIES = int(os.environ.get("TRITON_RETRIES", 5))
TRITON_CLIENT_TIMEOUT = int(os.environ.get("TRITON_CLIENT_TIMEOUT", 30))

_executor = ThreadPoolExecutor(max_workers=ASYNC_TASKS)


async def data_generator(data: list[np.ndarray], batch_size: int, queue: asyncio.Queue, stop: asyncio.Event):
    """
    batch data generator

    :param data:
    :param batch_size:
    :param queue:
    :param stop:
    :return:
    """
    if batch_size == 0:
        batch_iterable = zip(*data)
    else:
        split_indices = np.arange(batch_size, data[0].shape[0], batch_size)
        batch_iterable = zip(*[np.array_split(elem, split_indices) for elem in data])

    for idx, inputs_list in enumerate(batch_iterable):
        await queue.put((idx, inputs_list))
    stop.set()


async def send_request_async(
    inference_client: InferenceClient,
    data_queue,
    done_event,
    triton_client: grpcclient.InferenceServerClient | httpclient.InferenceServerClient,
    model_spec: TritonModelSpec,
    parameters: dict | None = None,
):
    ret = []
    while True:
        data = asyncio.create_task(data_queue.get(), name="tritony.data_queue.get")
        done = asyncio.create_task(done_event.wait(), name="tritony.done_event.wait")
        d, pending = await asyncio.wait({data, done}, return_when=asyncio.FIRST_COMPLETED)

        # Cancel pending task
        for p in pending:
            p.cancel()

        idx, batch_data = None, None
        if data in d:
            try:
                idx, batch_data = data.result()
            except asyncio.CancelledError:
                continue
        elif done in d:
            return ret

        try:
            a_pred = await request_async(
                inference_client.flag.protocol,
                inference_client.build_triton_input(batch_data, model_spec, parameters=parameters),
                triton_client,
                timeout=inference_client.client_timeout,
                compression=inference_client.flag.compression_algorithm,
                use_aio_tritonclient=inference_client.flag.use_aio_tritonclient,
            )
            ret.append((idx, a_pred))
        except InferenceServerException as triton_error:
            handle_triton_error(triton_error)
        except grpc.RpcError as grpc_error:
            handle_grpc_error(grpc_error)
        finally:
            data_queue.task_done()


def handle_triton_error(triton_error: InferenceServerException):
    """
    https://github.com/triton-inference-server/core/blob/0141e0651c4355bf8a9d1118aac45abda6569997/src/scheduler_utils.cc#L133
    ("Max_queue_size exceeded", "Server not ready", "failed to connect to all addresses")
    """
    if triton_error.status() == "400" and "batch-size must be <=" in triton_error.message():
        raise triton_error
    runtime_msg = f"{triton_error.status()} with {triton_error.message()}"
    raise RuntimeError(runtime_msg) from triton_error


def handle_grpc_error(grpc_error: grpc.RpcError):
    if grpc_error.code() == grpc.StatusCode.INVALID_ARGUMENT:
        raise grpc_error
    else:
        runtime_msg = f"{grpc_error.code()} with {grpc_error.details()}"
        raise RuntimeError(grpc_error.details()) from grpc_error


def request(
    protocol: TritonProtocol,
    model_input: dict,
    triton_client: grpcclient.InferenceServerClient | httpclient.InferenceServerClient,
    timeout: int,
    compression: str,
):
    st = time.time()

    if protocol == TritonProtocol.grpc:
        if "parameters" in grpc_get_inference_request.__code__.co_varnames:
            # check tritonclient[all]>=2.34.0, NGC 23.04
            model_input["parameters"] = model_input.get("parameters", None)
        elif "parameters" in model_input:
            warnings.warn(UserWarning("tritonclient[all]<2.34.0, NGC 23.04"))
            model_input.pop("parameters")
        request = grpc_get_inference_request(
            **model_input,
            priority=0,
            timeout=timeout,
            sequence_id=0,
            sequence_start=False,
            sequence_end=False,
        )
        ModelInferResponse = triton_client._client_stub.ModelInfer(
            request=request, timeout=timeout, compression=COMPRESSION_ALGORITHM_MAP[compression]
        )
        result = InferResult(ModelInferResponse)
    else:
        # TODO: implement http client more efficiently
        raise NotImplementedError("Not implemented for httpclient yet")

    ed = time.time()
    logger.debug(f"{model_input['model_name']} {model_input['inputs'][0].shape()} elapsed: {ed - st:.3f}")

    return [result.as_numpy(outputs.name()) for outputs in model_input["outputs"]]


@retry(
    (InferenceServerException, grpc.RpcError),
    tries=TRITON_RETRIES,
    delay=TRITON_LOAD_DELAY,
    backoff=TRITON_BACKOFF_COEFF,
)
async def request_async(
    protocol: TritonProtocol,
    model_input: dict,
    triton_client,
    timeout: int,
    compression: str,
    use_aio_tritonclient: bool,
):
    st = time.time()

    if protocol == TritonProtocol.grpc and not use_aio_tritonclient:
        loop = asyncio.get_running_loop()

        if "parameters" in grpc_get_inference_request.__code__.co_varnames:
            # check tritonclient[all]>=2.34.0, NGC 23.04
            model_input["parameters"] = model_input.get("parameters", None)
        elif "parameters" in model_input:
            warnings.warn(UserWarning("tritonclient[all]<2.34.0, NGC 23.04"))
            model_input.pop("parameters")
        request = grpc_get_inference_request(
            **model_input,
            priority=0,
            timeout=timeout,
            sequence_id=0,
            sequence_start=False,
            sequence_end=False,
        )
        func_partial = functools.partial(
            triton_client._client_stub.ModelInfer,
            request=request,
            timeout=timeout,
            compression=COMPRESSION_ALGORITHM_MAP[compression],
        )
        ModelInferResponse = await loop.run_in_executor(_executor, func_partial)
        result = InferResult(ModelInferResponse)
    elif protocol == TritonProtocol.http and not use_aio_tritonclient:
        # TODO: implement http client more efficiently
        result = triton_client.async_infer(**model_input, timeout=timeout).get_result()
    elif protocol == TritonProtocol.grpc and use_aio_tritonclient:
        result = await triton_client.infer(**model_input)
    elif protocol == TritonProtocol.http and use_aio_tritonclient:
        result = await triton_client.infer(**model_input)
    else:
        raise RuntimeError("Not Reachable")

    ed = time.time()
    logger.debug(f"{model_input['model_name']} {model_input['inputs'][0].shape()} elapsed: {ed - st:.3f}")

    return [result.as_numpy(outputs.name()) for outputs in model_input["outputs"]]


class InferenceClient:
    @classmethod
    def create_with(
        cls,
        model: str,
        url: str,
        input_dims: int = 2,
        model_version: str = "1",
        protocol: str = "grpc",
        run_async: bool = True,
        concurrency: int = ASYNC_TASKS,
        secure: bool = False,
        compression_algorithm: str | None = None,
    ):
        return cls(
            TritonClientFlag(
                url=url,
                model_name=model,
                model_version=model_version,
                protocol=protocol,
                async_set=run_async,
                concurrency=concurrency,
                input_dims=input_dims,
                compression_algorithm=compression_algorithm,
                ssl=secure,
            )
        )

    @classmethod
    def create_with_asyncio(
        cls,
        model: str,
        url: str,
        input_dims: int = 2,
        model_version: str = "1",
        protocol: str = "grpc",
        run_async: bool = True,
        concurrency: int = ASYNC_TASKS,
        secure: bool = False,
        compression_algorithm: str | None = None,
    ):
        return cls(
            TritonClientFlag(
                url=url,
                model_name=model,
                model_version=model_version,
                protocol=protocol,
                async_set=run_async,
                concurrency=concurrency,
                input_dims=input_dims,
                compression_algorithm=compression_algorithm,
                ssl=secure,
                use_aio_tritonclient=True,
            )
        )

    def __init__(self, flag: TritonClientFlag):
        self.__version__ = 1

        self.flag = flag
        self.default_model = (flag.model_name, flag.model_version)
        self.model_specs: dict[tuple[str, str], TritonModelSpec] = {}
        self.is_async = self.flag.async_set
        self.client_timeout = TRITON_CLIENT_TIMEOUT
        self._triton_client = None
        self.init_triton_client()

        self.sent_count = 0
        self.processed_count = 0

    def init_triton_client(self):
        if not self.flag.use_aio_tritonclient:
            self._triton_client: grpcclient.InferenceServerClient | httpclient.InferenceServerClient = (
                init_triton_client(self.flag)
            )
        else:
            self._triton_client: aio_grpcclient.InferenceServerClient | aio_httpclient.InferenceServerClient = (
                init_triton_client(self.flag)
            )

    @property
    def triton_client(self):
        return self._triton_client

    @property
    def default_model_spec(self):
        if len(self.model_specs.keys()) == 0:
            self._renew_triton_client(self._triton_client)
        return self.model_specs[self.default_model]

    @property
    async def async_default_model_spec(self):
        if len(self.model_specs.keys()) == 0:
            await self._async_renew_triton_client(self._triton_client)
        return self.model_specs[self.default_model]

    async def async_close(self):
        await self._triton_client.close()

    def __del__(self):
        try:
            if asyncio.iscoroutinefunction(self._triton_client.close):
                if not asyncio.get_event_loop().is_closed():
                    asyncio.create_task(self._triton_client.close())
            else:
                self.close()
        except Exception:
            pass

    @retry((InferenceServerException, grpc.RpcError), tries=TRITON_RETRIES, delay=TRITON_LOAD_DELAY, backoff=2)
    def _renew_triton_client(
        self,
        triton_client: grpcclient.InferenceServerClient | httpclient.InferenceServerClient,
        model_name: str | None = None,
        model_version: str | None = None,
    ):
        if not model_name:
            model_name = self.flag.model_name
        if not model_version:
            model_version = self.flag.model_version

        triton_client.is_server_live()
        triton_client.is_server_ready()
        triton_client.is_model_ready(model_name, model_version)

        (max_batch_size, input_list, output_name_list) = get_triton_client(
            triton_client, model_name=model_name, model_version=model_version, protocol=self.flag.protocol
        )

        self.model_specs[(model_name, model_version)] = TritonModelSpec(
            name=model_name,
            max_batch_size=max_batch_size,
            model_input=input_list,
            output_name=output_name_list,
        )

    @retry((InferenceServerException, grpc.RpcError), tries=TRITON_RETRIES, delay=TRITON_LOAD_DELAY, backoff=2)
    async def _async_renew_triton_client(
        self,
        triton_client: aio_grpcclient.InferenceServerClient | aio_httpclient.InferenceServerClient,
        model_name: str | None = None,
        model_version: str | None = None,
    ):
        if not model_name:
            model_name = self.flag.model_name
        if not model_version:
            model_version = self.flag.model_version

        await triton_client.is_server_live()
        await triton_client.is_server_ready()
        await triton_client.is_model_ready(model_name, model_version)

        (max_batch_size, input_list, output_name_list) = await async_get_triton_client(
            triton_client, model_name=model_name, model_version=model_version, protocol=self.flag.protocol
        )

        self.model_specs[(model_name, model_version)] = TritonModelSpec(
            name=model_name,
            max_batch_size=max_batch_size,
            model_input=input_list,
            output_name=output_name_list,
        )

    def _get_request_id(self):
        self.sent_count += 1
        return self.sent_count

    def get_model_spec(self, model_name: str | None, model_version: str | None) -> TritonModelSpec:
        if model_name is None:
            model_name = self.flag.model_name
        if model_version is None:
            model_version = self.flag.model_version
        if (model_name, model_version) not in self.model_specs:
            self._renew_triton_client(self.triton_client, model_name, model_version)

        return self.model_specs[(model_name, model_version)]

    async def async_get_model_spec(self, model_name: str | None, model_version: str | None) -> TritonModelSpec:
        if model_name is None:
            model_name = self.flag.model_name
        if model_version is None:
            model_version = self.flag.model_version
        if (model_name, model_version) not in self.model_specs:
            await self._async_renew_triton_client(self.triton_client, model_name, model_version)

        return self.model_specs[(model_name, model_version)]

    def __call__(
        self,
        sequences_or_dict: list[np.ndarray] | dict[str, list[Any]] | np.ndarray,
        parameters: dict | None = None,
        model_name: str | None = None,
        model_version: str | None = None,
    ):
        model_spec = self.get_model_spec(model_name, model_version)

        if type(sequences_or_dict) in [list, np.ndarray]:
            sequences_list = [sequences_or_dict]
        elif type(sequences_or_dict) is dict:
            sequences_list = [
                sequences_or_dict[model_input.name]
                for model_input in model_spec.model_input
                if model_input.optional is False  # check required
                or (model_input.optional is True and model_input.name in sequences_or_dict)  # check optional
            ]

        if self.is_async:
            return self._call_async(sequences_list, model_spec=model_spec, parameters=parameters)
        else:
            return self._call_request(sequences_list, model_spec=model_spec, parameters=parameters)

    def build_triton_input(
        self,
        _input_list: list[np.ndarray],
        model_spec: TritonModelSpec,
        parameters: dict | None = None,
    ):
        if self.flag.protocol is TritonProtocol.grpc:
            client = grpcclient
        else:
            client = httpclient
        infer_input_list = []
        for _input, _model_input in zip(_input_list, model_spec.model_input):
            infer_input = client.InferInput(_model_input.name, _input.shape, _model_input.dtype)
            infer_input.set_data_from_numpy(_input)
            infer_input_list.append(infer_input)

        infer_requested_output = [client.InferRequestedOutput(output_name) for output_name in model_spec.output_name]

        request_id = self._get_request_id()
        request_input = dict(
            model_name=model_spec.name,
            inputs=infer_input_list,
            request_id=str(request_id),
            model_version=model_spec.model_version,
            outputs=infer_requested_output,
            parameters=parameters,
        )

        return request_input

    async def aio_infer(
        self,
        sequences_or_dict: list[np.ndarray] | dict[str, list[Any]] | np.ndarray,
        parameters: dict | None = None,
        model_name: str | None = None,
        model_version: str | None = None,
    ):
        model_spec = await self.async_get_model_spec(model_name, model_version)

        if type(sequences_or_dict) in [list, np.ndarray]:
            sequences_list = [sequences_or_dict]
        elif type(sequences_or_dict) is dict:
            sequences_list = [
                sequences_or_dict[model_input.name]
                for model_input in model_spec.model_input
                if model_input.optional is False  # check required
                or (model_input.optional is True and model_input.name in sequences_or_dict)  # check optional
            ]

        return await self._aio_infer(sequences_list, model_spec=model_spec, parameters=parameters)

    async def _aio_infer(
        self,
        data: list[np.ndarray],
        model_spec: TritonModelSpec | None = None,
        parameters: dict | None = None,
    ) -> np.ndarray | None:
        for retry_idx in range(max(2, TRITON_RETRIES)):
            async_result = await self._call_async_item(data=data, model_spec=model_spec, parameters=parameters)

            is_invalid_argument_grpc = (
                self.flag.protocol is TritonProtocol.grpc
                and isinstance(async_result, grpc.RpcError)
                and async_result.code() == grpc.StatusCode.INVALID_ARGUMENT
            )
            is_invalid_argument_http = (
                self.flag.protocol is TritonProtocol.http
                and isinstance(async_result, InferenceServerException)
                and async_result.status() == "400"
            )

            if is_invalid_argument_grpc or is_invalid_argument_http:
                await asyncio.sleep(TRITON_LOAD_DELAY * TRITON_BACKOFF_COEFF**retry_idx)
                self._renew_triton_client(self._triton_client)
                model_spec = self.model_specs[(model_spec.name, model_spec.model_version)]
                continue
            elif isinstance(async_result, Exception):
                raise async_result
            break

        return async_result

    def _call_async(
        self,
        data: list[np.ndarray],
        model_spec: TritonModelSpec,
        parameters: dict | None = None,
    ) -> np.ndarray | None:
        for retry_idx in range(max(2, TRITON_RETRIES)):
            async_result = asyncio.run(self._call_async_item(data=data, model_spec=model_spec, parameters=parameters))

            is_invalid_argument_grpc = (
                self.flag.protocol is TritonProtocol.grpc
                and isinstance(async_result, grpc.RpcError)
                and async_result.code() == grpc.StatusCode.INVALID_ARGUMENT
            )
            is_invalid_argument_http = (
                self.flag.protocol is TritonProtocol.http
                and isinstance(async_result, InferenceServerException)
                and async_result.status() == "400"
            )

            if is_invalid_argument_grpc or is_invalid_argument_http:
                time.sleep(TRITON_LOAD_DELAY * TRITON_BACKOFF_COEFF**retry_idx)
                self._renew_triton_client(self._triton_client)
                model_spec = self.model_specs[(model_spec.name, model_spec.model_version)]
                continue
            elif isinstance(async_result, Exception):
                raise async_result
            break

        return async_result

    async def _call_async_item(
        self,
        data: list[np.ndarray],
        model_spec: TritonModelSpec,
        parameters: dict | None = None,
    ):
        current_grpc_async_tasks = []

        try:
            data_queue = asyncio.Queue(maxsize=ASYNC_TASKS)
            done_event = asyncio.Event()

            generator = asyncio.create_task(
                data_generator(data, model_spec.max_batch_size, data_queue, done_event), name="tritony.data_generator"
            )
            current_grpc_async_tasks.append(generator)

            predict_tasks = [
                asyncio.create_task(
                    send_request_async(self, data_queue, done_event, self.triton_client, model_spec, parameters),
                    name="tritony.send_request_async",
                )
                for idx in range(ASYNC_TASKS)
            ]
            current_grpc_async_tasks.extend(predict_tasks)

            ret = await asyncio.gather(*predict_tasks, generator, data_queue.join())
            ret = sorted(itertools.chain(*ret[:ASYNC_TASKS]))
            result_by_req_id = [output_result_list for req_id, output_result_list in ret]

            if model_spec.max_batch_size == 0:
                result_by_output_name = list(zip(*result_by_req_id))
            else:
                result_by_output_name = list(map(lambda ll: np.concatenate(ll, axis=0), zip(*result_by_req_id)))

            if len(result_by_output_name) == 1:
                result_by_output_name = result_by_output_name[0]

            return result_by_output_name
        except Exception as e:
            loop = asyncio.get_event_loop()
            cancelled_tasks = []

            for t in current_grpc_async_tasks:
                if not t.done() and t is not asyncio.current_task():
                    t.cancel()
                    cancelled_tasks.append(t)
            await asyncio.gather(*cancelled_tasks, return_exceptions=True)
            for task in cancelled_tasks:
                if task.cancelled():
                    continue
                if task.exception() is not None:
                    loop.call_exception_handler(
                        {
                            "message": "unknown",
                            "exception": task.exception(),
                            "task": task,
                        }
                    )

            loop.call_exception_handler(
                {
                    "exception": e,
                }
            )

            return e
        finally:
            del current_grpc_async_tasks

    def _call_request(
        self,
        data: list[np.ndarray],
        model_spec: TritonModelSpec,
        parameters: dict | None = None,
    ) -> list[np.ndarray]:
        for retry_idx in range(max(2, TRITON_RETRIES)):
            try:
                if model_spec.max_batch_size == 0:
                    batch_iterable = zip(*data)
                else:
                    split_indices = np.arange(model_spec.max_batch_size, data[0].shape[0], model_spec.max_batch_size)
                    batch_iterable = zip(*[np.array_split(elem, split_indices) for elem in data])

                result_by_req_id = []
                for inputs_list in batch_iterable:
                    result_by_req_id.append(
                        request(
                            TritonProtocol(self.flag.protocol),
                            self.build_triton_input(inputs_list, model_spec=model_spec, parameters=parameters),
                            self.triton_client,
                            timeout=self.client_timeout,
                            compression=str(self.flag.compression_algorithm),
                        )
                    )

                if model_spec.max_batch_size == 0:
                    result_by_output_name = list(zip(*result_by_req_id))
                else:
                    result_by_output_name = list(map(lambda ll: np.concatenate(ll, axis=0), zip(*result_by_req_id)))

                if len(result_by_output_name) == 1:
                    result_by_output_name = result_by_output_name[0]

            except Exception as e:
                result_by_output_name = e
                if e.code() == grpc.StatusCode.INVALID_ARGUMENT:
                    time.sleep(TRITON_LOAD_DELAY * TRITON_BACKOFF_COEFF**retry_idx)
                    self._renew_triton_client(self._triton_client)
                    model_spec = self.model_specs[(model_spec.name, model_spec.model_version)]
                    continue
                else:
                    raise e

            break

        return result_by_output_name
