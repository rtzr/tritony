import asyncio
import itertools
import logging
import os
import time
from typing import Any, Dict, List, Optional, Union

import grpc
import numpy as np
from more_itertools import chunked
from reretry import retry
from tritonclient import grpc as grpcclient
from tritonclient import http as httpclient
from tritonclient.utils import InferenceServerException

from tritony import ASYNC_TASKS
from tritony.helpers import TritonClientFlag, TritonProtocol, get_triton_client, init_triton_client, prepare_triton_flag

logger = logging.getLogger(__name__)

TRITON_LOAD_DELAY = float(os.environ.get("TRITON_LOAD_DELAY", 3))
TRITON_BACKOFF_COEFF = float(os.environ.get("TRITON_BACKOFF_COEFF", 0.2))
TRITON_RETRIES = int(os.environ.get("TRITON_RETRIES", 5))
TRITON_CLIENT_TIMEOUT = int(os.environ.get("TRITON_CLIENT_TIMEOUT", 5))


async def data_generator(data: List[np.ndarray], batch_size: int, queue: asyncio.Queue, stop: asyncio.Event):
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
        for idx, inputs in enumerate(batch_iterable):
            await queue.put((idx, inputs))
    else:
        batch_iterable = zip(*[chunked(item, batch_size) for item in data])
        for idx, inputs_list in enumerate(batch_iterable):
            await queue.put((idx, [np.asarray(inputs) for inputs in inputs_list]))
    stop.set()


async def send_request_async(inference_client, data_queue, done_event):
    ret = []
    while True:
        data = asyncio.create_task(data_queue.get())
        done = asyncio.create_task(done_event.wait())
        d, _ = await asyncio.wait({data, done}, return_when=asyncio.FIRST_COMPLETED)
        idx, batch_data = None, None
        if data in d:
            idx, batch_data = data.result()
        elif done in d:
            return ret
        try:
            a_pred = await request_async(
                inference_client.build_triton_input(batch_data),
                inference_client.triton_client,
                timeout=inference_client.client_timeout,
            )
        except InferenceServerException as triton_error:
            handel_triton_error(triton_error)
        ret.append((idx, a_pred))
        data_queue.task_done()


def handel_triton_error(triton_error: InferenceServerException):
    """
    https://github.com/triton-inference-server/core/blob/0141e0651c4355bf8a9d1118aac45abda6569997/src/scheduler_utils.cc#L133
    ("Max_queue_size exceeded", "Server not ready", "failed to connect to all addresses")
    """
    runtime_msg = f"{triton_error.status()} with {triton_error.message()}"
    raise RuntimeError(runtime_msg) from triton_error


@retry((InferenceServerException, grpc.RpcError), tries=TRITON_RETRIES, delay=TRITON_BACKOFF_COEFF, backoff=2)
async def request_async(model_input, triton_client, timeout):
    st = time.time()
    result = triton_client.infer(**model_input, timeout=timeout)

    ed = time.time()
    logger.debug(f"{model_input['model_name']} {model_input['inputs'][0].shape()} elapsed: {ed - st:.3f}")

    return [result.as_numpy(outputs.name()) for outputs in model_input["outputs"]]


def worker_parse_args(parser):
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        required=False,
        default=False,
        help="Enable verbose output",
    )
    parser.add_argument(
        "-a",
        "--async",
        dest="async_set",
        action="store_true",
        required=False,
        default=False,
        help="Use asynchronous inference API",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        required=False,
        default=False,
        help="Use streaming inference API. " + "The flag is only available with gRPC protocol.",
    )
    parser.add_argument("-m", "--model-name", type=str, required=True, help="Name of model")
    parser.add_argument(
        "-x",
        "--model-version",
        type=str,
        required=False,
        default="",
        help="Version of model. Default is to use latest version.",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        required=False,
        default=64,
        help="Batch size. Default is 1.",
    )
    parser.add_argument(
        "-u",
        "--url",
        type=str,
        required=False,
        default="localhost:8000",
        help="Inference server URL. Default is localhost:8000.",
    )
    parser.add_argument(
        "-i",
        "--protocol",
        type=str,
        required=False,
        default="HTTP",
        help="Protocol (HTTP/gRPC) used to communicate with " + "the inference service. Default is HTTP.",
    )


class InferenceClient:
    @classmethod
    def create_with(
        cls,
        model: str,
        url: str,
        input_dims: int = 2,
        protocol: str = "grpc",
        run_async: bool = True,
        secure: bool = False,
        compression_algorithm: Optional[str] = None,
    ):
        return cls(
            prepare_triton_flag(
                model_name=model,
                url=url,
                input_dims=input_dims,
                protocol=protocol,
                run_async=run_async,
                compression_algorithm=compression_algorithm,
                ssl=secure,
            )
        )

    def __init__(self, flag: TritonClientFlag):
        self.__version__ = 1

        self.flag = flag
        self.triton_client = init_triton_client(self.flag)
        self._renew_triton_client()

        self.is_async = self.flag.async_set
        self.client_timeout = TRITON_CLIENT_TIMEOUT

        self.sent_count = 0
        self.processed_count = 0

    def __del__(self):
        if self.flag.protocol is TritonProtocol.grpc and self.flag.streaming and hasattr(self, "triton_client"):
            self.triton_client.stop_stream()

    @retry((InferenceServerException, grpc.RpcError), tries=TRITON_RETRIES, delay=TRITON_LOAD_DELAY, backoff=2)
    def _renew_triton_client(self):
        self.triton_client.is_server_live()
        self.triton_client.is_server_ready()
        self.triton_client.is_model_ready(self.flag.model_name)
        (
            self.max_batch_size,
            self.input_name_list,
            self.output_name_list,
            self.dtype_list,
        ) = get_triton_client(self.triton_client, self.flag)

    def _get_request_id(self):
        self.sent_count += 1
        return self.sent_count

    def __call__(self, sequences_or_dict: Union[List[Any], Dict[str, List[Any]]]):
        if self.triton_client is None:
            self._renew_triton_client()

        if type(sequences_or_dict) in [list, np.ndarray]:
            sequences_list = [sequences_or_dict]
        elif type(sequences_or_dict) is dict:
            sequences_list = [sequences_or_dict[input_name] for input_name in self.input_name_list]

        return self._call_async(sequences_list)

    def build_triton_input(self, _input_list: List[np.array]):
        if self.flag.protocol is TritonProtocol.grpc:
            client = grpcclient
        else:
            client = httpclient
        infer_input_list = []
        for _input, _input_name, _dtype in zip(_input_list, self.input_name_list, self.dtype_list):
            infer_input = client.InferInput(_input_name, _input.shape, _dtype)
            infer_input.set_data_from_numpy(_input)
            infer_input_list.append(infer_input)

        infer_requested_output = [client.InferRequestedOutput(output_name) for output_name in self.output_name_list]

        request_id = self._get_request_id()
        request_input = dict(
            model_name=self.flag.model_name,
            inputs=infer_input_list,
            request_id=str(request_id),
            model_version=self.flag.model_version,
            outputs=infer_requested_output,
        )

        if self.flag.protocol is TritonProtocol.grpc:
            request_input.update(
                dict(
                    compression_algorithm=self.flag.compression_algorithm,
                    client_timeout=self.client_timeout,
                )
            )

        return request_input

    def _call_async(self, data: List[np.ndarray]) -> Optional[np.ndarray]:
        async_result = asyncio.run(self._call_async_item(data=data))

        if isinstance(async_result, Exception):
            raise async_result

        return async_result

    async def _call_async_item(self, data: List[np.ndarray]):
        current_grpc_async_tasks = []

        try:
            data_queue = asyncio.Queue(maxsize=ASYNC_TASKS)
            done_event = asyncio.Event()

            generator = asyncio.create_task(data_generator(data, self.max_batch_size, data_queue, done_event))
            current_grpc_async_tasks.append(generator)

            predict_tasks = [
                asyncio.create_task(send_request_async(self, data_queue, done_event)) for _ in range(ASYNC_TASKS)
            ]
            current_grpc_async_tasks.extend(predict_tasks)

            ret = await asyncio.gather(*predict_tasks, generator, data_queue.join())
            ret = sorted(itertools.chain(*ret[:ASYNC_TASKS]))
            result_by_req_id = [output_result_list for req_id, output_result_list in ret]

            if self.max_batch_size == 0:
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

            self.triton_client = None
            return e
        finally:
            del current_grpc_async_tasks
