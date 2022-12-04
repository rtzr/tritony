from __future__ import annotations

import json
from collections import defaultdict
from enum import Enum
from types import SimpleNamespace
from typing import Any, Optional, Union

import attrs
from attrs import define
from tritonclient import grpc as grpcclient
from tritonclient import http as httpclient


class TritonProtocol(Enum):
    grpc = "grpc"
    http = "http"


COMPRESSION_ALGORITHM_MAP = defaultdict(int)
COMPRESSION_ALGORITHM_MAP.update({"deflate": 1, "gzip": 2})


def dict_to_attr(obj: dict[str, Any]) -> SimpleNamespace:
    """
    Convert dict to attr like object as SimpleNamespace
    Only for grpc client's output
    :param obj:
    :return:
    """
    return json.loads(json.dumps(obj), object_hook=lambda d: SimpleNamespace(**d))


@define
class TritonModelSpec:
    name: str

    max_batch_size: int
    input_name: list[str]
    input_dtype: list[str]

    output_name: list[str]

    model_version: str = "1"


@attrs.define
class TritonClientFlag:
    """
    run_async=True,
    concurrency=6,
    streaming=False,
    compression_algorithm=None,
    ssl=False,
    """

    url: str
    model_name: str
    model_version: str = "1"
    protocol: TritonProtocol | str = attrs.field(converter=TritonProtocol, default=TritonProtocol.grpc)
    streaming: bool = False  # TODO: not implemented
    async_set: bool = True  # TODO: not totally implemented
    concurrency: int = 6  # only for TritonProtocol.http client
    verbose: bool = False
    input_dims: int = 1
    compression_algorithm: Optional[str] = None
    ssl: bool = False


def init_triton_client(
    flag: TritonClientFlag,
) -> Union[grpcclient.InferenceServerClient, httpclient.InferenceServerClient]:
    assert not (
        flag.streaming and not (flag.protocol is TritonProtocol.grpc)
    ), "Streaming is only allowed with gRPC protocol"

    if flag.protocol is TritonProtocol.grpc:
        # Create gRPC client for communicating with the server
        triton_client = grpcclient.InferenceServerClient(url=flag.url, verbose=flag.verbose, ssl=flag.ssl)
    else:
        # Specify large enough concurrency to handle the
        # the number of requests.
        concurrency = flag.concurrency if flag.async_set else 1
        triton_client = httpclient.InferenceServerClient(url=flag.url, verbose=flag.verbose, concurrency=concurrency)

    return triton_client


def get_triton_client(
    triton_client: Union[grpcclient.InferenceServerClient, httpclient.InferenceServerClient],
    model_name: str,
    model_version: str,
    protocol: TritonProtocol,
):
    """
    (required in)
    :param triton_client:
    :param flag:
    - protocol
    - streaming
    - async_set
    - model_name
    - model_version

    :return:
    """

    args = dict(model_name=model_name, model_version=model_version)

    model_metadata = triton_client.get_model_metadata(**args)
    model_config = triton_client.get_model_config(**args)
    if protocol is TritonProtocol.http:
        model_metadata = dict_to_attr(model_metadata)
        model_config = dict_to_attr(model_config)
    elif protocol is TritonProtocol.grpc:
        model_config = model_config.config

    max_batch_size, input_name_list, output_name_list, dtype_list = parse_model(model_metadata, model_config)

    return max_batch_size, input_name_list, output_name_list, dtype_list


def parse_model(model_metadata, model_config):
    return (
        model_config.max_batch_size,
        [input_metadata.name for input_metadata in model_metadata.inputs],
        [output_metadata.name for output_metadata in model_metadata.outputs],
        [input_metadata.datatype for input_metadata in model_metadata.inputs],
    )
