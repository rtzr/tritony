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
from tritonclient.grpc import model_config_pb2


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
class TritonModelInput:
    """
    Most of the fields are mapped to model_config_pb2.ModelInput(https://github.com/triton-inference-server/common/blob/a2de06f4c80b2c7b15469fa4d36e5f6445382bad/protobuf/model_config.proto#L317)

    Commented fields are not used.
    """

    name: str
    dtype: str  # data_type mapping to https://github.com/triton-inference-server/client/blob/d257c0e5c3de6e15d6ef289ff2b96cecd0a69b5f/src/python/library/tritonclient/utils/__init__.py#L163-L190

    format: int = 0
    dims: list[int] = []  # dims

    # reshape: list[int] = []
    # is_shape_tensor: bool = False
    # allow_ragged_batch: bool = False
    optional: bool = False


@define
class TritonModelSpec:
    name: str

    max_batch_size: int
    model_input: list[TritonModelInput]

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
) -> (int, list[TritonModelInput], list[str]):
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

    model_config = triton_client.get_model_config(**args)
    if protocol is TritonProtocol.http:
        model_config = dict_to_attr(model_config)
    elif protocol is TritonProtocol.grpc:
        model_config = model_config.config

    max_batch_size, input_list, output_name_list = parse_model(model_config)

    return max_batch_size, input_list, output_name_list


def parse_model_input(
    model_input: model_config_pb2.ModelInput | SimpleNamespace,
) -> TritonModelInput:
    """
    https://github.com/triton-inference-server/common/blob/r23.08/protobuf/model_config.proto#L317-L412
    """
    RAW_DTYPE = model_input.data_type
    if isinstance(model_input.data_type, int):
        RAW_DTYPE = model_config_pb2.DataType.Name(RAW_DTYPE)
    RAW_DTYPE = RAW_DTYPE.strip("TYPE_")

    if RAW_DTYPE == "STRING":
        RAW_DTYPE = "BYTES"  # https://github.com/triton-inference-server/client/blob/d257c0e5c3de6e15d6ef289ff2b96cecd0a69b5f/src/python/library/tritonclient/utils/__init__.py#L188-L189
    return TritonModelInput(
        name=model_input.name,
        dims=model_input.dims,
        dtype=RAW_DTYPE,
        optional=model_input.optional,
    )


def parse_model(
    model_config: model_config_pb2.ModelConfig | SimpleNamespace,
) -> (int, list[TritonModelInput], list[str]):
    return (
        model_config.max_batch_size,
        [parse_model_input(model_config_input) for model_config_input in model_config.input],
        [model_config_output.name for model_config_output in model_config.output],
    )
