from enum import Enum
from typing import Union

from tritonclient import grpc as grpcclient
from tritonclient import http as httpclient


class TritonProtocol(Enum):
    grpc = "grpc"
    http = "http"


class TritonClientFlag:
    def __init__(
        self,
        url: str,
        model_name: str,
        model_version: str = "",
        protocol: str = "grpc",
        streaming: bool = True,
        async_set: bool = False,
        verbose: bool = False,
        input_dims: int = 1,
        compression_algorithm: str = "grpc",
        ssl: bool = False,
    ):
        """

        :param url: host:port
        :param model_name: gptrescorer
        :param model_version: ""
        :param protocol: grpc
        :param streaming:
        :param async_set:
        :param verbose:
        """
        self.url = url
        self.async_set = async_set
        self.model_version = model_version
        self.model_name = model_name
        self.protocol = TritonProtocol(protocol.lower())
        self.streaming = streaming
        self.verbose = verbose
        self.input_dims = input_dims
        self.compression_algorithm = compression_algorithm
        self.ssl = ssl


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
        concurrency = 20 if flag.async_set else 1
        triton_client = httpclient.InferenceServerClient(url=flag.url, verbose=flag.verbose, concurrency=concurrency)

    return triton_client


def get_triton_client(triton_client: Union[grpcclient.InferenceServerClient, httpclient.InferenceServerClient], flag: TritonClientFlag):
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
    # Make sure the model matches our requirements, and get some
    # properties of the model that we need for preprocessing
    model_metadata = triton_client.get_model_metadata(model_name=flag.model_name, model_version=flag.model_version)

    model_config = triton_client.get_model_config(model_name=flag.model_name, model_version=flag.model_version)

    if flag.protocol is TritonProtocol.http:
        from attrdict import AttrDict

        model_metadata = AttrDict(model_metadata)
        model_config = AttrDict(model_config)
    else:
        model_config = model_config.config

    max_batch_size, input_name_list, output_name_list, dtype_list = parse_model(model_metadata, model_config)

    return max_batch_size, input_name_list, output_name_list, dtype_list


def parse_model(model_metadata, model_config):
    # TODO validate model_metadata
    model_config.input
    model_metadata.inputs
    model_metadata.outputs

    return (
        model_config.max_batch_size,
        [input_metadata.name for input_metadata in model_metadata.inputs],
        [output_metadata.name for output_metadata in model_metadata.outputs],
        [input_metadata.datatype for input_metadata in model_metadata.inputs],
    )


def prepare_triton_flag(
    model_name,
    url,
    input_dims,
    run_async=True,
    streaming=False,
    compression_algorithm="gzip",
    ssl=False,
):
    triton_flag = TritonClientFlag(
        url=url,
        model_name=model_name,
        model_version="1",
        protocol="grpc",
        streaming=streaming,
        async_set=run_async,
        verbose=False,
        input_dims=input_dims,  # without batch
        compression_algorithm=compression_algorithm if not streaming else None,
        ssl=ssl,
    )

    return triton_flag
