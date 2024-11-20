#!/bin/bash

HERE=$(dirname "$(readlink -f $0)")
PARENT_DIR=$(dirname "$HERE")

docker run -it --rm --name triton_tritony \
    -p8100:8000   \
    -p8101:8001   \
    -p8102:8002    \
    -v "${PARENT_DIR}"/model_repository:/models:ro \
    -e OMP_NUM_THREADS=2 \
    -e OPENBLAS_NUM_THREADS=2 \
    --shm-size=1g  \
    nvcr.io/nvidia/tritonserver:24.05-pyt-python-py3 \
    tritonserver --model-repository=/models \
    --exit-timeout-secs 15 \
    --min-supported-compute-capability 7.0 \
    --log-verbose 0  # 0-nothing, 1-info, 2-debug, 3-trace
