# tritony - Tiny configuration for Triton Inference Server

![CI](https://github.com/rtzr/tritony/actions/workflows/pre-commit_pytest.yml/badge.svg)

## Key Features

- [x] Simple configuration. Only `$host:$port` and `$model_name` are required.
- [x] Generating asynchronous requests with `asyncio.Queue`

## Requirements

    $ pip install tritonclient[all]

## Install

    $ pip install tritony

## Test

### With Triton

```bash
docker run --rm \
    -v ${PWD}:/models \
    nvcr.io/nvidia/tritonserver:22.01-pyt-python-py3 \
    tritonserver --model-repo=/models
```

```bash
pytest -m -s tests/test_tritony.py
```

### Example with image_client.py

- Follow steps
  in [the official triton server documentation](https://github.com/triton-inference-server/server#serve-a-model-in-3-easy-steps)

```bash
# Download Images from https://github.com/triton-inference-server/server.git
python ./example/image_client.py --image_folder "./server/qa/images"
```