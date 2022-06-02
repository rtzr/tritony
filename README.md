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