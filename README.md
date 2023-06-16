# tritony - Tiny configuration for Triton Inference Server

![Pypi](https://badge.fury.io/py/tritony.svg)
![CI](https://github.com/rtzr/tritony/actions/workflows/pre-commit_pytest.yml/badge.svg)
[![Coverage Status](https://coveralls.io/repos/github/rtzr/tritony/badge.svg?branch=main)](https://coveralls.io/github/rtzr/tritony?branch=main)

## What is this?

If you see [the official example](https://github.com/triton-inference-server/client/tree/main/src/python/examples), it is really confusing to use where to start.

Use tritony! You will get really short lines of code like example below.

```python
import argparse
import os
from glob import glob
import numpy as np
from PIL import Image

from tritony import InferenceClient


def preprocess(img, dtype=np.float32, h=224, w=224, scaling="INCEPTION"):
    sample_img = img.convert("RGB")

    resized_img = sample_img.resize((w, h), Image.Resampling.BILINEAR)
    resized = np.array(resized_img)
    if resized.ndim == 2:
        resized = resized[:, :, np.newaxis]

    scaled = (resized / 127.5) - 1
    ordered = np.transpose(scaled, (2, 0, 1))
    
    return ordered.astype(dtype)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, help="Input folder.")
    FLAGS = parser.parse_args()

    client = InferenceClient.create_with("densenet_onnx", "0.0.0.0:8001", input_dims=3, protocol="grpc")
    client.output_kwargs = {"class_count": 1}

    image_data = []
    for filename in glob(os.path.join(FLAGS.image_folder, "*")):
        image_data.append(preprocess(Image.open(filename)))

    result = client(np.asarray(image_data))

    for output in result:
        max_value, arg_max, class_name = output[0].decode("utf-8").split(":")
        print(f"{max_value} ({arg_max}) = {class_name}")
```

## Release Notes

- 23.06.16 Support tritonclient>=2.34.0
- Loosely modified the requirements related to tritonclient


## Key Features

- [x] Simple configuration. Only `$host:$port` and `$model_name` are required.
- [x] Generating asynchronous requests with `asyncio.Queue`
- [x] Simple Model switching
- [ ] Support async tritonclient

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
