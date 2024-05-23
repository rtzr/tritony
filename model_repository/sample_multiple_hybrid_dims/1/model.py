import json

import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        self.model_config = model_config = json.loads(args["model_config"])
        output_configs = model_config["output"]

        self.output_name_list = [
            output_config["name"] for output_config in output_configs
        ]
        self.output_dtype_list = [
            pb_utils.triton_string_to_numpy(output_config["data_type"])
            for output_config in output_configs
        ]

    def execute(self, requests):
        responses = [None for _ in requests]
        for idx, request in enumerate(requests):
            current_add_value = int(json.loads(request.parameters()).get("add", 0))
            in_tensor = [
                item.as_numpy() + current_add_value
                for item in request.inputs()
                if "model_in" in item.name()
            ]

            out_tensor = [
                pb_utils.Tensor(output_name, x.astype(output_dtype))
                for x, output_name, output_dtype in zip(
                    in_tensor, self.output_name_list, self.output_dtype_list
                )
            ]
            inference_response = pb_utils.InferenceResponse(output_tensors=out_tensor)
            out_tensor.append(
                pb_utils.Tensor(
                    "model_out2",
                    np.array([current_add_value], dtype=self.output_dtype_list[1]),
                )
            )

            responses[idx] = inference_response
        return responses
