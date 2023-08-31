import json

import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        self.model_config = model_config = json.loads(args["model_config"])
        output_configs = model_config["output"]

        self.output_name_list = [output_config["name"] for output_config in output_configs]
        self.output_dtype_list = [
            pb_utils.triton_string_to_numpy(output_config["data_type"]) for output_config in output_configs
        ]

    def execute(self, requests):
        responses = [None for _ in requests]
        for idx, request in enumerate(requests):
            current_add_value = int(json.loads(request.parameters()).get("add", 0))
            optional_in_tensor = pb_utils.get_input_tensor_by_name(request, "optional_model_sub")
            if optional_in_tensor:
                optional_in_tensor = optional_in_tensor.as_numpy()
            else:
                optional_in_tensor = 0

            in_tensor = [
                item.as_numpy() + current_add_value - optional_in_tensor
                for item in request.inputs()
                if item.name() == "model_in"
            ]
            out_tensor = [
                pb_utils.Tensor(output_name, x.astype(output_dtype))
                for x, output_name, output_dtype in zip(in_tensor, self.output_name_list, self.output_dtype_list)
            ]
            inference_response = pb_utils.InferenceResponse(output_tensors=out_tensor)
            responses[idx] = inference_response
        return responses
