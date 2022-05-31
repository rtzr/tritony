import json

import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        self.model_config = model_config = json.loads(args["model_config"])
        output_config = pb_utils.get_output_config_by_name(model_config, "model_out")
        self.output_dtype = pb_utils.triton_string_to_numpy(output_config["data_type"])

    def execute(self, requests):
        responses = [None for _ in requests]
        for idx, request in enumerate(requests):
            y = pb_utils.get_input_tensor_by_name(request, "model_in").as_numpy()
            out_tensor = pb_utils.Tensor("model_out", y.astype(self.output_dtype))
            inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor])
            responses[idx] = inference_response
        return responses
