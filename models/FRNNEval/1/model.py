from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import triton_python_backend_utils as pb_utils
from torch.utils.dlpack import from_dlpack

from .frnn_eval import FRNNEval, FRNNEvalConfig


class TritonPythonModel:
    """Run the MetricLearning embedding and compare the edge lists built by the
    original FRNN and libFRNN. RESULT is 0 when they agree, otherwise the number of
    differing edges. On disagreement, the embedding and both edge lists are saved.
    """

    def initialize(self, args):
        self.model_config = model_config = json.loads(args["model_config"])
        self.model_instance_device_id = json.loads(args["model_instance_device_id"])

        if torch.cuda.is_available():
            self.device = f"cuda:{self.model_instance_device_id}"
            torch.cuda.set_device(self.model_instance_device_id)
        else:
            raise RuntimeError("FRNNEval requires a GPU: both FRNN libraries are CUDA only.")

        parameters = model_config["parameters"]

        def get_parameter(name):
            if name not in parameters:
                raise ValueError(f"Parameter {name} is required but not provided.")
            return parameters[name]["string_value"]

        # Relative paths: the embedding model is resolved against the model repository,
        # the output directory against the server's working directory.
        embedding_model_dir = Path(get_parameter("embedding_model_dir"))
        if not embedding_model_dir.is_absolute():
            embedding_model_dir = Path(args["model_repository"]) / embedding_model_dir

        config = FRNNEvalConfig(
            embedding_model_dir=embedding_model_dir,
            output_dir=get_parameter("output_dir"),
            device=self.device,
            auto_cast=get_parameter("auto_cast").lower() == "true",
            debug=get_parameter("debug").lower() == "true",
            r_max=float(get_parameter("r_max")),
            k_max=int(get_parameter("k_max")),
        )
        self.debug = config.debug
        self.evaluator = FRNNEval(config)

        output0_config = pb_utils.get_output_config_by_name(model_config, "RESULT")
        self.output0_dtype = pb_utils.triton_string_to_numpy(output0_config["data_type"])

    def execute(self, requests):
        responses = []
        for request in requests:
            features = pb_utils.get_input_tensor_by_name(request, "FEATURES")
            features = from_dlpack(features.to_dlpack()).to(self.device)
            if self.debug:
                print(f"{features.shape[0]:,} space points with {features.shape[1]:,} features.")

            try:
                result = self.evaluator(features) if features.shape[0] > 2 else 0
            except Exception as error:  # report per request instead of killing the stub
                responses.append(
                    pb_utils.InferenceResponse(
                        output_tensors=[], error=pb_utils.TritonError(str(error))
                    )
                )
                continue

            out_tensor_0 = pb_utils.Tensor(
                "RESULT", np.array([result], dtype=self.output0_dtype)
            )
            responses.append(pb_utils.InferenceResponse(output_tensors=[out_tensor_0]))
        return responses

    def finalize(self):
        print("Cleaning up...")
