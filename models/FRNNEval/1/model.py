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
    original FRNN and libFRNN. On disagreement, the embedding and both edge lists are
    saved. To mimic MetricLearning for clients such as Athena, LABELS is always the
    dummy track candidates [0, 1, 2, 3, 4, 5, -1].
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
            save_eval_metrics=get_parameter("save_eval_metrics").lower() == "true",
            save_data=get_parameter("save_data").lower() == "true",
            measure_memory=get_parameter("measure_memory").lower() == "true",
            r_max=float(get_parameter("r_max")),
            k_max=int(get_parameter("k_max")),
        )
        self.debug = config.debug
        self.evaluator = FRNNEval(config)

        output0_config = pb_utils.get_output_config_by_name(model_config, "LABELS")
        self.output0_dtype = pb_utils.triton_string_to_numpy(output0_config["data_type"])
        self.dummy_labels = np.arange(6, dtype=self.output0_dtype)

    def execute(self, requests):
        responses = []
        for request in requests:
            features = pb_utils.get_input_tensor_by_name(request, "FEATURES")
            features = from_dlpack(features.to_dlpack()).to(self.device)
            if self.debug:
                print(f"{features.shape[0]:,} space points with {features.shape[1]:,} features.")

            # Errors are only logged: the client always gets the dummy labels.
            try:
                if features.shape[0] > 2:
                    num_diff = self.evaluator(features, request_id=request.request_id())
                    if num_diff > 0:
                        print(f"request {request.request_id()}: {num_diff:,} differing edges")
            except Exception as error:
                print(f"request {request.request_id()} failed: {error}")

            out_tensor_0 = pb_utils.Tensor("LABELS", self.dummy_labels)
            responses.append(pb_utils.InferenceResponse(output_tensors=[out_tensor_0]))
        return responses

    def finalize(self):
        print("Cleaning up...")
