from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import triton_python_backend_utils as pb_utils
from torch.utils.dlpack import from_dlpack

from .inference import MetricLearningInference

os.environ["RAPIDS_NO_INITIALIZE"] = "1"


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to intialize any state associated with this model.

        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """
        # You must parse model_config. JSON string is not parsed here
        self.model_config = model_config = json.loads(args["model_config"])
        self.model_instance_device_id = json.loads(args["model_instance_device_id"])

        parameters = model_config["parameters"]
        self.debug = False
        if "debug" in parameters:
            self.debug = parameters["debug"]["string_value"] == "true"

        def get_parameter(name):
            if name not in parameters:
                raise ValueError(f"Parameter {name} is required but not provided.")
            return parameters[name]["string_value"]

        model_path = Path(args["model_repository"]) / args["model_version"]
        self.inference = MetricLearningInference(
            model_path,
            float(get_parameter("r_max")),
            int(get_parameter("k_max")),
            float(get_parameter("filter_cut")),
            int(get_parameter("filter_batches")),
            float(get_parameter("cc_cut")),
            float(get_parameter("walk_min")),
            float(get_parameter("walk_max")),
            "cuda" if torch.cuda.is_available() else "cpu",
            self.debug,
        )

        # Get OUTPUT0 configuration
        output0_config = pb_utils.get_output_config_by_name(model_config, "LABELS")

        # Convert Triton types to numpy types
        self.output0_dtype = pb_utils.triton_string_to_numpy(output0_config["data_type"])

    def execute(self, requests):
        """`execute` MUST be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference request is made
        for this model. Depending on the batching configuration (e.g. Dynamic
        Batching) used, `requests` may contain multiple requests. Every
        Python model, must create one pb_utils.InferenceResponse for every
        pb_utils.InferenceRequest in `requests`. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse.

        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest

        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """
        output0_dtype = self.output0_dtype

        responses = []
        # print(torch.cuda.is_available())
        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        for request in requests:
            features = pb_utils.get_input_tensor_by_name(request, "FEATURES")
            features = from_dlpack(features.to_dlpack()).to(self.model_instance_device_id)
            if self.debug:
                print(f"{features.shape[0]:,} space points with {features.shape[1]:,} features.")

            # Run inference
            if self.debug:
                torch.save(features, "node_features_for_GNN4Pixel.pt")
            track_ids = self.inference(features)

            if self.debug:
                print(f"output shape: {track_ids.shape}")

            # Create output tensors. You need pb_utils.Tensor
            # objects to create pb_utils.InferenceResponse.
            out_tensor_0 = pb_utils.Tensor("LABELS", track_ids.astype(output0_dtype))
            # Create InferenceResponse. You can set an error here in case
            # there was a problem with handling this inference request.
            # Below is an example of how you can set errors in inference
            # response:
            #
            # pb_utils.InferenceResponse(
            #    output_tensors=..., TritonError("An error occured"))
            inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor_0])
            responses.append(inference_response)

        # You should return a list of pb_utils.InferenceResponse. Length
        # of this list must match the length of `requests` list.
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is OPTIONAL. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print("Cleaning up...")
