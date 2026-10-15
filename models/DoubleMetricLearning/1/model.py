from __future__ import annotations

import json
import os
from pathlib import Path
from threading import Lock

import numpy as np
import torch
import triton_python_backend_utils as pb_utils
from torch.utils.dlpack import from_dlpack

from .inference import MetricLearningInference, MetricLearningInferenceConfig

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

        # Store device in consistent format
        if torch.cuda.is_available():
            self.device = f"cuda:{self.model_instance_device_id}"
            self.device_id = self.model_instance_device_id
        else:
            self.device = "cpu"
            self.device_id = "cpu"

        if torch.cuda.is_available():
            torch.cuda.set_device(self.device_id)

        parameters = model_config["parameters"]
        self.debug = False

        def get_parameter(name):
            if name not in parameters:
                raise ValueError(f"Parameter {name} is required but not provided.")
            return parameters[name]["string_value"]

        def get_optional_parameter(name, default):
            if name in parameters:
                return parameters[name]["string_value"]
            return default

        model_path = Path(args["model_repository"]) / args["model_version"]
        auto_cast = get_parameter("auto_cast").lower() == "true"
        compiling = get_parameter("compiling").lower() == "true"
        graph_segmentation_method = parameters.get(
            "graph_segmentation_method", {"string_value": "fastwalk"}
        )["string_value"]
        use_dwalk = get_optional_parameter("use_dwalk", "False").lower() == "true"
        dwalk_use_gpu = get_optional_parameter("dwalk_use_gpu", "True").lower() == "true"
        dwalk_cc_backend = get_optional_parameter("dwalk_cc_backend", "custom_cuda")
        dwalk_path_metrics = get_optional_parameter("dwalk_path_metrics", "score_weighted_length")
        jr_use_gpu = (
            parameters.get("jr_use_gpu", {"string_value": "True"})["string_value"].lower()
            == "true"
        )
        jr_cc_backend = parameters.get(
            "jr_cc_backend", {"string_value": "custom_cuda"}
        )["string_value"]
        jr_score_cut = float(
            parameters.get("jr_score_cut", {"string_value": "0.9"})["string_value"]
        )
        jr_junction_cut = float(
            parameters.get("jr_junction_cut", {"string_value": "0.2"})["string_value"]
        )
        jr_min_chain_length = int(
            parameters.get("jr_min_chain_length", {"string_value": "5"})["string_value"]
        )
        jr_skip_remove_cycles = (
            parameters.get("jr_skip_remove_cycles", {"string_value": "True"})["string_value"].lower()
            == "true"
        )
        acts_jr_use_gpu = (
            parameters.get("acts_jr_use_gpu", {"string_value": "True"})[
                "string_value"
            ].lower()
            == "true"
        )
        acts_jr_cc_backend = parameters.get(
            "acts_jr_cc_backend", {"string_value": "custom_cuda"}
        )["string_value"]
        acts_jr_min_candidate_size = int(
            parameters.get("acts_jr_min_candidate_size", {"string_value": "3"})[
                "string_value"
            ]
        )
        elc_weights_cut = float(
            parameters.get("elc_weights_cut", {"string_value": "0.01"})[
                "string_value"
            ]
        )
        elc_min_hits = int(
            parameters.get("elc_min_hits", {"string_value": "3"})["string_value"]
        )
        elc_block_size = int(
            parameters.get("elc_block_size", {"string_value": "512"})["string_value"]
        )
        elc_max_hits_per_track = int(
            parameters.get("elc_max_hits_per_track", {"string_value": "30"})[
                "string_value"
            ]
        )
        cc_cut = (
            float(parameters["cc_cut"]["string_value"])
            if "cc_cut" in parameters
            else None
        )
        collect_timing = (
            parameters.get("collect_timing", {"string_value": "True"})["string_value"].lower()
            == "true"
        )
        filter_timing_skip_events = max(
            int(parameters.get("filter_timing_skip_events", {"string_value": "0"})["string_value"]),
            0,
        )
        timing_report_interval = int(
            parameters.get("timing_report_interval", {"string_value": "100"})[
                "string_value"
            ]
        )
        config = MetricLearningInferenceConfig(
            model_path=model_path,
            device=self.device,
            auto_cast=auto_cast,
            compiling=compiling,
            graph_segmentation_method=graph_segmentation_method,
            use_dwalk=use_dwalk,
            dwalk_use_gpu=dwalk_use_gpu,
            dwalk_cc_backend=dwalk_cc_backend,
            dwalk_path_metrics=dwalk_path_metrics,
            jr_use_gpu=jr_use_gpu,
            jr_cc_backend=jr_cc_backend,
            jr_score_cut=jr_score_cut,
            jr_junction_cut=jr_junction_cut,
            jr_min_chain_length=jr_min_chain_length,
            jr_skip_remove_cycles=jr_skip_remove_cycles,
            acts_jr_use_gpu=acts_jr_use_gpu,
            acts_jr_cc_backend=acts_jr_cc_backend,
            acts_jr_min_candidate_size=acts_jr_min_candidate_size,
            elc_weights_cut=elc_weights_cut,
            elc_min_hits=elc_min_hits,
            elc_block_size=elc_block_size,
            elc_max_hits_per_track=elc_max_hits_per_track,
            cc_cut=cc_cut,
            collect_timing=collect_timing,
            filter_timing_skip_events=filter_timing_skip_events,
            debug=self.debug,
            save_debug_data=False,
        )

        self.inference = MetricLearningInference(config)
        self.collect_timing = collect_timing
        self.timing_report_interval = max(timing_report_interval, 1)
        self._timing_lock = Lock()
        self._timed_request_count = 0
        self._filter_time_total = 0.0
        self._graph_segmentation_time_total = 0.0

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

        Returns:
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
            features = from_dlpack(features.to_dlpack()).to(self.device)
            if self.debug:
                print(f"{features.shape[0]:,} space points with {features.shape[1]:,} features.")

            track_ids = (
                self.inference(features)
                if features.shape[0] > 2
                else np.array([-1], dtype=np.int64)
            )

            if self.collect_timing and self.inference.last_timing:
                with self._timing_lock:
                    self._timed_request_count += 1
                    self._filter_time_total += self.inference.last_timing.get(
                        "filtering_seconds", 0.0
                    )
                    self._graph_segmentation_time_total += self.inference.last_timing.get(
                        "graph_segmentation_seconds", 0.0
                    )
                    if self._timed_request_count % self.timing_report_interval == 0:
                        average_filter_ms = (
                            1000.0 * self._filter_time_total / self._timed_request_count
                        )
                        average_graph_segmentation_ms = (
                            1000.0
                            * self._graph_segmentation_time_total
                            / self._timed_request_count
                        )
                        print(
                            "[timing] DoubleMetricLearning average filtering time: "
                            f"{average_filter_ms:.3f} ms over {self._timed_request_count} requests",
                            flush=True,
                        )
                        print(
                            "[timing] DoubleMetricLearning average graph segmentation time: "
                            f"{average_graph_segmentation_ms:.3f} ms over {self._timed_request_count} requests",
                            flush=True,
                        )

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
