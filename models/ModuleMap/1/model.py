from __future__ import annotations

import json
import time
from pathlib import Path
from threading import Lock

import torch
import triton_python_backend_utils as pb_utils
from torch.utils.dlpack import from_dlpack

from .inference import ModuleMapInference, ModuleMapInferenceConfig


class TritonPythonModel:
    def initialize(self, args):
        model_config = json.loads(args["model_config"])
        instance_device_id = int(json.loads(args["model_instance_device_id"]))

        if torch.cuda.is_available():
            self.device = f"cuda:{instance_device_id}"
            self.device_id = instance_device_id
            torch.cuda.set_device(self.device_id)
        else:
            self.device = "cpu"
            self.device_id = 0

        model_dir = Path(args["model_repository"]) / args["model_version"]
        parameters = model_config.get("parameters", {})

        def get_parameter(name, default):
            return parameters.get(name, {"string_value": default})["string_value"]

        self.debug = get_parameter("debug_input", "true").lower() == "true"
        self.debug_every_n = max(int(get_parameter("debug_input_every_n", "1")), 1)
        onnx_model_path = get_parameter(
            "onnx_model_path", str(model_dir / "gnn_v9_meanrms.onnx")
        )
        config = ModuleMapInferenceConfig(
            model_path=onnx_model_path,
            module_map_path=get_parameter("module_map_pattern_path", ""),
            device=self.device,
            device_id=self.device_id,
            gpu_blocks=int(get_parameter("module_map_gpu_blocks", "512")),
            graph_segmentation_method=get_parameter(
                "graph_segmentation_method", "edge_layer_connector"
            ),
            edge_score_cut=float(get_parameter("edge_score_cut", "0.6")),
            onnx_use_cuda=get_parameter("onnx_use_cuda", "true").lower() == "true",
            dwalk_score_cut=float(get_parameter("dwalk_score_cut", "0.01")),
            dwalk_use_gpu=get_parameter("dwalk_use_gpu", "true").lower() == "true",
            dwalk_cc_backend=get_parameter("dwalk_cc_backend", "custom_cuda"),
            dwalk_path_metrics=get_parameter(
                "dwalk_path_metrics", "score_weighted_length"
            ),
            dwalk_walk_min=float(get_parameter("dwalk_walk_min", "0.1")),
            dwalk_walk_max=float(get_parameter("dwalk_walk_max", "0.6")),
            acts_jr_score_cut=float(get_parameter("acts_jr_score_cut", "0.5")),
            acts_jr_use_gpu=get_parameter("acts_jr_use_gpu", "true").lower() == "true",
            acts_jr_cc_backend=get_parameter("acts_jr_cc_backend", "custom_cuda"),
            acts_jr_min_candidate_size=int(
                get_parameter("acts_jr_min_candidate_size", "3")
            ),
            fastwalk_score_cut=float(get_parameter("fastwalk_score_cut", "0.01")),
            fastwalk_walk_min=float(get_parameter("fastwalk_walk_min", "0.1")),
            fastwalk_walk_max=float(get_parameter("fastwalk_walk_max", "0.6")),
            fastwalk_allow_node_reuse=get_parameter(
                "fastwalk_allow_node_reuse", "false"
            ).lower()
            == "true",
            elc_weights_cut=float(get_parameter("elc_weights_cut", "0.01")),
            elc_min_hits=int(get_parameter("elc_min_hits", "3")),
            elc_block_size=int(get_parameter("elc_block_size", "512")),
            elc_max_hits_per_track=int(
                get_parameter("elc_max_hits_per_track", "30")
            ),
        )
        self.inference = ModuleMapInference(config)
        self.graph_segmentation_method = config.graph_segmentation_method

        output_config = pb_utils.get_output_config_by_name(model_config, "LABELS")
        self.output_dtype = pb_utils.triton_string_to_numpy(output_config["data_type"])
        self._request_count = 0
        self._lock = Lock()

        print(
            "[ModuleMap] initialized "
            f"device={self.device}, module_map_path={config.module_map_path}, "
            f"onnx_model_path={config.model_path}, "
            f"onnx_providers={self.inference.onnx_session.get_providers()}, "
            f"onnx_cuda_active={self.inference.onnx_cuda_active}, "
            f"graph_segmentation_method={self.graph_segmentation_method}",
            flush=True,
        )

    def execute(self, requests):
        responses = []
        for request in requests:
            with self._lock:
                self._request_count += 1
                request_count = self._request_count

            try:
                result = self._execute_one(request, request_count)
                out_tensor = pb_utils.Tensor(
                    "LABELS", result.labels.astype(self.output_dtype)
                )
                responses.append(pb_utils.InferenceResponse(output_tensors=[out_tensor]))
            except Exception as exc:
                responses.append(
                    pb_utils.InferenceResponse(
                        error=pb_utils.TritonError(f"ModuleMap inference failed: {exc}")
                    )
                )
        return responses

    def _execute_one(self, request, request_count):
        start = time.perf_counter()
        features_tensor = pb_utils.get_input_tensor_by_name(request, "FEATURES")
        if features_tensor is None:
            raise ValueError("ModuleMap requires input tensor FEATURES")

        features = from_dlpack(features_tensor.to_dlpack()).to(self.device)
        result = self.inference(features)

        if self.debug and request_count % self.debug_every_n == 0:
            elapsed = time.perf_counter() - start
            print(
                "[ModuleMap] "
                f"request={request_count} input_hits={features.shape[0]} "
                f"graph_nodes={result.graph_nodes} graph_edges={result.graph_edges} "
                f"kept_edges={result.kept_edges} labels={result.labels.shape[0]} "
                f"graph_segmentation={self.graph_segmentation_method} "
                f"elapsed={elapsed:.3f}s",
                flush=True,
            )
        return result

    def finalize(self):
        print("[ModuleMap] cleaning up", flush=True)
