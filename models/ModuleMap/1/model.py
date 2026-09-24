import json
import sys
import time
from pathlib import Path
from threading import Lock

import numpy as np
import torch
import triton_python_backend_utils as pb_utils
from torch_geometric.data import Data
from torch.utils.dlpack import from_dlpack


INPUT_FEATURES = [
    "x",
    "y",
    "z",
    "module_id",
    "hit_id",
    "r",
    "phi",
    "eta",
    "cluster_r_1",
    "cluster_phi_1",
    "cluster_z_1",
    "cluster_eta_1",
    "cluster_r_2",
    "cluster_phi_2",
    "cluster_z_2",
    "cluster_eta_2",
]


class TritonPythonModel:
    def initialize(self, args):
        self.model_config = model_config = json.loads(args["model_config"])
        self.model_instance_device_id = json.loads(args["model_instance_device_id"])

        if torch.cuda.is_available():
            self.device = f"cuda:{self.model_instance_device_id}"
            self.device_id = int(self.model_instance_device_id)
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
        self.module_map_path = get_parameter("module_map_pattern_path", "")
        self.onnx_model_path = get_parameter(
            "onnx_model_path", str(model_dir / "gnn_v9_meanrms.onnx")
        )
        self.gpu_blocks = int(get_parameter("module_map_gpu_blocks", "512"))
        self.graph_segmentation_method = get_parameter(
            "graph_segmentation_method", "edge_layer_connector"
        ).lower()
        self.edge_score_cut = float(get_parameter("edge_score_cut", "0.6"))
        self.onnx_use_cuda = get_parameter("onnx_use_cuda", "true").lower() == "true"
        self.dwalk_score_cut = float(get_parameter("dwalk_score_cut", "0.01"))
        self.dwalk_use_gpu = get_parameter("dwalk_use_gpu", "true").lower() == "true"
        self.dwalk_cc_backend = get_parameter("dwalk_cc_backend", "custom_cuda")
        self.dwalk_path_metrics = get_parameter("dwalk_path_metrics", "score_weighted_length")
        self.dwalk_walk_min = float(get_parameter("dwalk_walk_min", "0.1"))
        self.dwalk_walk_max = float(get_parameter("dwalk_walk_max", "0.6"))
        self.acts_jr_score_cut = float(get_parameter("acts_jr_score_cut", "0.5"))
        self.acts_jr_use_gpu = get_parameter("acts_jr_use_gpu", "true").lower() == "true"
        self.acts_jr_cc_backend = get_parameter("acts_jr_cc_backend", "custom_cuda")
        self.acts_jr_min_candidate_size = int(get_parameter("acts_jr_min_candidate_size", "3"))
        self.fastwalk_score_cut = float(get_parameter("fastwalk_score_cut", "0.01"))
        self.fastwalk_walk_min = float(get_parameter("fastwalk_walk_min", "0.1"))
        self.fastwalk_walk_max = float(get_parameter("fastwalk_walk_max", "0.6"))
        self.fastwalk_allow_node_reuse = (
            get_parameter("fastwalk_allow_node_reuse", "false").lower() == "true"
        )
        self.elc_weights_cut = float(get_parameter("elc_weights_cut", "0.01"))
        self.elc_min_hits = int(get_parameter("elc_min_hits", "3"))
        self.elc_block_size = int(get_parameter("elc_block_size", "512"))
        self.elc_max_hits_per_track = int(get_parameter("elc_max_hits_per_track", "30"))

        repo_root = Path.cwd()
        if not (repo_root / "taas_graph_segmentation").exists():
            repo_root = Path(__file__).resolve().parents[3]
        if (repo_root / "taas_cuda_ext").exists() and str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from taas_cuda_ext.module_map_graph import ModuleMapGraphCuda
        from taas_cuda_ext import _edge_layer_connector_cuda
        from taas_graph_segmentation.acts_junction_removal import acts_junction_removal
        from taas_graph_segmentation.dwalk import (
            dwalk,
            get_simple_path as get_dwalk_simple_path,
        )
        from taas_graph_segmentation import fastwalkthrough as fastwalk

        self.acts_junction_removal = acts_junction_removal
        self.dwalk = dwalk
        self.get_dwalk_simple_path = get_dwalk_simple_path
        self.fastwalk = fastwalk
        self.onnx_session = self._create_onnx_session(self.onnx_model_path)
        self.onnx_input_names = [input_meta.name for input_meta in self.onnx_session.get_inputs()]
        self.onnx_output_name = self.onnx_session.get_outputs()[0].name
        self.onnx_cuda_active = (
            self.onnx_use_cuda
            and torch.cuda.is_available()
            and "CUDAExecutionProvider" in self.onnx_session.get_providers()
        )
        if self.onnx_use_cuda and not self.onnx_cuda_active:
            print(
                "[ModuleMap] WARNING: ONNX CUDA was requested by default/config, "
                "but CUDAExecutionProvider is not available or CUDA is not visible; "
                "falling back to ONNX CPU.",
                flush=True,
            )
        self.module_map_graph = ModuleMapGraphCuda(
            self.module_map_path,
            gpu_blocks=self.gpu_blocks,
            gpu_device=self.device_id,
        )
        self.edge_layer_connector_cuda = _edge_layer_connector_cuda

        output_config = pb_utils.get_output_config_by_name(model_config, "LABELS")
        self.output_dtype = pb_utils.triton_string_to_numpy(output_config["data_type"])
        self._request_count = 0
        self._lock = Lock()

        print(
            "[ModuleMap] initialized "
            f"device={self.device}, module_map_path={self.module_map_path}, "
                f"onnx_model_path={self.onnx_model_path}, "
                f"onnx_providers={self.onnx_session.get_providers()}, "
                f"onnx_cuda_active={self.onnx_cuda_active}, "
                f"graph_segmentation_method={self.graph_segmentation_method}",
                flush=True,
            )

    def _create_onnx_session(self, model_path):
        import onnxruntime as ort

        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1

        providers = ["CPUExecutionProvider"]
        if self.onnx_use_cuda and torch.cuda.is_available():
            providers = [
                (
                    "CUDAExecutionProvider",
                    {"device_id": self.device_id},
                ),
                "CPUExecutionProvider",
            ]

        return ort.InferenceSession(
            str(model_path),
            sess_options=session_options,
            providers=providers,
        )

    def execute(self, requests):
        responses = []
        for request in requests:
            with self._lock:
                self._request_count += 1
                request_count = self._request_count

            try:
                labels = self._execute_one(request, request_count)
                out_tensor = pb_utils.Tensor("LABELS", labels.astype(self.output_dtype))
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
        if features.ndim != 2 or features.shape[1] != len(INPUT_FEATURES):
            raise ValueError(
                "ModuleMap expects FEATURES with shape [N, 16] ordered as "
                f"{INPUT_FEATURES}; got {tuple(features.shape)}"
            )
        if features.shape[0] < 3:
            return torch.tensor([-1], dtype=torch.int64, device="cpu").numpy()

        node_features, edge_index, edge_features, hit_ids = self.module_map_graph.build(
            features.contiguous()
        )

        if edge_index.numel() == 0 or hit_ids.numel() == 0:
            return torch.tensor([-1], dtype=torch.int64, device="cpu").numpy()

        edge_scores = self._run_edge_classifier(node_features, edge_index, edge_features)
        if self.graph_segmentation_method in {"edge_layer_connector", "elc"}:
            keep = edge_scores > self.edge_score_cut
            edge_index = edge_index[:, keep]
            edge_scores = edge_scores[keep]
            labels = self._run_edge_layer_connector(hit_ids, edge_index, edge_scores)
            kept_edges = int(edge_scores.numel())
        elif self.graph_segmentation_method in {"dwalk", "d-walk"}:
            labels, kept_edges = self._run_dwalk(
                node_features,
                hit_ids,
                edge_index,
                edge_scores,
            )
        elif self.graph_segmentation_method in {
            "acts_junction_removal",
            "acts-junction-removal",
            "acts_jr",
        }:
            labels, kept_edges = self._run_acts_junction_removal(
                node_features,
                hit_ids,
                edge_index,
                edge_scores,
            )
        elif self.graph_segmentation_method in {"fastwalk", "fastwalkthrough"}:
            labels, kept_edges = self._run_fastwalkthrough(
                node_features,
                hit_ids,
                edge_index,
                edge_scores,
            )
        else:
            raise ValueError(
                "Unsupported ModuleMap graph_segmentation_method "
                f"'{self.graph_segmentation_method}'. Expected edge_layer_connector, "
                "dwalk, acts_junction_removal, or fastwalk."
            )
        if self.debug and request_count % self.debug_every_n == 0:
            elapsed = time.perf_counter() - start
            print(
                "[ModuleMap] "
                f"request={request_count} input_hits={features.shape[0]} "
                f"graph_nodes={node_features.shape[0]} graph_edges={edge_features.shape[0]} "
                f"kept_edges={kept_edges} labels={labels.shape[0]} "
                f"graph_segmentation={self.graph_segmentation_method} "
                f"elapsed={elapsed:.3f}s",
                flush=True,
            )
        return labels

    def _run_edge_classifier(self, node_features, edge_index, edge_features):
        if self.onnx_cuda_active:
            return self._run_edge_classifier_cuda(node_features, edge_index, edge_features)

        ort_inputs = {
            "x": node_features.detach().cpu().numpy().astype(np.float32, copy=False),
            "edge_index": edge_index.detach().cpu().numpy().astype(np.int64, copy=False),
            "edge_attr": edge_features.detach().cpu().numpy().astype(np.float32, copy=False),
        }
        output = self.onnx_session.run(None, ort_inputs)[0]
        logits = torch.as_tensor(output, dtype=torch.float32, device=self.device).reshape(-1)
        return torch.sigmoid(logits)

    def _run_edge_classifier_cuda(self, node_features, edge_index, edge_features):
        output = torch.empty(
            (edge_index.shape[1],),
            dtype=torch.float32,
            device=self.device,
        )
        io_binding = self.onnx_session.io_binding()
        tensors = {
            self.onnx_input_names[0]: node_features.contiguous(),
            self.onnx_input_names[1]: edge_index.contiguous(),
            self.onnx_input_names[2]: edge_features.contiguous(),
        }
        dtypes = {
            torch.float32: np.float32,
            torch.int64: np.int64,
        }
        for name, tensor in tensors.items():
            io_binding.bind_input(
                name=name,
                device_type="cuda",
                device_id=self.device_id,
                element_type=dtypes[tensor.dtype],
                shape=tuple(tensor.shape),
                buffer_ptr=tensor.data_ptr(),
            )
        io_binding.bind_output(
            name=self.onnx_output_name,
            device_type="cuda",
            device_id=self.device_id,
            element_type=np.float32,
            shape=tuple(output.shape),
            buffer_ptr=output.data_ptr(),
        )
        self.onnx_session.run_with_iobinding(io_binding)
        return torch.sigmoid(output.reshape(-1))

    def _run_dwalk(self, node_features, hit_ids, edge_index, edge_scores):
        if edge_index.numel() == 0 or edge_scores.numel() == 0:
            return torch.tensor([-1], dtype=torch.int64, device="cpu").numpy(), 0

        graph = Data(
            edge_index=edge_index,
            hit_id=hit_ids.long(),
            num_nodes=int(node_features.shape[0]),
            R=node_features[:, 0].float(),
            hit_z=node_features[:, 2].float(),
        )
        graph.scores = edge_scores.float()
        kept_edges = int((edge_scores > self.dwalk_score_cut).sum().item())
        simple_tracks, dwalk_graph = self.get_dwalk_simple_path(
            graph,
            "scores",
            self.dwalk_score_cut,
            use_gpu=self.dwalk_use_gpu,
            cc_backend=self.dwalk_cc_backend,
            return_tensors=True,
        )
        dwalk_tracks = self.dwalk(
            dwalk_graph,
            "scores",
            self.dwalk_walk_min,
            self.dwalk_walk_max,
            self.dwalk_path_metrics,
            use_gpu=self.dwalk_use_gpu,
            cc_backend=self.dwalk_cc_backend,
            return_tensors=True,
        )
        tracks = simple_tracks + dwalk_tracks
        return self._tracks_to_labels(tracks), kept_edges

    def _run_acts_junction_removal(self, node_features, hit_ids, edge_index, edge_scores):
        if edge_index.numel() == 0 or edge_scores.numel() == 0:
            return torch.tensor([-1], dtype=torch.int64, device="cpu").numpy(), 0

        keep = edge_scores > self.acts_jr_score_cut
        edge_index = edge_index[:, keep]
        edge_scores = edge_scores[keep]
        kept_edges = int(edge_scores.numel())
        if kept_edges == 0:
            return torch.tensor([-1], dtype=torch.int64, device="cpu").numpy(), 0

        graph = Data(
            edge_index=edge_index,
            hit_id=hit_ids.long(),
            num_nodes=int(node_features.shape[0]),
            R=node_features[:, 0].float(),
            hit_z=node_features[:, 2].float(),
        )
        graph.scores = edge_scores.float()
        tracks = self.acts_junction_removal(
            graph,
            "scores",
            min_candidate_size=self.acts_jr_min_candidate_size,
            use_gpu=self.acts_jr_use_gpu,
            cc_backend=self.acts_jr_cc_backend,
        )
        return self._tracks_to_labels(tracks), kept_edges

    def _run_fastwalkthrough(self, node_features, hit_ids, edge_index, edge_scores):
        if edge_index.numel() == 0 or edge_scores.numel() == 0:
            return torch.tensor([-1], dtype=torch.int64, device="cpu").numpy(), 0

        keep = edge_scores > self.fastwalk_score_cut
        keep &= self._forward_edge_mask(node_features, edge_index)
        kept_edges = int(keep.sum().item())
        if kept_edges == 0:
            return torch.tensor([-1], dtype=torch.int64, device="cpu").numpy(), 0

        cpu_edge_index = edge_index[:, keep].detach().cpu().long()
        cpu_scores = edge_scores[keep].detach().cpu().float()
        cpu_hit_ids = hit_ids.detach().cpu().long()
        cpu_node_features = node_features.detach().cpu()

        graph = Data(
            edge_index=cpu_edge_index,
            hit_id=cpu_hit_ids,
            num_nodes=int(node_features.shape[0]),
            hit_r=cpu_node_features[:, 0].float(),
            hit_z=cpu_node_features[:, 2].float(),
        )
        graph, cpu_scores = self._remove_isolated_nodes_with_edge_scores(graph, cpu_scores)
        graph.scores = cpu_scores
        if graph.edge_index.numel() == 0 or graph.hit_id.numel() == 0:
            return torch.tensor([-1], dtype=torch.int64, device="cpu").numpy(), 0

        simple_tracks, fastwalk_graph = self.fastwalk.get_simple_path(graph)
        fastwalk_graph = self._ensure_edge_scores(
            fastwalk_graph, graph, "scores", "fastwalk_graph"
        )
        walk_tracks = self.fastwalk.walk_through(
            fastwalk_graph,
            "scores",
            self.fastwalk_walk_min,
            self.fastwalk_walk_max,
            self.fastwalk_allow_node_reuse,
        )
        return self._tracks_to_labels(simple_tracks + list(walk_tracks)), kept_edges

    def _remove_isolated_nodes_with_edge_scores(self, graph, edge_scores):
        edge_index = graph.edge_index
        if edge_index.numel() == 0:
            cleaned = Data(
                edge_index=edge_index,
                hit_id=graph.hit_id[:0],
                num_nodes=0,
                hit_r=graph.hit_r[:0],
                hit_z=graph.hit_z[:0],
            )
            return cleaned, edge_scores[:0]

        node_ids = torch.unique(edge_index.flatten(), sorted=True)
        old_to_new = torch.full(
            (int(graph.num_nodes),),
            -1,
            dtype=torch.long,
            device=edge_index.device,
        )
        old_to_new[node_ids] = torch.arange(
            node_ids.numel(), dtype=torch.long, device=edge_index.device
        )

        cleaned = Data(
            edge_index=old_to_new[edge_index],
            hit_id=graph.hit_id[node_ids],
            num_nodes=int(node_ids.numel()),
            hit_r=graph.hit_r[node_ids],
            hit_z=graph.hit_z[node_ids],
        )
        return cleaned, edge_scores

    def _ensure_edge_scores(self, graph, source_graph, score_name, context):
        edge_count = int(graph.edge_index.shape[1])
        scores = graph[score_name]
        if torch.is_tensor(scores) and int(scores.numel()) == edge_count:
            return graph

        source_scores = source_graph[score_name]
        if not torch.is_tensor(source_scores) or int(source_scores.numel()) != int(
            source_graph.edge_index.shape[1]
        ):
            raise ValueError(
                f"{context}: cannot repair {score_name}; "
                f"source_scores={getattr(source_scores, 'numel', lambda: 'n/a')()} "
                f"source_edges={source_graph.edge_index.shape[1]}"
            )

        score_by_edge = {}
        source_edges = source_graph.edge_index.t().tolist()
        source_hit_ids = source_graph.hit_id.detach().cpu().long().tolist()
        source_values = source_scores.detach().cpu().float().tolist()
        for edge, score in zip(source_edges, source_values):
            key = (
                int(source_hit_ids[int(edge[0])]),
                int(source_hit_ids[int(edge[1])]),
            )
            if key not in score_by_edge or score > score_by_edge[key]:
                score_by_edge[key] = float(score)

        repaired = []
        missing = 0
        graph_hit_ids = graph.hit_id.detach().cpu().long().tolist()
        for edge in graph.edge_index.t().tolist():
            key = (
                int(graph_hit_ids[int(edge[0])]),
                int(graph_hit_ids[int(edge[1])]),
            )
            if key not in score_by_edge:
                missing += 1
            else:
                repaired.append(score_by_edge[key])
        if missing:
            raise ValueError(
                f"{context}: cannot map {missing} {score_name} value(s) "
                "from source graph by hit_id edge pairs"
            )
        graph[score_name] = torch.tensor(repaired, dtype=torch.float32, device=graph.edge_index.device)
        return graph

    def _forward_edge_mask(self, node_features, edge_index):
        radial_distance = node_features[:, 0].square() + node_features[:, 2].square()
        src = edge_index[0]
        dst = edge_index[1]
        return (radial_distance[src] < radial_distance[dst]) | (
            (radial_distance[src] == radial_distance[dst]) & (src < dst)
        )

    def _tracks_to_labels(self, tracks):
        pieces = []
        total = 0
        for track_group in tracks:
            if hasattr(track_group, "grouped_hit_ids") and hasattr(track_group, "counts"):
                start = 0
                for count in track_group.counts.detach().cpu().tolist():
                    end = start + int(count)
                    if count <= 0:
                        start = end
                        continue
                    track = (
                        track_group.grouped_hit_ids[start:end]
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.int64)
                    )
                    pieces.append(track)
                    total += len(track) + 1
                    start = end
            elif torch.is_tensor(track_group):
                track = track_group.detach().cpu().numpy().astype(np.int64)
                if track.size == 0:
                    continue
                pieces.append(track)
                total += len(track) + 1
            else:
                track = np.asarray(track_group, dtype=np.int64)
                if track.size == 0:
                    continue
                pieces.append(track)
                total += len(track) + 1

        if not pieces:
            return torch.tensor([-1], dtype=torch.int64, device="cpu").numpy()

        labels = np.empty(total, dtype=np.int64)
        cursor = 0
        for track in pieces:
            labels[cursor : cursor + len(track)] = track
            cursor += len(track)
            labels[cursor] = -1
            cursor += 1
        return labels

    def _run_edge_layer_connector(self, hit_ids, edge_index, edge_scores):
        if edge_index.numel() == 0 or edge_scores.numel() == 0:
            return torch.tensor([-1], dtype=torch.int64, device="cpu").numpy()

        padded_hits, counts, _ = self.edge_layer_connector_cuda.edge_layer_connector_cuda(
            hit_ids.to(dtype=torch.int32).contiguous(),
            edge_index[0].to(dtype=torch.int32).contiguous(),
            edge_index[1].to(dtype=torch.int32).contiguous(),
            edge_scores.to(dtype=torch.float32).contiguous(),
            float(self.elc_weights_cut),
            int(self.elc_min_hits),
            int(self.elc_block_size),
            int(self.elc_max_hits_per_track),
        )
        if counts.numel() == 0:
            return torch.tensor([-1], dtype=torch.int64, device="cpu").numpy()

        counts = counts.long()
        num_tracks = int(counts.numel())
        padded_width = int(padded_hits.numel() // num_tracks)
        padded_hits = padded_hits.long().view(num_tracks, padded_width)

        pieces = []
        total = 0
        for track_index, count in enumerate(counts.detach().cpu().tolist()):
            if count <= 0:
                continue
            track = padded_hits[track_index, :count].detach().cpu().numpy().astype(np.int64)
            pieces.append(track)
            total += len(track) + 1
        if not pieces:
            return torch.tensor([-1], dtype=torch.int64, device="cpu").numpy()

        labels = np.empty(total, dtype=np.int64)
        cursor = 0
        for track in pieces:
            labels[cursor : cursor + len(track)] = track
            cursor += len(track)
            labels[cursor] = -1
            cursor += 1
        return labels

    def finalize(self):
        print("[ModuleMap] cleaning up", flush=True)
