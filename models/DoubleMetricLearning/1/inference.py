from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
import sys

from taas.graph_construction.double_metric_learning import DoubleMetricLearningGraphConstructor

build_edges = DoubleMetricLearningGraphConstructor.build_edges
import numpy as np
import torch
import torch.cuda.nvtx as nvtx
from taas.torch_models.double_metric_learning import DoubleMetricLearning
from taas.torch_models.interaction_gnn import (
    ChainedInteractionGNN2,
    GNNFilterJitable,
    RecurrentInteractionGNN2,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from taas.graph_segmentation import fastwalkthrough as walkutils
from taas.graph_segmentation.acts_junction_removal import acts_junction_removal
from taas.graph_segmentation.dwalk import dwalk, get_simple_path as get_dwalk_simple_path
from taas.graph_segmentation.edge_layer_connector import edge_layer_connector
from taas.graph_segmentation.junction_removal import junction_removal
from torch_geometric.data import Data
from torch_geometric.transforms import RemoveIsolatedNodes
from functools import partial
from taas.edge_classification.filter import FilterProfile, run_gnn_filter as _run_gnn_filter, run_torch_model as _run_torch_model

run_gnn_filter = partial(_run_gnn_filter, profile=FilterProfile.DOUBLE_METRIC_CURRENT)
run_torch_model = partial(_run_torch_model, profile=FilterProfile.DOUBLE_METRIC_CURRENT)

torch.manual_seed(42)
torch.set_float32_matmul_precision("high")


def to_trk_tensor(trk, device):
    if isinstance(trk, torch.Tensor):
        return trk.to(device=device, dtype=torch.long)

    # Convert numba.typed.List or Python list -> numpy
    if not isinstance(trk, np.ndarray):
        trk = np.array(trk, dtype=np.int64)
    else:
        trk = trk.astype(np.int64, copy=False)

    # Finally -> torch tensor on correct device
    return torch.as_tensor(trk, dtype=torch.long, device=device)


def synchronize_if_cuda(device) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def iter_track_tensors(tracks):
    for track in tracks:
        if hasattr(track, "grouped_hit_ids") and hasattr(track, "counts"):
            yield from track
        else:
            yield track



@dataclass
class MetricLearningInferenceConfig:
    model_path: str | Path
    device: str
    auto_cast: bool
    compiling: bool
    debug: bool
    graph_segmentation_method: str = "fastwalk"
    use_dwalk: bool = False
    dwalk_use_gpu: bool = True
    dwalk_cc_backend: str = "custom_cuda"
    dwalk_path_metrics: str = "score_weighted_length"
    jr_use_gpu: bool = True
    jr_cc_backend: str = "custom_cuda"
    jr_score_cut: float = 0.9
    jr_junction_cut: float = 0.2
    jr_min_chain_length: int = 5
    jr_skip_remove_cycles: bool = True
    acts_jr_use_gpu: bool = True
    acts_jr_cc_backend: str = "custom_cuda"
    acts_jr_min_candidate_size: int = 3
    elc_weights_cut: float = 0.01
    elc_min_hits: int = 3
    elc_block_size: int = 512
    elc_max_hits_per_track: int = 30
    save_debug_data: bool = False
    r_max: float = 0.14
    k_max: int = 1000
    filter_cut: float = 0.01
    filter_batches: int = 10
    collect_timing: bool = False
    filter_timing_skip_events: int = 0
    cc_cut: float | None = None
    walk_min: float = 0.1
    walk_max: float = 0.6
    embedding_node_features: str = "r, phi, z, cluster_x_1, cluster_y_1, cluster_z_1, cluster_x_2, cluster_y_2, cluster_z_2, count_1, charge_count_1, loc_eta_1, loc_phi_1, localDir0_1, localDir1_1, localDir2_1, lengthDir0_1, lengthDir1_1, lengthDir2_1, glob_eta_1, glob_phi_1, eta_angle_1, phi_angle_1, count_2, charge_count_2, loc_eta_2, loc_phi_2, localDir0_2, localDir1_2, localDir2_2, lengthDir0_2, lengthDir1_2, lengthDir2_2, glob_eta_2, glob_phi_2, eta_angle_2, phi_angle_2"
    embedding_node_scale: str = "1000, 3.14, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1, 1, 3.14, 3.14, 1, 1, 1, 1, 1, 1, 3.14, 3.14, 3.14, 3.14, 1, 1, 3.14, 3.14, 1, 1, 1, 1, 1, 1, 3.14, 3.14, 3.14, 3.14"
    filter_node_features: str = "r, phi, z, cluster_x_1, cluster_y_1, cluster_z_1, cluster_x_2, cluster_y_2, cluster_z_2, count_1, charge_count_1, loc_eta_1, loc_phi_1, localDir0_1, localDir1_1, localDir2_1, lengthDir0_1, lengthDir1_1, lengthDir2_1, glob_eta_1, glob_phi_1, eta_angle_1, phi_angle_1, count_2, charge_count_2, loc_eta_2, loc_phi_2, localDir0_2, localDir1_2, localDir2_2, lengthDir0_2, lengthDir1_2, lengthDir2_2, glob_eta_2, glob_phi_2, eta_angle_2, phi_angle_2"
    filter_node_scale: str = "1000, 3.14, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1, 1, 3.14, 3.14, 1, 1, 1, 1, 1, 1, 3.14, 3.14, 3.14, 3.14, 1, 1, 3.14, 3.14, 1, 1, 1, 1, 1, 1, 3.14, 3.14, 3.14, 3.14"
    gnn_node_features: str = "r, phi, z, eta, cluster_r_1, cluster_phi_1, cluster_z_1, cluster_eta_1, cluster_r_2, cluster_phi_2, cluster_z_2, cluster_eta_2"
    gnn_node_scale: str = "1000.0, 3.14159265359, 1000.0, 1.0, 1000.0, 3.14159265359, 1000.0, 1.0, 1000.0, 3.14159265359, 1000.0, 1.0"

    def __post_init__(self):
        self.embedding_node_features = [x.strip() for x in self.embedding_node_features.split(",")]
        self.embedding_node_scale = [
            float(x.strip()) for x in self.embedding_node_scale.split(",")
        ]
        assert len(self.embedding_node_features) == len(self.embedding_node_scale)

        self.filter_node_features = [x.strip() for x in self.filter_node_features.split(",")]
        self.filter_node_scale = [float(x.strip()) for x in self.filter_node_scale.split(",")]
        assert len(self.filter_node_features) == len(self.filter_node_scale)

        self.gnn_node_features = [x.strip() for x in self.gnn_node_features.split(",")]
        self.gnn_node_scale = [float(x.strip()) for x in self.gnn_node_scale.split(",")]
        assert len(self.gnn_node_features) == len(self.gnn_node_scale)

        self.model_path = (
            Path(self.model_path) if isinstance(self.model_path, str) else self.model_path
        )
        self.graph_segmentation_method = self.graph_segmentation_method.lower()
        if self.graph_segmentation_method == "fastwalk" and self.use_dwalk:
            self.graph_segmentation_method = "dwalk"
        if self.cc_cut is None:
            self.cc_cut = (
                0.5
                if self.graph_segmentation_method
                in {"acts_junction_removal", "acts_jr"}
                else 0.01
            )


class MetricLearningInference:
    def __init__(self, config: MetricLearningInferenceConfig):
        self.config = config
        self.last_timing: dict[str, float] = {}
        self._timed_event_count = 0
        self._skip_timing_this_event = False
        print(self.config)
        model_path = (
            Path(self.config.model_path)
            if not isinstance(self.config.model_path, Path)
            else self.config.model_path
        )

        # Load checkpoints instead of .pt files
        # Find checkpoint directory
        embedding_path = model_path / "embedding.ckpt"
        filtering_path = model_path / "filter.ckpt"
        gnn_path = model_path / "gnn.ckpt"

        # load the checkpoint

        print(f"Loading checkpoint from {embedding_path}")
        checkpoint = torch.load(embedding_path, map_location="cpu")
        self.embedding_model = DoubleMetricLearning(checkpoint["hyper_parameters"])
        # print("JAYYYY", checkpoint["state_dict"].keys())
        keys = list(checkpoint["state_dict"].keys())
        for key in keys:
            if "src_network" in key or "tgt_network" in key:
                checkpoint["state_dict"].pop(key)
        self.embedding_model.load_state_dict(checkpoint["state_dict"])
        self.embedding_model.to(self.config.device).eval()

        print(f"Loading checkpoint from {filtering_path}")
        checkpoint = torch.load(filtering_path, map_location="cpu")
        new_model = GNNFilterJitable(checkpoint["hyper_parameters"])
        new_model.load_state_dict(checkpoint["state_dict"])
        self.filter_model = new_model
        self.filter_model.to(self.config.device).eval()

        print(f"Loading checkpoint from {gnn_path}")
        checkpoint = torch.load(gnn_path, map_location="cpu")
        model_config = checkpoint["hyper_parameters"]
        state_dict = checkpoint["state_dict"]

        is_recurrent = model_config["node_net_recurrent"] and model_config["edge_net_recurrent"]
        print(f"Is a recurrent GNN?: {is_recurrent}")

        if is_recurrent:
            print("Use RecurrentInteractionGNN2")
            new_gnn = RecurrentInteractionGNN2(model_config)
        else:
            print("Use ChainedInteractionGNN2")
            new_gnn = ChainedInteractionGNN2(model_config)

        new_gnn.load_state_dict(state_dict)

        self.gnn_model = new_gnn
        self.gnn_model.to(self.config.device).eval()

        device = self.config.device
        self.embedding_scale = torch.tensor(config.embedding_node_scale, device=device)
        self.filter_scale = torch.tensor(config.filter_node_scale, device=device)
        self.gnn_scale = torch.tensor(config.gnn_node_scale, device=device)

        if self.config.compiling:
            print("compiling models works now...")
            torch.set_float32_matmul_precision("high")
            # self.embedding_model = torch._dynamo.optimize("inductor")(self.embedding_model)
            self.embedding_model = torch.compile(
                self.embedding_model, mode="reduce-overhead"
            )
            # # Compile GNNFilter
            self.filter_model.gnn = torch.compile(
                self.filter_model.gnn, mode="reduce-overhead"
            )
            self.filter_model.net = torch.compile(
                self.filter_model.net, mode="reduce-overhead"
            )
            # # Compile interaction gnn
            self.gnn_model = torch.compile(self.gnn_model, mode="reduce-overhead")
            # self.embedding_model.eval()
            # self.filter_model.eval()
            # self.gnn_model.eval()

        self.input_node_features = [
            "r",
            "phi",
            "z",
            "cluster_x_1",
            "cluster_y_1",
            "cluster_z_1",
            "cluster_x_2",
            "cluster_y_2",
            "cluster_z_2",
            "count_1",
            "charge_count_1",
            "loc_eta_1",
            "loc_phi_1",
            "localDir0_1",
            "localDir1_1",
            "localDir2_1",
            "lengthDir0_1",
            "lengthDir1_1",
            "lengthDir2_1",
            "glob_eta_1",
            "glob_phi_1",
            "eta_angle_1",
            "phi_angle_1",
            "count_2",
            "charge_count_2",
            "loc_eta_2",
            "loc_phi_2",
            "localDir0_2",
            "localDir1_2",
            "localDir2_2",
            "lengthDir0_2",
            "lengthDir1_2",
            "lengthDir2_2",
            "glob_eta_2",
            "glob_phi_2",
            "eta_angle_2",
            "phi_angle_2",
            "eta",
            "cluster_r_1",
            "cluster_phi_1",
            "cluster_eta_1",
            "cluster_r_2",
            "cluster_phi_2",
            "cluster_eta_2",
        ]

    def _record_timing(self, timing_key: str, duration_seconds: float) -> None:
        if self._skip_timing_this_event:
            return
        self.last_timing[timing_key] = duration_seconds

    def forward(
        self,
        node_features: torch.Tensor,
        hit_id: torch.Tensor | None = None,
        nvtx_enabled: bool = False,
    ):
        self.last_timing.clear()
        if self.config.collect_timing:
            self._timed_event_count += 1
            self._skip_timing_this_event = (
                self._timed_event_count <= self.config.filter_timing_skip_events
            )
        else:
            self._skip_timing_this_event = False
        device = self.config.device
        use_nvtx = nvtx_enabled and torch.cuda.is_available() and str(device).startswith("cuda")

        track_candidates = np.array([-1], dtype=np.int64)
        if node_features is None or node_features.shape[0] < 3:
            return track_candidates

        node_features = node_features.to(device).float()
        node_features = torch.nan_to_num(node_features, nan=0.0, posinf=0.0, neginf=0.0)

        if hit_id is None:
            hit_id = torch.arange(node_features.shape[0], device=device)

        if use_nvtx:
            nvtx.range_push("ML Inference one event")

        if use_nvtx:
            nvtx.range_push("Metric Learning Inference")

        embedding_inputs = node_features[
            :, [self.input_node_features.index(x) for x in self.config.embedding_node_features]
        ]
        embedding_inputs /= self.embedding_scale

        src_embedding, tgt_embedding = run_torch_model(
            self.embedding_model, self.config.auto_cast, embedding_inputs
        )

        if use_nvtx:
            torch.cuda.synchronize()
            nvtx.range_pop()

        # delete the embedding inputs if not needed.
        if self.config.filter_node_features == self.config.embedding_node_features:
            filtering_inputs = embedding_inputs
        else:
            del embedding_inputs
            filtering_inputs = node_features[
                :, [self.input_node_features.index(x) for x in self.config.filter_node_features]
            ]
            filtering_inputs /= self.filter_scale

        if use_nvtx:
            nvtx.range_push("Build Edges")

        edge_index = build_edges(
            src_embedding, tgt_embedding, r_max=self.config.r_max, k_max=self.config.k_max
        )

        if use_nvtx:
            torch.cuda.synchronize()
            nvtx.range_pop()

        if edge_index.shape[1] < 2:
            if use_nvtx:
                nvtx.range_pop()
            return track_candidates

        # order the edges by their distance from the collision point.
        R = (
            node_features[:, self.input_node_features.index("r")] ** 2
            + node_features[:, self.input_node_features.index("z")] ** 2
        )
        edge_flip_mask = (R[edge_index[0]] > R[edge_index[1]]) | (
            (R[edge_index[0]] == R[edge_index[1]]) & (edge_index[0] > edge_index[1])
        )
        edge_index[:, edge_flip_mask] = edge_index[:, edge_flip_mask].flip(0)
        edge_index = torch.unique(edge_index, dim=-1)

        if use_nvtx:
            nvtx.range_push("GNN Filtering")

        filter_start_time = time.perf_counter() if self.config.collect_timing else None
        edge_scores, edge_index, _ = run_gnn_filter(
            model=self.filter_model,
            auto_cast=self.config.auto_cast,
            batches=self.config.filter_batches,
            x=filtering_inputs,
            edge_index=edge_index,
        )
        if self.config.collect_timing:
            if str(device).startswith("cuda") and torch.cuda.is_available():
                torch.cuda.synchronize()
            self._record_timing("filtering_seconds", time.perf_counter() - filter_start_time)

        if use_nvtx:
            torch.cuda.synchronize()
            nvtx.range_pop()

        # apply fitlering score cuts.
        edge_index = edge_index[:, edge_scores >= self.config.filter_cut]
        # del edge_scores
        if edge_index.shape[1] < 2:
            if use_nvtx:
                nvtx.range_pop()
            return track_candidates

        # prepare GNN inputs
        gnn_input = node_features[
            :, [self.input_node_features.index(x) for x in self.config.gnn_node_features]
        ]
        gnn_input /= self.gnn_scale

        # calculate edge features: dr, dphi, dz, deta, phislope, rphislope
        def reset_angle(angles):
            angles[angles > torch.pi] -= 2 * torch.pi
            angles[angles < -torch.pi] += 2 * torch.pi
            return angles

        def calculate_edge_features():
            r = (
                node_features[:, self.input_node_features.index("r")]
                / self.config.gnn_node_scale[self.config.gnn_node_features.index("r")]
            )
            phi = (
                node_features[:, self.input_node_features.index("phi")]
                / self.config.gnn_node_scale[self.config.gnn_node_features.index("phi")]
            )
            z = (
                node_features[:, self.input_node_features.index("z")]
                / self.config.gnn_node_scale[self.config.gnn_node_features.index("z")]
            )
            eta = (
                node_features[:, self.input_node_features.index("eta")]
                / self.config.gnn_node_scale[self.config.gnn_node_features.index("eta")]
            )

            src, dst = edge_index
            dr = r[dst] - r[src]
            dphi = reset_angle((phi[dst] - phi[src]) * torch.pi) / torch.pi
            dz = z[dst] - z[src]
            deta = eta[dst] - eta[src]
            phislope = torch.clamp(
                torch.nan_to_num(dphi / dr, nan=0.0, posinf=100, neginf=-100), -100, 100
            )
            r_avg = (r[dst] + r[src]) / 2.0
            rphislope = torch.nan_to_num(torch.multiply(r_avg, phislope), nan=0.0)
            return torch.stack([dr, dphi, dz, deta, phislope, rphislope], dim=1)

        edge_features = calculate_edge_features().to(device).float()

        if use_nvtx:
            nvtx.range_push("GNN Inference")

        edge_scores = (
            run_torch_model(
                self.gnn_model, self.config.auto_cast, gnn_input, edge_index, edge_features
            )
            .sigmoid()
            .to(torch.float32)
        )

        if use_nvtx:
            torch.cuda.synchronize()
            nvtx.range_pop()

        segmentation_method = self.config.graph_segmentation_method
        if segmentation_method not in {
            "junction_removal",
            "jr",
            "edge_layer_connector",
            "elc",
        }:
            good_edge_mask = edge_scores > self.config.cc_cut
            edge_index = edge_index[:, good_edge_mask]
            edge_scores = edge_scores[good_edge_mask]
            del good_edge_mask

        score_name = "edge_scores"
        graph = Data(
            R=R,
            edge_index=edge_index,
            hit_id=hit_id,
            num_nodes=node_features.shape[0],
        ).to(device)
        graph[score_name] = edge_scores
        graph = RemoveIsolatedNodes()(graph)

        all_trkx = {}
        if segmentation_method == "dwalk":
            if self.config.collect_timing:
                synchronize_if_cuda(device)
                segmentation_start_time = time.perf_counter()
            else:
                segmentation_start_time = None
            all_trkx["cc"], dwalk_graph = get_dwalk_simple_path(
                graph,
                score_name,
                self.config.cc_cut,
                use_gpu=self.config.dwalk_use_gpu,
                cc_backend=self.config.dwalk_cc_backend,
                return_tensors=True,
            )
            all_trkx["walk"] = dwalk(
                dwalk_graph,
                score_name,
                self.config.walk_min,
                self.config.walk_max,
                self.config.dwalk_path_metrics,
                use_gpu=self.config.dwalk_use_gpu,
                cc_backend=self.config.dwalk_cc_backend,
                return_tensors=True,
            )
            if self.config.collect_timing:
                synchronize_if_cuda(device)
                self._record_timing(
                    "graph_segmentation_seconds",
                    time.perf_counter() - segmentation_start_time,
                )
        elif segmentation_method in {"junction_removal", "jr"}:
            if self.config.collect_timing:
                synchronize_if_cuda(device)
                segmentation_start_time = time.perf_counter()
            else:
                segmentation_start_time = None
            all_trkx["cc"] = []
            all_trkx["walk"] = junction_removal(
                graph,
                score_name,
                score_cut=self.config.jr_score_cut,
                junction_cut=self.config.jr_junction_cut,
                min_chain_length=self.config.jr_min_chain_length,
                skip_remove_cycles=self.config.jr_skip_remove_cycles,
                use_gpu=self.config.jr_use_gpu,
                cc_backend=self.config.jr_cc_backend,
            )
            if self.config.collect_timing:
                synchronize_if_cuda(device)
                self._record_timing(
                    "graph_segmentation_seconds",
                    time.perf_counter() - segmentation_start_time,
                )
        elif segmentation_method in {"acts_junction_removal", "acts_jr"}:
            if self.config.collect_timing:
                synchronize_if_cuda(device)
                segmentation_start_time = time.perf_counter()
            else:
                segmentation_start_time = None
            all_trkx["cc"] = []
            all_trkx["walk"] = acts_junction_removal(
                graph,
                score_name,
                min_candidate_size=self.config.acts_jr_min_candidate_size,
                use_gpu=self.config.acts_jr_use_gpu,
                cc_backend=self.config.acts_jr_cc_backend,
            )
            if self.config.collect_timing:
                synchronize_if_cuda(device)
                self._record_timing(
                    "graph_segmentation_seconds",
                    time.perf_counter() - segmentation_start_time,
                )
        elif segmentation_method in {"edge_layer_connector", "elc"}:
            if self.config.collect_timing:
                synchronize_if_cuda(device)
                segmentation_start_time = time.perf_counter()
            else:
                segmentation_start_time = None
            all_trkx["cc"] = []
            all_trkx["walk"] = edge_layer_connector(
                graph,
                score_name,
                weights_cut=self.config.elc_weights_cut,
                min_hits=self.config.elc_min_hits,
                block_size=self.config.elc_block_size,
                max_hits_per_track=self.config.elc_max_hits_per_track,
            )
            if self.config.collect_timing:
                synchronize_if_cuda(device)
                self._record_timing(
                    "graph_segmentation_seconds",
                    time.perf_counter() - segmentation_start_time,
                )
        else:
            if self.config.collect_timing:
                synchronize_if_cuda(device)
                segmentation_start_time = time.perf_counter()
            else:
                segmentation_start_time = None
            all_trkx["cc"], graph = walkutils.get_simple_path(graph)
            all_trkx["walk"] = walkutils.walk_through(
                graph, score_name, self.config.walk_min, self.config.walk_max, False
            )
            if self.config.collect_timing:
                synchronize_if_cuda(device)
                self._record_timing(
                    "graph_segmentation_seconds",
                    time.perf_counter() - segmentation_start_time,
                )

        if use_nvtx:
            torch.cuda.synchronize()
            nvtx.range_pop()

        tracks = list(iter_track_tensors(all_trkx["cc"])) + list(
            iter_track_tensors(all_trkx["walk"])
        )
        total_length = sum(len(trk) + 1 for trk in tracks)
        track_candidates = np.empty(total_length, dtype=int)

        i = 0
        track_r = R
        for trk in tracks:
            trk_tensor = to_trk_tensor(trk, device)
            sorted_trk = trk_tensor[torch.argsort(track_r[trk_tensor])]

            n = len(sorted_trk)
            track_candidates[i : i + n] = sorted_trk.cpu().tolist()
            i += n
            track_candidates[i] = -1
            i += 1

        return track_candidates

    def __call__(
        self,
        node_features: torch.Tensor,
        hit_id: torch.Tensor | None = None,
        nvtx_enabled: bool = False,
    ):
        return self.forward(node_features, hit_id=hit_id, nvtx_enabled=nvtx_enabled)


def create_metric_learning_end2end_rel24(
    model_path: str,
    device: str,
    debug: bool,
    precision: str,
    auto_cast: bool,
    compiling: bool,
    use_dwalk: bool,
    save_data_for_debug: bool,
    graph_segmentation_method: str = "fastwalk",
    dwalk_use_gpu: bool = True,
    dwalk_cc_backend: str = "custom_cuda",
    dwalk_path_metrics: str = "score_weighted_length",
    jr_use_gpu: bool = True,
    jr_cc_backend: str = "custom_cuda",
    jr_score_cut: float = 0.9,
    jr_junction_cut: float = 0.2,
    jr_min_chain_length: int = 5,
    jr_skip_remove_cycles: bool = True,
    acts_jr_use_gpu: bool = True,
    acts_jr_cc_backend: str = "custom_cuda",
    acts_jr_min_candidate_size: int = 3,
    elc_weights_cut: float = 0.01,
    elc_min_hits: int = 3,
    elc_block_size: int = 512,
    elc_max_hits_per_track: int = 30,
    cc_cut: float | None = None,
    filter_timing_skip_events: int = 0,
):
    config = MetricLearningInferenceConfig(
        model_path=model_path,
        device=device,
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
        debug=debug,
        save_debug_data=save_data_for_debug,
        r_max=0.14,
        k_max=1000,
        filter_cut=0.01,
        filter_batches=1,
        filter_timing_skip_events=max(filter_timing_skip_events, 0),
        cc_cut=cc_cut,
        walk_min=0.1,
        walk_max=0.6,
        embedding_node_features="r, phi, z, cluster_x_1, cluster_y_1, cluster_z_1, cluster_x_2, cluster_y_2, cluster_z_2, count_1, charge_count_1, loc_eta_1, loc_phi_1, localDir0_1, localDir1_1, localDir2_1, lengthDir0_1, lengthDir1_1, lengthDir2_1, glob_eta_1, glob_phi_1, eta_angle_1, phi_angle_1, count_2, charge_count_2, loc_eta_2, loc_phi_2, localDir0_2, localDir1_2, localDir2_2, lengthDir0_2, lengthDir1_2, lengthDir2_2, glob_eta_2, glob_phi_2, eta_angle_2, phi_angle_2",
        embedding_node_scale="1000, 3.14, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1, 1, 3.14, 3.14, 1, 1, 1, 1, 1, 1, 3.14, 3.14, 3.14, 3.14, 1, 1, 3.14, 3.14, 1, 1, 1, 1, 1, 1, 3.14, 3.14, 3.14, 3.14",
        filter_node_features="r, phi, z, cluster_x_1, cluster_y_1, cluster_z_1, cluster_x_2, cluster_y_2, cluster_z_2, count_1, charge_count_1, loc_eta_1, loc_phi_1, localDir0_1, localDir1_1, localDir2_1, lengthDir0_1, lengthDir1_1, lengthDir2_1, glob_eta_1, glob_phi_1, eta_angle_1, phi_angle_1, count_2, charge_count_2, loc_eta_2, loc_phi_2, localDir0_2, localDir1_2, localDir2_2, lengthDir0_2, lengthDir1_2, lengthDir2_2, glob_eta_2, glob_phi_2, eta_angle_2, phi_angle_2",
        filter_node_scale="1000, 3.14, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1, 1, 3.14, 3.14, 1, 1, 1, 1, 1, 1, 3.14, 3.14, 3.14, 3.14, 1, 1, 3.14, 3.14, 1, 1, 1, 1, 1, 1, 3.14, 3.14, 3.14, 3.14",
        gnn_node_features="r, phi, z, eta, cluster_r_1, cluster_phi_1, cluster_z_1, cluster_eta_1, cluster_r_2, cluster_phi_2, cluster_z_2, cluster_eta_2",
        gnn_node_scale="1000.0, 3.14159265359, 1000.0, 1.0, 1000.0, 3.14159265359, 1000.0, 1.0, 1000.0, 3.14159265359, 1000.0, 1.0",
    )

    torch.set_float32_matmul_precision(precision)
    model = MetricLearningInference(config)
    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inference for Metric Learning")
    parser.add_argument("-i", "--input", type=str, default="node_features.pt", help="Input file")
    parser.add_argument("-m", "--model", type=str, default=".", help="Model path")
    parser.add_argument("-p", "--precision", type=str, default="highest", help="Precision")
    parser.add_argument("-a", "--auto_cast", action="store_true", help="Use autocast")
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug mode")
    parser.add_argument("-d", "--device", default="cuda", help="Device")
    parser.add_argument("-t", "--timing", action="store_true", help="Time the inference")
    parser.add_argument("-c", "--compiling", action="store_true", help="Use compiling")
    parser.add_argument(
        "--graph-segmentation-method",
        default="fastwalk",
        choices=(
            "fastwalk",
            "dwalk",
            "junction_removal",
            "jr",
            "acts_junction_removal",
            "acts_jr",
            "edge_layer_connector",
            "elc",
        ),
        help="Graph segmentation method",
    )
    parser.add_argument("--use-dwalk", action="store_true", help="Use D-WALK")
    parser.add_argument(
        "--dwalk-use-gpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run D-WALK on CUDA when available",
    )
    parser.add_argument(
        "--dwalk-cc-backend",
        dest="dwalk_cc_backend",
        default="custom_cuda",
        choices=("auto", "scipy", "custom_cuda"),
        help="Connected-components backend for D-WALK",
    )
    parser.add_argument(
        "--dwalk-path-metrics",
        dest="dwalk_path_metrics",
        default="score_weighted_length",
        choices=("score_weighted_length", "length"),
        help="Path metric for D-WALK",
    )
    parser.add_argument(
        "--jr-use-gpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run Junction Removal CC on CUDA when available",
    )
    parser.add_argument(
        "--jr-cc-backend",
        default="custom_cuda",
        choices=("auto", "scipy", "custom_cuda"),
        help="Connected-components backend for Junction Removal",
    )
    parser.add_argument("--jr-score-cut", type=float, default=0.9, help="High-score simple-chain cut for Junction Removal")
    parser.add_argument("--jr-junction-cut", type=float, default=0.2, help="Junction edge score cut")
    parser.add_argument("--jr-min-chain-length", type=int, default=5, help="Minimum simple-chain length")
    parser.add_argument("--jr-remove-cycles", action="store_true", help="Orient/remove cycles before Junction Removal")
    parser.add_argument(
        "--acts-jr-use-gpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run ACTS-style Junction Removal CC on CUDA when available",
    )
    parser.add_argument(
        "--acts-jr-cc-backend",
        default="custom_cuda",
        choices=("auto", "scipy", "custom_cuda"),
        help="Connected-components backend for ACTS-style Junction Removal",
    )
    parser.add_argument("--acts-jr-min-candidate-size", type=int, default=3, help="Minimum ACTS-style JR candidate size")
    parser.add_argument("--elc-weights-cut", type=float, default=0.01, help="EdgeLayerConnector internal edge weight cut")
    parser.add_argument("--elc-min-hits", type=int, default=3, help="EdgeLayerConnector minimum hits per track")
    parser.add_argument("--elc-block-size", type=int, default=512, help="EdgeLayerConnector CUDA block size")
    parser.add_argument("--elc-max-hits-per-track", type=int, default=30, help="EdgeLayerConnector maximum hits per track")
    parser.add_argument(
        "--cc-cut",
        type=float,
        default=None,
        help="Pre-segmentation GNN edge score cut; defaults to 0.5 for ACTS JR and 0.01 otherwise",
    )
    parser.add_argument(
        "--filter-timing-skip-events",
        type=int,
        default=0,
        help="Exclude the first N filtered events from timing averages",
    )
    parser.add_argument(
        "-s", "--save-data-for-debug", action="store_true", help="Save debugging data"
    )

    args = parser.parse_args()
    if not Path(args.model).exists():
        raise FileNotFoundError(f"Model path {args.model} does not exist.")

    inference = create_metric_learning_end2end_rel24(
        model_path=args.model,
        device=args.device,
        debug=args.verbose,
        precision=args.precision,
        auto_cast=args.auto_cast,
        compiling=args.compiling,
        use_dwalk=args.use_dwalk,
        save_data_for_debug=args.save_data_for_debug,
        graph_segmentation_method=args.graph_segmentation_method,
        dwalk_use_gpu=args.dwalk_use_gpu,
        dwalk_cc_backend=args.dwalk_cc_backend,
        dwalk_path_metrics=args.dwalk_path_metrics,
        jr_use_gpu=args.jr_use_gpu,
        jr_cc_backend=args.jr_cc_backend,
        jr_score_cut=args.jr_score_cut,
        jr_junction_cut=args.jr_junction_cut,
        jr_min_chain_length=args.jr_min_chain_length,
        jr_skip_remove_cycles=not args.jr_remove_cycles,
        acts_jr_use_gpu=args.acts_jr_use_gpu,
        acts_jr_cc_backend=args.acts_jr_cc_backend,
        acts_jr_min_candidate_size=args.acts_jr_min_candidate_size,
        elc_weights_cut=args.elc_weights_cut,
        elc_min_hits=args.elc_min_hits,
        elc_block_size=args.elc_block_size,
        elc_max_hits_per_track=args.elc_max_hits_per_track,
        cc_cut=args.cc_cut,
        filter_timing_skip_events=args.filter_timing_skip_events,
    )
    print("start a warm-up run.")
    node_features = torch.load(args.input)
    track_ids = inference(node_features, nvtx_enabled=False)

    # time the inference function.
    if args.timing:
        import time

        from tqdm import tqdm

        start_time = time.time()
        num_trials = 10
        print(">>> Starting NSYS-captured inference")
        nvtx.range_push("Inference_Loop")
        for _ in range(num_trials):  # tqdm(range(num_trials)):
            track_ids = inference(node_features, nvtx_enabled=True)
            # print("time for one inference:", timed(lambda: inference(node_features))[1])
        nvtx.range_pop()
        # print("average time per event", (time.time() - start_time) / num_trials)
    print("total tracks", np.sum(track_ids == -1))
