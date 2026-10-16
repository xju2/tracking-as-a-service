from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data


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


@dataclass
class ModuleMapInferenceConfig:
    model_path: str | Path
    module_map_path: str
    device: str = "cuda:0"
    device_id: int = 0
    gpu_blocks: int = 512
    graph_segmentation_method: str = "edge_layer_connector"
    edge_score_cut: float = 0.6
    onnx_use_cuda: bool = True
    dwalk_score_cut: float = 0.01
    dwalk_use_gpu: bool = True
    dwalk_cc_backend: str = "custom_cuda"
    dwalk_path_metrics: str = "score_weighted_length"
    dwalk_walk_min: float = 0.1
    dwalk_walk_max: float = 0.6
    acts_jr_score_cut: float = 0.5
    acts_jr_use_gpu: bool = True
    acts_jr_cc_backend: str = "custom_cuda"
    acts_jr_min_candidate_size: int = 3
    fastwalk_score_cut: float = 0.01
    fastwalk_walk_min: float = 0.1
    fastwalk_walk_max: float = 0.6
    fastwalk_allow_node_reuse: bool = False
    elc_weights_cut: float = 0.01
    elc_min_hits: int = 3
    elc_block_size: int = 512
    elc_max_hits_per_track: int = 30

    def __post_init__(self):
        self.model_path = Path(self.model_path)
        self.graph_segmentation_method = self.graph_segmentation_method.lower()


@dataclass
class ModuleMapInferenceResult:
    labels: np.ndarray
    graph_nodes: int
    graph_edges: int
    kept_edges: int


class ModuleMapInference:
    def __init__(self, config: ModuleMapInferenceConfig):
        self.config = config
        self.device = config.device
        self.device_id = config.device_id

        from taas.cuda_ext import _edge_layer_connector_cuda
        from taas.edge_classification.interaction_gnn_onnx import InteractionGNNOnnx
        from taas.graph_construction.module_map import ModuleMapGraphConstructor
        from taas.graph_segmentation.acts_junction_removal import acts_junction_removal
        from taas.graph_segmentation.dwalk import (
            dwalk,
            get_simple_path as get_dwalk_simple_path,
        )
        from taas.graph_segmentation import fastwalkthrough

        self.acts_junction_removal = acts_junction_removal
        self.dwalk = dwalk
        self.get_dwalk_simple_path = get_dwalk_simple_path
        self.fastwalk = fastwalkthrough
        self.edge_classifier = InteractionGNNOnnx(
            config.model_path,
            device_id=config.device_id,
            use_cuda=config.onnx_use_cuda,
            apply_sigmoid=True,
        )
        self.onnx_session = self.edge_classifier.session
        self.onnx_cuda_active = self.edge_classifier.cuda_active
        self.module_map_graph = ModuleMapGraphConstructor(
            config.module_map_path,
            gpu_blocks=config.gpu_blocks,
            gpu_device=config.device_id,
        )
        self.edge_layer_connector_cuda = _edge_layer_connector_cuda

        for name, value in vars(config).items():
            if name not in {"model_path", "device", "device_id"}:
                setattr(self, name, value)

    def __call__(self, features: torch.Tensor) -> ModuleMapInferenceResult:
        return self.forward(features)

    def forward(self, features: torch.Tensor) -> ModuleMapInferenceResult:
        if features.ndim != 2 or features.shape[1] != len(INPUT_FEATURES):
            raise ValueError(
                "ModuleMap expects FEATURES with shape [N, 16] ordered as "
                f"{INPUT_FEATURES}; got {tuple(features.shape)}"
            )
        if features.shape[0] < 3:
            return self._empty_result()

        graph = self.module_map_graph.build(features)
        node_features = graph.node_features
        edge_index = graph.edge_index
        edge_features = graph.edge_features
        hit_ids = graph.hit_ids
        graph_nodes = int(node_features.shape[0])
        graph_edges = int(edge_features.shape[0])

        if edge_index.numel() == 0 or hit_ids.numel() == 0:
            return self._empty_result(graph_nodes, graph_edges)

        edge_scores = self._run_edge_classifier(node_features, edge_index, edge_features)
        if self.graph_segmentation_method in {"edge_layer_connector", "elc"}:
            keep = edge_scores > self.edge_score_cut
            edge_index = edge_index[:, keep]
            edge_scores = edge_scores[keep]
            labels = self._run_edge_layer_connector(hit_ids, edge_index, edge_scores)
            kept_edges = int(edge_scores.numel())
        elif self.graph_segmentation_method in {"dwalk", "d-walk"}:
            labels, kept_edges = self._run_dwalk(
                node_features, hit_ids, edge_index, edge_scores
            )
        elif self.graph_segmentation_method in {
            "acts_junction_removal",
            "acts-junction-removal",
            "acts_jr",
        }:
            labels, kept_edges = self._run_acts_junction_removal(
                node_features, hit_ids, edge_index, edge_scores
            )
        elif self.graph_segmentation_method in {"fastwalk", "fastwalkthrough"}:
            labels, kept_edges = self._run_fastwalkthrough(
                node_features, hit_ids, edge_index, edge_scores
            )
        else:
            raise ValueError(
                "Unsupported ModuleMap graph_segmentation_method "
                f"'{self.graph_segmentation_method}'. Expected edge_layer_connector, "
                "dwalk, acts_junction_removal, or fastwalk."
            )

        return ModuleMapInferenceResult(labels, graph_nodes, graph_edges, kept_edges)

    @staticmethod
    def _empty_result(graph_nodes: int = 0, graph_edges: int = 0):
        labels = np.array([-1], dtype=np.int64)
        return ModuleMapInferenceResult(labels, graph_nodes, graph_edges, 0)


    def _run_edge_classifier(self, node_features, edge_index, edge_features):
        return self.edge_classifier.score(node_features, edge_index, edge_features)

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
