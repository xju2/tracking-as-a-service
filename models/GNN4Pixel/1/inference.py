from __future__ import annotations

from pathlib import Path

from taas.graph_construction.metric_learning import MetricLearningGraphConstructor

build_edges = MetricLearningGraphConstructor.build_edges
import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

from taas.graph_segmentation.networkx_walkthrough import get_simple_path, get_tracks



class MetricLearningInference:
    def __init__(
        self,
        model_path: str,
        r_max: float = 0.1,
        k_max: int = 300,
        filter_cut: float = 0.2,
        filter_batches: int = 10,
        cc_cut: float = 0.01,
        walk_min: float = 0.1,
        walk_max: float = 0.6,
        device: str = "cuda",
        r_index: int = 0,
        z_index: int = 2,
        debug: bool = False,
        node_feature_names: str = "r,phi,z,cluster_x_1,cluster_y_1,cluster_z_1,charge_count_1,count_1,loc_eta_1,loc_phi_1,glob_eta_1,glob_phi_1,localDir0_1,localDir1_1,localDir2_1",  # noqa: E501
        node_feature_scales: str = "300, 3.15, 3000, 300, 300, 3000, 30, 20, -2, 1.5, 2, 3.15, 1, 1, 1",  # noqa: E501
    ):
        model_path = Path(model_path) if isinstance(model_path, str) else model_path
        self.r_max = r_max
        self.k_max = k_max
        self.filter_cut = filter_cut
        self.filter_batches = filter_batches
        self.cc_cut = cc_cut
        self.walk_min = walk_min
        self.walk_max = walk_max
        self.device = device
        self.r_index = r_index
        self.z_index = z_index
        self.debug = debug
        self.node_feature_names = [x.strip() for x in node_feature_names.split(",")]
        self.node_feature_scales = [float(x.strip()) for x in node_feature_scales.split(",")]

        embedding_path = model_path / "embedding.pt"
        filtering_path = model_path / "filter.pt"
        gnn_path = model_path / "gnn.pt"

        self.embedding_model = torch.jit.load(embedding_path).to(device)
        self.filter_model = torch.jit.load(filtering_path).to(device)
        self.gnn_model = torch.jit.load(gnn_path).to(device)

    def forward(self, node_features: torch.Tensor, hit_id: torch.Tensor | None = None):
        node_features = node_features.to(self.device)
        assert node_features.shape[1] == len(self.node_feature_names)

        node_features /= torch.tensor(self.node_feature_scales, device=self.device)

        if hit_id is None:
            hit_id = torch.arange(node_features.shape[0], device=self.device)

        # Metric Learning
        with torch.no_grad():
            embedding = self.embedding_model(node_features)

        if self.debug:
            print(f"after embedding, shape = {embedding.shape}")

        # Build edges
        edge_index = build_edges(embedding, embedding, r_max=self.r_max, k_max=self.k_max)
        del embedding

        if self.debug:
            print(f"Number of edges after embedding: {edge_index.shape[1]:,}")

        # Filtering
        with torch.no_grad():
            edge_score = [
                self.filter_model(node_features, subset).squeeze(-1)
                for subset in torch.tensor_split(edge_index, self.filter_batches, dim=1)
            ]

        edge_score = torch.cat(edge_score).sigmoid()
        edge_index = edge_index[:, edge_score > self.filter_cut]
        if self.debug:
            print(f"Number of edges after filtering: {edge_index.shape[1]:,}")

        # GNN
        # rearrange the edges by their distances from collision point.
        R = node_features[:, self.r_index] ** 2 + node_features[:, self.z_index] ** 2
        edge_flip_mask = R[edge_index[0]] > R[edge_index[1]]
        edge_index[:, edge_flip_mask] = edge_index[:, edge_flip_mask].flip(0)

        # apply GNN
        with torch.no_grad():
            edge_score = self.gnn_model(node_features, edge_index)
        edge_score = edge_score.sigmoid()

        # CC and Walkthrough
        if self.debug:
            print("After GNN...")

        score_name = "edge_score"
        graph = Data(
            x=node_features,
            edge_index=edge_index,
            hit_id=hit_id,
            edge_score=edge_score,
        )
        G = to_networkx(graph, ["hit_id"], [score_name], to_undirected=False)

        # Remove edges below threshold
        list_fake_edges = [(u, v) for u, v, e in G.edges(data=True) if e[score_name] <= self.cc_cut]
        G.remove_edges_from(list_fake_edges)

        G.remove_nodes_from(list(nx.isolates(G)))
        # print(f"After removing fake edges; Number of nodes
        # = {G.number_of_nodes():,}, number of edges = {G.number_of_edges():,}")

        all_trkx = {}
        all_trkx["cc"], G = get_simple_path(G)
        all_trkx["walk"] = get_tracks(G, self.walk_min, self.walk_max, score_name)
        if self.debug:
            print(f"Number of tracks found by CC: {len(all_trkx['cc'])}")
            print(f"Number of tracks found by Walkthrough: {len(all_trkx['walk'])}")

        # label each hits.
        track_candidates = np.array([
            item for track in all_trkx["cc"] + all_trkx["walk"] for item in [*track, -1]
        ])
        return track_candidates

    def __call__(self, node_features: torch.Tensor, hit_id: torch.Tensor | None = None):
        return self.forward(node_features, hit_id)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inference for Metric Learning")
    add_arg = parser.add_argument("-i", "--input", type=str, default="node_features.pt")

    args = parser.parse_args()
    inference = MetricLearningInference(model_path="./", r_max=0.11, debug=True)
    node_features = torch.load(args.input)
    track_ids = inference(node_features)
    print(track_ids)
