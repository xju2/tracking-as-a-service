from __future__ import annotations

import operator
from functools import partial
from pathlib import Path

import frnn
import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import RemoveIsolatedNodes

import fastwalkthrough as walkutils

torch.manual_seed(42)


def build_edges(
    query: torch.Tensor,
    database: torch.Tensor,
    r_max: float = 0.1,
    k_max: int = 1000,
) -> torch.Tensor:
    # Compute edges
    _, idxs, _, _ = frnn.frnn_grid_points(
        points1=query.unsqueeze(0),
        points2=database.unsqueeze(0),
        lengths1=None,
        lengths2=None,
        K=k_max,
        r=r_max,
        grid=None,
        return_nn=False,
        return_sorted=True,
    )

    idxs: torch.Tensor = idxs.squeeze().int()
    ind = torch.arange(idxs.shape[0], device=query.device).repeat(idxs.shape[1], 1).T.int()
    positive_idxs = idxs >= 0
    edge_list = torch.stack([ind[positive_idxs], idxs[positive_idxs]]).long()

    # Remove self-loops
    edge_list = edge_list[:, edge_list[0] != edge_list[1]]
    return edge_list


def calc_eta(r, z):
    theta = torch.arctan2(r, z)
    return -1.0 * torch.log(torch.tan(theta / 2.0))


class MetricLearningInference:
    def __init__(
        self,
        model_path: str,
        r_max: float = 0.10,
        k_max: int = 1000,
        filter_cut: float = 0.2,
        filter_batches: int = 10,
        cc_cut: float = 0.01,
        walk_min: float = 0.1,
        walk_max: float = 0.6,
        device: str = "cuda",
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
        self.debug = debug
        self.node_feature_names = [x.strip() for x in node_feature_names.split(",")]
        self.node_feature_scales = [float(x.strip()) for x in node_feature_scales.split(",")]
        self.r_index = self.node_feature_names.index("r")
        self.z_index = self.node_feature_names.index("z")

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

        track_candidates = np.array([-1], dtype=np.int64)
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

        # order edges by their distances from collision point,
        # and remove duplicated edges.
        R = node_features[:, self.r_index] ** 2 + node_features[:, self.z_index] ** 2
        edge_flip_mask = R[edge_index[0]] > R[edge_index[1]]
        edge_index[:, edge_flip_mask] = edge_index[:, edge_flip_mask].flip(0)
        edge_index = torch.unique(edge_index, dim=-1)

        if self.debug:
            print(f"after removing duplications: {edge_index.shape[1]:,}")

        # Filtering
        with torch.no_grad():
            edge_score = [
                self.filter_model(node_features, subset).squeeze(-1)
                for subset in torch.tensor_split(edge_index, self.filter_batches, dim=1)
            ]

        edge_score = torch.cat(edge_score).sigmoid()
        edge_index = edge_index[:, edge_score >= self.filter_cut]
        if edge_index.shape[1] < 2:
            return track_candidates

        if self.debug:
            print(f"Number of edges after filtering: {edge_index.shape[1]:,}")

        # GNN
        gnn_node_feature_names = "r, phi, z, eta, cluster_r_1, cluster_phi_1, cluster_z_1, cluster_eta_1, cluster_r_2, cluster_phi_2, cluster_z_2, cluster_eta_2"
        gnn_node_feature_names = [x.strip() for x in gnn_node_feature_names.split(",")]
        gnn_node_feature_scales = [
            1000.0,
            3.15,
            1000.0,
            1.0,
            1000.0,
            3.15,
            1000.0,
            1.0,
            1000.0,
            3.15,
            1000.0,
            1.0,
        ]
        hit_eta = calc_eta(
            node_features[:, self.r_index] * self.node_feature_scales[self.r_index],
            node_features[:, self.z_index] * self.node_feature_scales[self.z_index],
        )
        cluster_x = (
            node_features[:, self.node_feature_names.index("cluster_x_1")]
            * self.node_feature_scales[self.node_feature_names.index("cluster_x_1")]
        )
        cluster_y = (
            node_features[:, self.node_feature_names.index("cluster_y_1")]
            * self.node_feature_scales[self.node_feature_names.index("cluster_y_1")]
        )
        cluster_z = (
            node_features[:, self.node_feature_names.index("cluster_z_1")]
            * self.node_feature_scales[self.node_feature_names.index("cluster_z_1")]
        )
        cluster_r = torch.sqrt(cluster_x**2 + cluster_y**2)
        cluster_phi = torch.atan2(cluster_y, cluster_x)
        cluster_eta = calc_eta(cluster_r, cluster_z)

        gnn_node_features = torch.stack(
            [
                node_features[:, 0] * self.node_feature_scales[0] / gnn_node_feature_scales[0],
                node_features[:, 1] * self.node_feature_scales[1] / gnn_node_feature_scales[1],
                node_features[:, 2] * self.node_feature_scales[2] / gnn_node_feature_scales[2],
                hit_eta / gnn_node_feature_scales[3],
                cluster_r,
                cluster_phi,
                cluster_z,
                cluster_eta,
                cluster_r,
                cluster_phi,
                cluster_z,
                cluster_eta,
            ],
            dim=1,
        )

        with torch.no_grad():
            edge_score = self.gnn_model(gnn_node_features, edge_index)
        edge_score = edge_score.sigmoid()

        # CC and Walkthrough
        if self.debug:
            print("After GNN...")

        good_edge_mask = edge_score > self.cc_cut
        edge_index = edge_index[:, good_edge_mask]
        edge_score = edge_score[good_edge_mask]
        del good_edge_mask

        score_name = "edge_score"
        graph = Data(
            R=R,
            edge_index=edge_index,
            hit_id=hit_id,
            num_nodes=node_features.shape[0],
        ).to(self.device)
        graph[score_name] = edge_score
        graph = RemoveIsolatedNodes()(graph)

        all_trkx = {}
        all_trkx["cc"], graph = walkutils.get_simple_path(graph)
        all_trkx["walk"] = walkutils.walk_through(
            graph, score_name, self.walk_min, self.walk_max, self.cc_cut
        )
        if self.debug:
            print(f"Number of tracks found by CC: {len(all_trkx['cc'])}")
            print(f"Number of tracks found by Walkthrough: {len(all_trkx['walk'])}")

        tracks = all_trkx["cc"] + list(all_trkx["walk"])
        track_candidates = np.array([item for track in tracks for item in [*track, -1]])
        return track_candidates

    def __call__(self, node_features: torch.Tensor, hit_id: torch.Tensor | None = None):
        return self.forward(node_features, hit_id)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inference for Metric Learning")
    add_arg = parser.add_argument("-i", "--input", type=str, default="node_features.pt")

    args = parser.parse_args()
    inference = MetricLearningInference(model_path="./", debug=True)
    node_features = torch.load(args.input).float()
    track_ids = inference(node_features)
    print(track_ids)
