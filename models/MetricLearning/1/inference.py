from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import frnn
import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from torch_model_inference import run_gnn_filter, run_torch_model
from walkthrough import get_simple_path, get_tracks

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


@dataclass
class MetricLearningInferenceConfig:
    model_path: str | Path
    device: str
    auto_cast: bool
    debug: bool
    r_max: float
    k_max: int
    filter_cut: float
    filter_batches: int
    cc_cut: float
    walk_min: float
    walk_max: float
    embedding_node_features: str
    embedding_node_scale: str
    filter_node_features: str
    filter_node_scale: str
    gnn_node_features: str
    gnn_node_scale: str

    def __post_init__(self):
        self.embedding_node_features = [x.strip() for x in self.embedding_node_features.split(",")]
        self.embedding_node_scale = [float(x.strip()) for x in self.embedding_node_scale.split(",")]
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


class MetricLearningInference:
    def __init__(self, config: MetricLearningInferenceConfig):
        self.config = config
        embedding_path = self.config.model_path / "embedding.pt"
        filtering_path = self.config.model_path / "filter.pt"
        gnn_path = self.config.model_path / "gnn.pt"

        self.embedding_model = torch.jit.load(embedding_path).to(self.config.device)
        self.filter_model = torch.jit.load(filtering_path).to(self.config.device)
        self.gnn_model = torch.jit.load(gnn_path).to(self.config.device)

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
            "region",
        ]

    def forward(self, node_features: torch.Tensor, hit_id: torch.Tensor | None = None):
        device = self.config.device
        debug = self.config.debug

        node_features = node_features.to(device).float()
        if hit_id is None:
            hit_id = torch.arange(node_features.shape[0], device=device)

        # Metric Learning
        embedding_inputs = node_features[
            :, [self.input_node_features.index(x) for x in self.config.embedding_node_features]
        ]
        embedding_inputs /= torch.tensor(self.config.embedding_node_scale, device=device).float()

        embedding = run_torch_model(self.embedding_model, self.config.auto_cast, embedding_inputs)
        if debug:
            print(f"after embedding, shape = {embedding.shape}")
            print("embedding data", embedding[0])
            print("embedding data type", embedding.dtype)
            out_data = Data(embedding=embedding)

        # delete the embedding inputs if not needed.
        if self.config.filter_node_features == self.config.embedding_node_features:
            filtering_inputs = embedding_inputs
        else:
            del embedding_inputs
            filtering_inputs = node_features[
                :, [self.input_node_features.index(x) for x in self.config.filter_node_features]
            ]
            filtering_inputs /= torch.tensor(self.config.filter_node_scale, device=device).float()

        # Build edges
        edge_index = build_edges(
            embedding, embedding, r_max=self.config.r_max, k_max=self.config.k_max
        )
        if edge_index.shape[1] == 0:
            return torch.full((1,), -1, dtype=torch.long, device=device)

        if debug:
            print(f"Number of edges after embedding: {edge_index.shape[1]:,}")
        else:
            del embedding

        # make it undirected and remove duplicates.
        edge_index[:, edge_index[0] > edge_index[1]] = edge_index[
            :, edge_index[0] > edge_index[1]
        ].flip(0)
        edge_index = torch.unique(edge_index, dim=-1)

        # random flip
        random_flip = torch.randint(2, (edge_index.shape[1],), dtype=torch.bool)
        edge_index[:, random_flip] = edge_index[:, random_flip].flip(0)

        if debug:
            print("after removing duplications: ", edge_index.shape, edge_index[:, 0])

        # GNNFiltering
        edge_scores, edge_index, _ = run_gnn_filter(
            self.filter_model,
            self.config.auto_cast,
            self.config.filter_batches,
            filtering_inputs,
            edge_index,
        )

        if debug:
            print("edge_score", edge_scores[:10])
            print("edge_index", edge_index[:, :10])
            out_data.edge_index = edge_index
            out_data.edge_scores = edge_scores

        # apply fitlering score cuts.
        edge_index = edge_index[:, edge_scores >= self.config.filter_cut]
        del edge_scores

        if debug:
            print(f"Number of edges after filtering: {edge_index.shape[1]:,}")

        edge_index[:, edge_index[0] > edge_index[1]] = edge_index[
            :, edge_index[0] > edge_index[1]
        ].flip(0)
        # prepare GNN inputs
        gnn_input = node_features[
            :, [self.input_node_features.index(x) for x in self.config.gnn_node_features]
        ]
        gnn_input /= torch.tensor(self.config.gnn_node_scale, device=device).float()

        # # same features on the 3 channels in the STRIP ENDCAP.
        if (
            "region" in self.input_node_features
            and self.input_node_features.index("region") < node_features.shape[1]
        ):
            hit_region = node_features[:, self.input_node_features.index("region")]
            gnn_input_mask = torch.logical_or(hit_region == 2, hit_region == 6).reshape(-1)
            gnn_input[gnn_input_mask] = torch.cat([gnn_input[gnn_input_mask, 0:4]] * 3, dim=1)

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

        edge_features = calculate_edge_features()

        edge_scores = run_torch_model(
            self.gnn_model, self.config.auto_cast, gnn_input, edge_index, edge_features
        ).sigmoid()

        # CC and Walkthrough
        if debug:
            print("After GNN...")
            out_data.gnn_scores = edge_scores
            out_data.gnn_input_edges = edge_index
            out_data.gnn_edge_features = edge_features
            out_data.gnn_node_features = gnn_input
            torch.save(out_data, "debug.pt")

        # rearrange the edges by their distances from collision point.
        R = (
            node_features[:, self.input_node_features.index("r")] ** 2
            + node_features[:, self.input_node_features.index("z")] ** 2
        )
        edge_flip_mask = R[edge_index[0]] > R[edge_index[1]]
        edge_index[:, edge_flip_mask] = edge_index[:, edge_flip_mask].flip(0)

        score_name = "edge_scores"
        graph = Data(
            x=node_features,
            edge_index=edge_index,
            hit_id=hit_id,
            edge_scores=edge_scores,
        )
        G = to_networkx(graph, ["hit_id"], [score_name], to_undirected=False)

        # Remove edges below threshold
        list_fake_edges = [
            (u, v) for u, v, e in G.edges(data=True) if e[score_name] <= self.config.cc_cut
        ]
        G.remove_edges_from(list_fake_edges)
        G.remove_nodes_from(list(nx.isolates(G)))

        all_trkx = {}
        all_trkx["cc"], G = get_simple_path(G)
        if debug:
            print("the graph information")
            with open("graph_info.txt", "w") as f:
                f.write(f"{G.nodes(data=True)}\n")
                f.write(f"{G.edges(data=True)}\n")
        all_trkx["walk"] = get_tracks(G, self.config.walk_min, self.config.walk_max, score_name)
        if debug:
            print(f"Number of tracks found by CC: {len(all_trkx['cc'])}")
            print(f"Number of tracks found by Walkthrough: {len(all_trkx['walk'])}")

        # sort each track by R.
        for trk in all_trkx["cc"] + all_trkx["walk"]:
            trk.sort(key=lambda x: R[x])

        # label each hits.
        track_candidates = np.array([
            item for track in all_trkx["cc"] + all_trkx["walk"] for item in [*track, -1]
        ])
        # write candidates to a file.
        if debug:
            out_str = [
                "".join([f"{item} " for item in track]) + "\n"
                for track in all_trkx["cc"] + all_trkx["walk"]
            ]
            with open("track_candidates.txt", "w") as f:
                f.writelines(out_str)
        return track_candidates

    def __call__(self, node_features: torch.Tensor, hit_id: torch.Tensor | None = None):
        return self.forward(node_features, hit_id)


def create_metric_learning_end2end_rel24(
    model_path: str, device: str, debug: bool, precision: str, auto_cast: bool
):
    config = MetricLearningInferenceConfig(
        model_path=model_path,
        device=device,
        auto_cast=auto_cast,
        debug=debug,
        r_max=0.12,
        k_max=1000,
        filter_cut=0.05,
        filter_batches=10,
        cc_cut=0.01,
        walk_min=0.1,
        walk_max=0.6,
        embedding_node_features="r, phi, z, cluster_x_1, cluster_y_1, cluster_z_1, cluster_x_2, cluster_y_2, cluster_z_2, count_1, charge_count_1, loc_eta_1, loc_phi_1, localDir0_1, localDir1_1, localDir2_1, lengthDir0_1, lengthDir1_1, lengthDir2_1, glob_eta_1, glob_phi_1, eta_angle_1, phi_angle_1, count_2, charge_count_2, loc_eta_2, loc_phi_2, localDir0_2, localDir1_2, localDir2_2, lengthDir0_2, lengthDir1_2, lengthDir2_2, glob_eta_2, glob_phi_2, eta_angle_2, phi_angle_2",  # noqa: E501
        embedding_node_scale="1000, 3.14, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1, 1, 3.14, 3.14, 1, 1, 1, 1, 1, 1, 3.14, 3.14, 3.14, 3.14, 1, 1, 3.14, 3.14, 1, 1, 1, 1, 1, 1, 3.14, 3.14, 3.14, 3.14",  # noqa: E501
        filter_node_features="r, phi, z, cluster_x_1, cluster_y_1, cluster_z_1, cluster_x_2, cluster_y_2, cluster_z_2, count_1, charge_count_1, loc_eta_1, loc_phi_1, localDir0_1, localDir1_1, localDir2_1, lengthDir0_1, lengthDir1_1, lengthDir2_1, glob_eta_1, glob_phi_1, eta_angle_1, phi_angle_1, count_2, charge_count_2, loc_eta_2, loc_phi_2, localDir0_2, localDir1_2, localDir2_2, lengthDir0_2, lengthDir1_2, lengthDir2_2, glob_eta_2, glob_phi_2, eta_angle_2, phi_angle_2",  # noqa: E501
        filter_node_scale="1000, 3.14, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1, 1, 3.14, 3.14, 1, 1, 1, 1, 1, 1, 3.14, 3.14, 3.14, 3.14, 1, 1, 3.14, 3.14, 1, 1, 1, 1, 1, 1, 3.14, 3.14, 3.14, 3.14",  # noqa: E501
        gnn_node_features="r, phi, z, eta, cluster_r_1, cluster_phi_1, cluster_z_1, cluster_eta_1, cluster_r_2, cluster_phi_2, cluster_z_2, cluster_eta_2",  # noqa: E501
        gnn_node_scale="1000.0, 3.14159265359, 1000.0, 1.0, 1000.0, 3.14159265359, 1000.0, 1.0, 1000.0, 3.14159265359, 1000.0, 1.0",  # noqa: E501
    )

    torch.set_float32_matmul_precision(precision)
    model = MetricLearningInference(config)
    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inference for Metric Learning")
    parser.add_argument("-i", "--input", type=str, default="node_features.pt", help="Input file")
    parser.add_argument("-m", "--model", type=str, default="./", help="Model path")
    parser.add_argument("-p", "--precision", type=str, default="highest", help="Precision")
    parser.add_argument("-a", "--auto_cast", action="store_true", help="Use autocast")
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug mode")
    parser.add_argument("-d", "--device", default="cuda", help="Device")
    parser.add_argument("-t", "--timing", action="store_true", help="Time the inference")

    args = parser.parse_args()
    if not Path(args.model).exists():
        raise FileNotFoundError(f"Model path {args.model} does not exist.")

    inference = create_metric_learning_end2end_rel24(
        model_path=args.model,
        device=args.device,
        debug=args.verbose,
        precision=args.precision,
        auto_cast=args.auto_cast,
    )
    print("start a warm-up run.")
    node_features = torch.load(args.input)
    track_ids = inference(node_features)

    # time the inference function.
    if args.timing:
        import time

        from tqdm import tqdm

        start_time = time.time()
        num_trials = 4
        print("start timing...")
        for _ in tqdm(range(num_trials)):
            track_ids = inference(node_features)
        print("average time per event", (time.time() - start_time) / num_trials)
    print("total tracks", np.sum(track_ids == -1))
