from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from operator import itemgetter

import frnn
import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import RemoveIsolatedNodes
from torch_geometric.utils import to_networkx

from torch_model_inference import run_gnn_filter, run_torch_model
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


@dataclass
class MetricLearningInferenceConfig:
    model_path: str | Path
    device: str
    auto_cast: bool
    compling: bool
    debug: bool
    save_debug_data: bool = False
    r_max: float = 0.12
    k_max: int = 1000
    filter_cut: float = 0.05
    filter_batches: int = 10
    cc_cut: float = 0.01
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


class MetricLearningInference:
    def __init__(self, config: MetricLearningInferenceConfig):
        self.config = config
        print(self.config)
        model_path = (
            Path(self.config.model_path)
            if not isinstance(self.config.model_path, Path)
            else self.config.model_path
        )
        embedding_path = model_path / "embedding.pt"
        filtering_path = model_path / "filter.pt"
        gnn_path = model_path / "gnn.pt"

        self.embedding_model = (
            torch.jit.load(embedding_path).to(self.config.device, non_blocking=True).eval()
        )
        self.filter_model = (
            torch.jit.load(filtering_path).to(self.config.device, non_blocking=True).eval()
        )
        self.gnn_model = torch.jit.load(gnn_path).to(self.config.device, non_blocking=True).eval()

        if self.config.compling:
            print("compling models do not work now...")
            # self.embedding_model = torch.compile(self.embedding_model)
            # self.filter_model.gnn = torch.compile(self.filter_model.gnn)
            # self.filter_model.net = torch.compile(self.filter_model.net)
            # self.gnn_model = torch.compile(self.gnn_model)

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

    def forward(self, node_features: torch.Tensor, hit_id: torch.Tensor | None = None):
        device = self.config.device
        debug = self.config.debug
        save_debug_data = self.config.save_debug_data
        out_debug_data_name = "debug_data.pt"

        track_candidates = np.array([-1], dtype=np.int64)
        if node_features is None or node_features.shape[0] < 3:
            return track_candidates

        node_features = node_features.to(device).float()
        node_features = torch.nan_to_num(node_features, nan=0.0, posinf=0.0, neginf=0.0)

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

        out_data = Data()
        if save_debug_data:
            out_data = Data(
                embedding_inputs=embedding_inputs, embedding=embedding, node_features=node_features
            )

        # delete the embedding inputs if not needed.
        if self.config.filter_node_features == self.config.embedding_node_features:
            filtering_inputs = embedding_inputs
        else:
            del embedding_inputs
            filtering_inputs = node_features[
                :, [self.input_node_features.index(x) for x in self.config.filter_node_features]
            ]
            filtering_inputs /= torch.tensor(self.config.filter_node_scale, device=device).float()

        if save_debug_data:
            out_data.filtering_nodes = filtering_inputs

        # Build edges
        edge_index = build_edges(
            embedding, embedding, r_max=self.config.r_max, k_max=self.config.k_max
        )

        if debug:
            print(f"Number of edges after embedding: {edge_index.shape[1]:,}")
        else:
            del embedding

        if save_debug_data:
            out_data.embedding_edge_list = edge_index

        if edge_index.shape[1] < 2:
            if save_debug_data:
                torch.save(out_data, out_debug_data_name)
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

        if debug:
            print(f"after removing duplications: {edge_index.shape[1]:,}")

        if save_debug_data:
            out_data.filter_edge_list_before = edge_index

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

        if save_debug_data:
            out_data.filter_node_features = filtering_inputs
            out_data.filter_scores = edge_scores
            out_data.filter_edge_list_after = edge_index

        # apply fitlering score cuts.
        edge_index = edge_index[:, edge_scores >= self.config.filter_cut]
        del edge_scores
        if edge_index.shape[1] < 2:
            return track_candidates

        if debug:
            print(f"Number of edges after filtering: {edge_index.shape[1]:,}")

        # prepare GNN inputs
        gnn_input = node_features[
            :, [self.input_node_features.index(x) for x in self.config.gnn_node_features]
        ]
        gnn_input /= torch.tensor(self.config.gnn_node_scale, device=device).float()

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

        if save_debug_data:
            out_data.gnn_scores = edge_scores
            out_data.gnn_edge_lists = edge_index
            out_data.gnn_edge_features = edge_features
            out_data.gnn_node_features = gnn_input

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
        all_trkx["cc"], graph = walkutils.get_simple_path(graph)
        if debug:
            print("the graph information")
            # with open("graph_info.txt", "w") as f:
            #     f.write(f"{G.nodes(data=True)}\n")
            #     f.write(f"{G.edges(data=True)}\n")
        all_trkx["walk"] = walkutils.walk_through(
            graph, score_name, self.config.walk_min, self.config.walk_max, False
        )
        if debug:
            print(f"Number of tracks found by CC: {len(all_trkx['cc'])}")
            print(f"Number of tracks found by Walkthrough: {len(all_trkx['walk'])}")

        tracks = all_trkx["cc"] + list(all_trkx["walk"])
        total_length = sum(len(trk) + 1 for trk in tracks)
        track_candidates = np.empty(total_length, dtype=int)

        i = 0
        for trk in tracks:
            trk_tensor = torch.tensor(trk, device=R.device)
            sorted_trk = trk_tensor[torch.argsort(R[trk_tensor])]

            n = len(sorted_trk)
            track_candidates[i : i + n] = sorted_trk.cpu().tolist()
            i += n
            track_candidates[i] = -1
            i += 1

        # write candidates to a file.
        if debug:
            print("track_candidates", track_candidates[:20])
        if save_debug_data:
            out_data.track_candidates = torch.from_numpy(track_candidates).to(torch.int64)
            torch.save(out_data, out_debug_data_name)

        return track_candidates

    def __call__(self, node_features: torch.Tensor, hit_id: torch.Tensor | None = None):
        return self.forward(node_features, hit_id)


def create_metric_learning_end2end_rel24(
    model_path: str,
    device: str,
    debug: bool,
    precision: str,
    auto_cast: bool,
    compiling: bool,
    save_data_for_debug: bool,
):
    config = MetricLearningInferenceConfig(
        model_path=model_path,
        device=device,
        auto_cast=auto_cast,
        compling=compiling,
        debug=debug,
        save_debug_data=save_data_for_debug,
        r_max=0.12,
        k_max=1000,
        filter_cut=0.05,
        filter_batches=10,
        cc_cut=0.01,
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
    parser.add_argument("-m", "--model", type=str, default="./", help="Model path")
    parser.add_argument("-p", "--precision", type=str, default="highest", help="Precision")
    parser.add_argument("-a", "--auto_cast", action="store_true", help="Use autocast")
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug mode")
    parser.add_argument("-d", "--device", default="cuda", help="Device")
    parser.add_argument("-t", "--timing", action="store_true", help="Time the inference")
    parser.add_argument("-c", "--compiling", action="store_true", help="Use compiling")
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
        save_data_for_debug=args.save_data_for_debug,
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
