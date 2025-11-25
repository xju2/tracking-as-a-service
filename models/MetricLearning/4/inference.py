from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import fastwalkthrough as walkutils
import frnn
import numpy as np
import torch
import torch.cuda.nvtx as nvtx
from interaction_gnn import (
    ChainedInteractionGNN2,
    GNNFilterJitable,
    RecurrentInteractionGNN2,
)
from metric_learning import MetricLearning
from torch_geometric.data import Data
from torch_geometric.transforms import RemoveIsolatedNodes
from torch_model_inference import run_gnn_filter_optimized, run_torch_model


torch.manual_seed(42)
torch.set_float32_matmul_precision("high")


def to_trk_tensor(trk, device):
    # Convert numba.typed.List or Python list -> numpy
    if not isinstance(trk, np.ndarray):
        trk = np.array(trk, dtype=np.int64)
    else:
        trk = trk.astype(np.int64, copy=False)

    # Finally -> torch tensor on correct device
    return torch.as_tensor(trk, dtype=torch.long, device=device)


def build_edges(
    query: torch.Tensor,
    database: torch.Tensor,
    r_max: float = 0.1,
    k_max: int = 1000,
) -> torch.Tensor:
    # FRNN expects float32 inputs - cast to float32.
    orig_device = query.device
    query = query.to(torch.float32)
    database = database.to(torch.float32)
        
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
    compiling: bool
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

        device = self.config.device
        # Load checkpoints instead of .pt files
        # Find checkpoint directory
        embedding_path = model_path / "embedding.ckpt"
        filtering_path = model_path / "filter.ckpt"
        gnn_path = model_path / "gnn.ckpt"

        print(f"Loading checkpoint from {embedding_path}")
        checkpoint = torch.load(embedding_path, map_location="cpu")
        self.embedding_model = MetricLearning(checkpoint["hyper_parameters"])
        self.embedding_model.load_state_dict(checkpoint["state_dict"])
        self.embedding_model.to(device).eval()

        print(f"Loading checkpoint from {filtering_path}")
        checkpoint = torch.load(filtering_path, map_location="cpu")
        new_model = GNNFilterJitable(checkpoint["hyper_parameters"])
        new_model.load_state_dict(checkpoint["state_dict"])
        self.filter_model = new_model
        self.filter_model.to(device).eval()

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
        self.gnn_model.to(device).eval()

        self.embedding_scale = torch.tensor(
            self.config.embedding_node_scale, device=device
        ).float()
        self.filter_scale = torch.tensor(self.config.filter_node_scale, device=device).float()
        self.gnn_scale = torch.tensor(self.config.gnn_node_scale, device=device).float()

        if self.config.auto_cast:
            # Move models to half precision before compiling when appropriate.
            try:
                self.embedding_model.half().eval()
            except Exception:
                pass

            try:
                # filter_model has submodules gnn and net
                self.filter_model.gnn.half().eval()
            except Exception:
                pass
            try:
                self.filter_model.net.half().eval()
            except Exception:
                pass

            try:
                self.gnn_model.half().eval()
            except Exception:
                pass

        if self.config.compiling:
            print("compiling models works now...")
            torch.set_float32_matmul_precision("high")

            self.embedding_model = torch.compile(
                self.embedding_model, dynamic=True, mode="max-autotune"
            )
            # # Compile GNNFilter
            self.filter_model.gnn = torch.compile(
                self.filter_model.gnn, dynamic=True, mode="max-autotune"
            )
            self.filter_model.net = torch.compile(
                self.filter_model.net, dynamic=True, mode="max-autotune"
            )
            # # Compile interaction gnn
            self.gnn_model = torch.compile(self.gnn_model, dynamic=True, mode="max-autotune")

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

    def forward(
        self,
        node_features: torch.Tensor,
        hit_id: torch.Tensor | None = None,
        nvtx_enabled: bool = False,
    ):
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

        if nvtx_enabled:
            nvtx.range_push("ML Inference one event")

        # Metric Learning
        if nvtx_enabled:
            nvtx.range_push("Metric Learning Inference")

        embedding_inputs = node_features[
            :, [self.input_node_features.index(x) for x in self.config.embedding_node_features]
        ]
        embedding_inputs /= self.embedding_scale

        # Time embedding
        torch.cuda.synchronize()
        total_t0 = time.perf_counter()

        torch.cuda.synchronize()
        embed_t0 = time.perf_counter()
        embedding = run_torch_model(self.embedding_model, self.config.auto_cast, embedding_inputs)
        torch.cuda.synchronize()
        embed_t1 = time.perf_counter()
        embed_time = embed_t1 - embed_t0

        if nvtx_enabled:
            nvtx.range_pop()

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
            filtering_inputs /= self.filter_scale

        if save_debug_data:
            out_data.filtering_nodes = filtering_inputs

        # Build edges
        if nvtx_enabled:
            nvtx.range_push("Build Edges")

        # Time edge build
        if nvtx_enabled:
            nvtx.range_push("Build Edges")
        torch.cuda.synchronize()
        edges_t0 = time.perf_counter()
        edge_index = build_edges(
            embedding, embedding, r_max=self.config.r_max, k_max=self.config.k_max
        )
        torch.cuda.synchronize()
        edges_t1 = time.perf_counter()
        edge_time = edges_t1 - edges_t0
        if nvtx_enabled:
            nvtx.range_pop()

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
        if nvtx_enabled:
            nvtx.range_push("GNN Filtering")

        # Time filter
        if nvtx_enabled:
            nvtx.range_push("GNN Filtering")
        torch.cuda.synchronize()
        filter_t0 = time.perf_counter()
        edge_scores, edge_index, _ = run_gnn_filter_optimized(
            self.filter_model,
            self.config.auto_cast,
            self.config.filter_batches,
            filtering_inputs,
            edge_index,
        )
        torch.cuda.synchronize()
        filter_t1 = time.perf_counter()
        filter_time = filter_t1 - filter_t0
        if nvtx_enabled:
            nvtx.range_pop()

        if debug:
            print("edge_score", edge_scores[:10])
            print("edge_index", edge_index[:, :10])

        if save_debug_data:
            out_data.filter_node_features = filtering_inputs
            out_data.filter_scores = edge_scores
            out_data.filter_edge_list_after = edge_index

        # apply filtering score cuts.
        edge_index = edge_index[:, edge_scores >= self.config.filter_cut]
        # del edge_scores
        if edge_index.shape[1] < 2:
            return track_candidates

        if debug:
            print(f"Number of edges after filtering: {edge_index.shape[1]:,}")
        torch.cuda.synchronize()
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
            return {
                "dr": dr,
                "dphi": dphi,
                "dz": dz,
                "deta": deta,
                "phislope": phislope,
                "rphislope": rphislope,
            }

        edge_features_dict = calculate_edge_features()
        edge_features = torch.stack([v.to(device).float() for v in edge_features_dict.values()], dim=1)

        # torch.cuda.synchronize()
        # t0 = time.time()
        # Time GNN
        if nvtx_enabled:
            nvtx.range_push("GNN Inference")
        torch.cuda.synchronize()
        gnn_t0 = time.perf_counter()
        edge_scores = (
            run_torch_model(
                self.gnn_model, self.config.auto_cast, gnn_input, edge_index, edge_features
            )
            .sigmoid()
            .to(torch.float32)
        )
        torch.cuda.synchronize()
        gnn_t1 = time.perf_counter()
        gnn_time = gnn_t1 - gnn_t0
        if nvtx_enabled:
            nvtx.range_pop()
        torch.cuda.synchronize()
        if nvtx_enabled:
            nvtx.range_pop()

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
            edge_index=edge_index,
            hit_id=hit_id,
            num_nodes=node_features.shape[0],
        ).to(device)
        graph[score_name] = edge_scores
        graph = RemoveIsolatedNodes()(graph)

        # Time CC and Walkthrough (track-building)
        torch.cuda.synchronize()
        track_build_t0 = time.perf_counter()

        all_trkx = {}
        all_trkx["cc"], graph = walkutils.get_simple_path(graph)

        all_trkx["walk"] = walkutils.walk_through(
            graph, score_name, self.config.walk_min, self.config.walk_max, False
        )

        torch.cuda.synchronize()
        track_build_t1 = time.perf_counter()
        track_time = track_build_t1 - track_build_t0
        if debug:
            print(f"Number of tracks found by CC: {len(all_trkx['cc'])}")
            print(f"Number of tracks found by Walkthrough: {len(all_trkx['walk'])}")
        if nvtx_enabled:
            nvtx.range_pop()

        tracks = all_trkx["cc"] + list(all_trkx["walk"])
        total_length = sum(len(trk) + 1 for trk in tracks)
        track_candidates = np.empty(total_length, dtype=int)

        # Time track assembly (tensor construction & sorting)
        torch.cuda.synchronize()
        assemble_t0 = time.perf_counter()

        i = 0
        for trk in tracks:
            trk_tensor = to_trk_tensor(trk, device)
            sorted_trk = trk_tensor[torch.argsort(R[trk_tensor])]

            n = len(sorted_trk)
            track_candidates[i : i + n] = sorted_trk.tolist()
            i += n
            track_candidates[i] = -1
            i += 1

        torch.cuda.synchronize()
        assemble_t1 = time.perf_counter()
        assemble_time = assemble_t1 - assemble_t0

        # Total time
        torch.cuda.synchronize()
        total_t1 = time.perf_counter()
        total_time = total_t1 - total_t0

        print(f"{int(time.time())},{embed_time:.6f},{edge_time:.6f},{filter_time:.6f},{gnn_time:.6f},{track_time:.6f},{assemble_time:.6f},{total_time:.6f}\n")
  

        # write candidates to a file.
        if debug:
            print("track_candidates", track_candidates[:20])
        if save_debug_data:
            out_data.track_candidates = torch.from_numpy(track_candidates).to(torch.int64)
            torch.save(out_data, out_debug_data_name)

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
    save_data_for_debug: bool,
):
    config = MetricLearningInferenceConfig(
        model_path=model_path,
        device=device,
        auto_cast=auto_cast,
        compiling=compiling,
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
