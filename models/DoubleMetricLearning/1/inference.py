from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from operator import itemgetter

import frnn
import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import RemoveIsolatedNodes
from torch_geometric.utils import to_networkx
import torch.cuda.nvtx as nvtx

from torch_model_inference import run_gnn_filter, run_gnn_filter_optimized, run_torch_model
import fastwalkthrough as walkutils
import time
import yaml
import onnxruntime as ort

from double_metric_learning import DoubleMetricLearning
from interaction_gnn import (
    RecurrentInteractionGNN2,
    ChainedInteractionGNN2,
    GNNFilterJitable,
)

# from triton_radius_nn import build_edges_triton as build_edges

torch.manual_seed(42)
torch.set_float32_matmul_precision("high")

# def timed(fn):
#     start = torch.cuda.Event(enable_timing=True)
#     end = torch.cuda.Event(enable_timing=True)
#     start.record()
#     result = fn()
#     end.record()
#     torch.cuda.synchronize()
#     return result, start.elapsed_time(end) / 1000


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
    compiling: bool
    debug: bool
    save_debug_data: bool = False
    r_max: float = 0.14
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

        if self.config.compiling:
            print("compiling models works now...")
            torch.set_float32_matmul_precision("high")
            # self.embedding_model = torch._dynamo.optimize("inductor")(self.embedding_model)
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
        embedding_inputs /= torch.tensor(self.config.embedding_node_scale, device=device).float()

        # Time the embedding, filtering and GNN calls. Use cuda sync
        # around each measured region for accurate timing on GPU.
        total_t0 = time.perf_counter()

        torch.cuda.synchronize()
        embed_t0 = time.perf_counter()
        src_embedding, tgt_embedding = run_torch_model(
            self.embedding_model, self.config.auto_cast, embedding_inputs
        )
        torch.cuda.synchronize()
        embed_t1 = time.perf_counter()
        embed_time = embed_t1 - embed_t0

        if nvtx_enabled:
            nvtx.range_pop()

        if debug:
            print(f"after embedding, shape = {src_embedding.shape}, {tgt_embedding.shape}")
            print("embedding data", src_embedding[0], tgt_embedding[0])
            print("embedding data type", src_embedding.dtype, tgt_embedding.dtype)

        out_data = Data()
        if save_debug_data:
            out_data = Data(
                embedding_inputs=embedding_inputs,
                src_embedding=src_embedding,
                tgt_embedding=tgt_embedding,
                node_features=node_features,
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
        if nvtx_enabled:
            nvtx.range_push("Build Edges")
        # Time the build_edges call for GPU-accurate timing
        torch.cuda.synchronize()
        edges_t0 = time.perf_counter()
        # self.config.k_max  = 1024
        edge_index = build_edges(
            src_embedding, tgt_embedding, r_max=self.config.r_max, k_max=self.config.k_max
        )
        torch.cuda.synchronize()
        edges_t1 = time.perf_counter()
        edge_time = edges_t1 - edges_t0
        if nvtx_enabled:
            nvtx.range_pop()

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
        # Time the filter step
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
        # torch.cuda.synchronize()
        # print(f"run_torch_model (filtering) time: {time.time() - t0:.4f} s")

        if debug:
            print("edge_score", edge_scores[:10])
            print("edge_index", edge_index[:, :10])

        if save_debug_data:
            out_data.filter_node_features = filtering_inputs
            out_data.filter_scores = edge_scores
            out_data.filter_edge_list_after = edge_index

        # apply fitlering score cuts.
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
            return {
                "dr": dr,
                "dphi": dphi,
                "dz": dz,
                "deta": deta,
                "phislope": phislope,
                "rphislope": rphislope,
            }

        edge_features_dict = calculate_edge_features()
        edge_features = torch.stack(list(edge_features_dict.values()), dim=1)

        # torch.cuda.synchronize()
        # t0 = time.time()
        if nvtx_enabled:
            nvtx.range_push("GNN Inference")
        # Time the GNN inference
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
        total_t1 = time.perf_counter()
        total_time = total_t1 - total_t0
        torch.cuda.synchronize()
        if nvtx_enabled:
            nvtx.range_pop()
        torch.cuda.synchronize()
        if nvtx_enabled:
            nvtx.range_pop()
        # print(f"run_torch_model (GNN) time: {time.time() - t0:.4f} s")

        # CC and Walkthrough
        # if nvtx_enabled:
        #     nvtx.range_push("CC and Walkthrough")
        if debug:
            print("After GNN...")

        if save_debug_data:
            out_data.gnn_scores = edge_scores
            out_data.gnn_edge_lists = edge_index
            out_data.gnn_edge_features = edge_features
            out_data.gnn_node_features = gnn_input

        # try:
        #     # Append timing for this event (header created at model initialize)
        #     with open(timing_path, "a") as tf:
        #         tf.write(
        #             f"{int(time.time())},{embed_time:.6f},{edge_time:.6f},{filter_time:.6f},{gnn_time:.6f},{total_time:.6f}\n"
        #         )
        # except Exception as e:
        #     if debug:
        #         print("Failed to write timing file:", e)
        print(f"{int(time.time())},{embed_time:.6f},{edge_time:.6f},{filter_time:.6f},{gnn_time:.6f},{total_time:.6f}")

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
        if nvtx_enabled:
            nvtx.range_pop()

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
        r_max=0.14,
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
