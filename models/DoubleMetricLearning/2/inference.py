from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import graph_segment
import numpy as np
import torch
import torch.cuda.nvtx as nvtx
from double_metric_learning import DoubleMetricLearning
from frnn_cuda.torch import build_edges as frnn_build_edges
from interaction_gnn import (
    ChainedInteractionGNN2,
    GNNFilterJitable,
    RecurrentInteractionGNN2,
    make_deterministic,
)
from torch_model_inference import run_gnn_filter, run_torch_model

torch.manual_seed(42)
torch.set_float32_matmul_precision("high")


def build_edges(
    query: torch.Tensor,
    database: torch.Tensor,
    r_max: float = 0.1,
    k_max: int = 1000,
) -> torch.Tensor:
    """Directed edges query[i] -> database[j] within r_max, at most k_max per query, via libFRNN.

    query and database are different embeddings of the same hits, so libFRNN's exclude_self
    (which only applies to identical inputs) does not drop i -> i; that is done here. As with
    the original FRNN, the k_max neighbors are chosen before the self-loop is removed.
    """
    edge_list = frnn_build_edges(
        query.float().contiguous(),
        database.float().contiguous(),
        radius=r_max,
        max_neighbors=k_max,
        directed=True,
    )
    return edge_list[:, edge_list[0] != edge_list[1]]


def segment(
    num_nodes: int,
    edge_index: torch.Tensor,
    edge_scores: torch.Tensor,
    backend: str,
    cuts: dict,
) -> list:
    """Cut the scored graph into tracks with a graph_segment backend.

    edge_index must point outward (from the hit nearer the interaction point).
    Returns every track the backend makes (e.g. cc + walk), as lists of node indices.
    """
    tracks = graph_segment.get_tracks(
        num_nodes,
        edge_index.cpu().numpy(),
        edge_scores.float().cpu().numpy(),
        backend,
        **cuts,
    )
    return [trk for method_tracks in tracks.values() for trk in method_tracks]


@dataclass
class MetricLearningInferenceConfig:
    model_path: str | Path
    device: str
    auto_cast: bool
    compiling: bool
    debug: bool
    save_debug_data: bool = False
    # Sum GNN messages in a fixed order instead of with CUDA atomics: the same event then
    # always gives the same tracks, at some latency cost.
    deterministic: bool = False
    r_max: float = 0.14
    k_max: int = 1000
    filter_cut: float = 0.01
    filter_batches: int = 10
    # Any key of graph_segment.BACKENDS: cc, cc_walk, cc_bestpred_walk, cc_junction_removal.
    segment_backend: str = "cc_walk"
    # Cuts passed to graph_segment.get_tracks; each backend uses only the cuts it knows.
    # The defaults reproduce version 1: cc_cut=0.01, walk_min=0.1, walk_max=0.6.
    segment_cuts: dict | str = field(
        default_factory=lambda: {"cc_cut": 0.01, "th_min": 0.1, "th_add": 0.6}
    )
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

        # Fail at load time, not on the first request, for a bad backend or cut name.
        if isinstance(self.segment_cuts, str):
            self.segment_cuts = json.loads(self.segment_cuts)
        if self.segment_backend not in graph_segment.BACKENDS:
            raise ValueError(
                f"unknown segment_backend {self.segment_backend!r}, "
                f"choose from {list(graph_segment.BACKENDS)}"
            )
        graph_segment.get_tracks(
            1, np.zeros((2, 0), dtype=np.int64), np.zeros(0, dtype=np.float32),
            self.segment_backend, **self.segment_cuts,
        )  # fmt: skip

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

        if self.config.deterministic:
            make_deterministic(self.filter_model)
            make_deterministic(self.gnn_model)

        device = self.config.device
        self.embedding_scale = torch.tensor(config.embedding_node_scale, device=device)
        self.filter_scale = torch.tensor(config.filter_node_scale, device=device)
        self.gnn_scale = torch.tensor(config.gnn_node_scale, device=device)

        if self.config.compiling:
            print("compiling models works now...")
            torch.set_float32_matmul_precision("high")
            # self.embedding_model = torch._dynamo.optimize("inductor")(self.embedding_model)
            self.embedding_model = torch.compile(
                self.embedding_model, dynamic=True, mode="max-autotune-no-cudagraphs"
            )
            # # Compile GNNFilter
            self.filter_model.gnn = torch.compile(
                self.filter_model.gnn, dynamic=True, mode="max-autotune-no-cudagraphs"
            )
            self.filter_model.net = torch.compile(
                self.filter_model.net, dynamic=True, mode="max-autotune-no-cudagraphs"
            )
            # # Compile interaction gnn
            self.gnn_model = torch.compile(
                self.gnn_model, dynamic=True, mode="max-autotune-no-cudagraphs"
            )
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

        edge_scores, edge_index, _ = run_gnn_filter(
            model=self.filter_model,
            auto_cast=self.config.auto_cast,
            batches=self.config.filter_batches,
            x=filtering_inputs,
            edge_index=edge_index,
        )

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

        if use_nvtx:
            nvtx.range_push("Graph Segmentation")

        # edge_index already points outward (flipped by R above), as graph_segment requires.
        tracks = segment(
            node_features.shape[0],
            edge_index,
            edge_scores,
            self.config.segment_backend,
            self.config.segment_cuts,
        )

        if use_nvtx:
            nvtx.range_pop()  # Graph Segmentation
            torch.cuda.synchronize()
            nvtx.range_pop()  # ML Inference one event

        # Each track ordered by R, mapped to hit_id, terminated by -1.
        cpu_R = R.cpu().numpy()
        cpu_hit_id = hit_id.cpu().numpy()
        track_candidates = np.empty(sum(len(trk) + 1 for trk in tracks), dtype=np.int64)
        i = 0
        for trk in tracks:
            trk = np.asarray(trk, dtype=np.int64)
            n = len(trk)
            track_candidates[i : i + n] = cpu_hit_id[trk[np.argsort(cpu_R[trk])]]
            track_candidates[i + n] = -1
            i += n + 1

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
    deterministic: bool = False,
    segment_backend: str = "cc_walk",
    segment_cuts: dict | str | None = None,
):
    config = MetricLearningInferenceConfig(
        model_path=model_path,
        device=device,
        auto_cast=auto_cast,
        compiling=compiling,
        debug=debug,
        save_debug_data=save_data_for_debug,
        deterministic=deterministic,
        r_max=0.14,
        k_max=1000,
        filter_cut=0.01,
        filter_batches=10,
        segment_backend=segment_backend,
        segment_cuts=segment_cuts or {"cc_cut": 0.01, "th_min": 0.1, "th_add": 0.6},
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
        "-b", "--segment-backend", default="cc_walk",
        help="graph_segment backend: cc, cc_walk, cc_bestpred_walk, cc_junction_removal",
    )  # fmt: skip
    parser.add_argument(
        "--segment-cuts", default=None,
        help='JSON cuts for graph_segment, e.g. \'{"cc_cut": 0.01, "th_min": 0.1, "th_add": 0.6}\'',
    )  # fmt: skip
    parser.add_argument(
        "-s", "--save-data-for-debug", action="store_true", help="Save debugging data"
    )
    parser.add_argument(
        "--deterministic", action="store_true", help="Sum GNN messages in a fixed order"
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
        deterministic=args.deterministic,
        segment_backend=args.segment_backend,
        segment_cuts=args.segment_cuts,
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
