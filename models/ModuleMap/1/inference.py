from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from operator import itemgetter

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import RemoveIsolatedNodes
from torch_geometric.utils import to_networkx
import torch.cuda.nvtx as nvtx

from pymmg import GraphBuilder
from torch_model_inference import run_torch_model
import fastwalkthrough as walkutils
import time


from interaction_gnn import (
    RecurrentInteractionGNN2,
    ChainedInteractionGNN2,
    GNNFilterJitable,
)

torch.manual_seed(42)
torch.set_float32_matmul_precision("high")


@dataclass
class ModuleMapInferenceConfig:
    model_path: str | Path
    module_map_pattern_path: str | Path
    device: str
    auto_cast: bool
    compiling: bool
    debug: bool
    device_id: int = 0
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
        self.module_map_pattern_path = (
            Path(self.module_map_pattern_path)
            if isinstance(self.module_map_pattern_path, str)
            else self.module_map_pattern_path
        )


class ModuleMapInference:
    def __init__(self, config: ModuleMapInferenceConfig):
        self.config = config
        print(self.config)
        self.device = self.config.device

        print("Path: ", self.config.module_map_pattern_path)
        self.graph_builder = GraphBuilder(
            module_map_path=str(self.config.module_map_pattern_path),
            device=self.config.device_id,
        )

        gnn_path = "./MM_minmax_ignn2.ckpt"

        # load the checkpoint
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
        self.gnn_model.to(self.device).eval()

        if self.config.compiling:
            # # Compile interaction gnn
            self.gnn_model = torch.compile(self.gnn_model, dynamic=True, mode="max-autotune")

        self.input_node_features = [
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

    def forward(
        self,
        node_features: torch.Tensor,
        hit_id: torch.Tensor | None = None,
        nvtx_enabled: bool = False,
    ):
        device = self.config.device
        debug = self.config.debug

        track_candidates = np.array([-1], dtype=np.int64)
        if node_features is None or node_features.shape[0] < 3:
            return track_candidates

        node_features = node_features.to(device).float()
        node_features = torch.nan_to_num(node_features, nan=0.0, posinf=0.0, neginf=0.0)

        if hit_id is None:
            hit_id = torch.arange(node_features.shape[0], device=device)

        if nvtx_enabled:
            nvtx.range_push("MM Inference one event")

        # Build edges
        if nvtx_enabled:
            nvtx.range_push("Build Edges")

        edge_index = self.graph_builder.build_edge_index(
            hit_id=hit_id,
            hit_module_id=node_features[:, self.input_node_features.index("module_id")],
            hit_x=node_features[:, self.input_node_features.index("x")],
            hit_y=node_features[:, self.input_node_features.index("y")],
            hit_z=node_features[:, self.input_node_features.index("z")],
            nb_hits=hit_id.shape[0],
        )

        if debug:
            print(f"Number of edges after embedding: {edge_index.shape[1]:,}")
        # edge_index = edge_index.to(device)

        # # order the edges by their distance from the collision point.
        R = (
            node_features[:, self.input_node_features.index("r")] ** 2
            + node_features[:, self.input_node_features.index("z")] ** 2
        )

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

        if nvtx_enabled:
            nvtx.range_push("GNN Inference")

        edge_scores = (
            run_torch_model(
                self.gnn_model, self.config.auto_cast, gnn_input, edge_index, edge_features
            )
            .sigmoid()
            .to(torch.float32)
        )
        torch.cuda.synchronize()
        if nvtx_enabled:
            nvtx.range_pop()
        torch.cuda.synchronize()
        if nvtx_enabled:
            nvtx.range_pop()

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

        return track_candidates

    def __call__(
        self,
        node_features: torch.Tensor,
        hit_id: torch.Tensor | None = None,
        nvtx_enabled: bool = False,
    ):
        return self.forward(node_features, hit_id=hit_id, nvtx_enabled=nvtx_enabled)


def create_module_map_end2end_rel24(
    model_path: str,
    module_map_pattern_path: str,
    device: str,
    device_id: int,
    debug: bool,
    precision: str,
    auto_cast: bool,
    compiling: bool,
    save_data_for_debug: bool,
):
    config = ModuleMapInferenceConfig(
        model_path=model_path,
        module_map_pattern_path=module_map_pattern_path,
        device=device,
        device_id=device_id,
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
    model = ModuleMapInference(config)
    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inference for ModuleMap pipeline")
    parser.add_argument("-i", "--input", type=str, default="node_features.pt", help="Input file")
    parser.add_argument("-m", "--model", type=str, default="./", help="Model path")
    parser.add_argument(
        "-mmp", "--module_map_pattern_path", type=str, default="./", help="Module map pattern path"
    )
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

    inference = create_module_map_end2end_rel24(
        model_path=args.model,
        module_map_pattern_path=args.module_map_pattern_path,
        device=args.device,
        device_id=0,
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
        for _ in tqdm(range(num_trials)):
            track_ids = inference(node_features, nvtx_enabled=True)
        nvtx.range_pop()
        print("average time per event", (time.time() - start_time) / num_trials)
    print("total tracks", np.sum(track_ids == -1))
