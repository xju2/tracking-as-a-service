"""Compare the edge lists built by the original FRNN and by libFRNN.

The embedding stage of the MetricLearning pipeline turns the GNN4ITk node features
into points in the embedding space. Both libraries then build the radius graph from
the same float32 embedding and the two directed edge lists are compared as sets.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class FRNNEvalConfig:
    # Directory holding embedding.ckpt and the MetricLearning python modules.
    embedding_model_dir: str | Path
    output_dir: str | Path
    device: str = "cuda"
    auto_cast: bool = False
    debug: bool = False
    # Append per-request performance metrics to output_dir/eval_metrics_pid<pid>.csv.
    save_eval_metrics: bool = False
    # Save every event's embedding and edge lists under output_dir/data/ for libFRNN.
    save_data: bool = False
    # Record each library's peak GPU memory in an extra, untimed pass (see peak_memory_mb).
    measure_memory: bool = False
    r_max: float = 0.12
    k_max: int = 1000
    embedding_node_features: str = "r, phi, z, cluster_x_1, cluster_y_1, cluster_z_1, cluster_x_2, cluster_y_2, cluster_z_2, count_1, charge_count_1, loc_eta_1, loc_phi_1, localDir0_1, localDir1_1, localDir2_1, lengthDir0_1, lengthDir1_1, lengthDir2_1, glob_eta_1, glob_phi_1, eta_angle_1, phi_angle_1, count_2, charge_count_2, loc_eta_2, loc_phi_2, localDir0_2, localDir1_2, localDir2_2, lengthDir0_2, lengthDir1_2, lengthDir2_2, glob_eta_2, glob_phi_2, eta_angle_2, phi_angle_2"
    embedding_node_scale: str = "1000, 3.14, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1, 1, 3.14, 3.14, 1, 1, 1, 1, 1, 1, 3.14, 3.14, 3.14, 3.14, 1, 1, 3.14, 3.14, 1, 1, 1, 1, 1, 1, 3.14, 3.14, 3.14, 3.14"

    def __post_init__(self):
        self.embedding_model_dir = Path(self.embedding_model_dir)
        self.output_dir = Path(self.output_dir)
        self.embedding_node_features = [x.strip() for x in self.embedding_node_features.split(",")]
        self.embedding_node_scale = [
            float(x.strip()) for x in self.embedding_node_scale.split(",")
        ]
        assert len(self.embedding_node_features) == len(self.embedding_node_scale)


# Column order of the 44 input features, identical to MetricLearning.
INPUT_NODE_FEATURES = [
    "r", "phi", "z",
    "cluster_x_1", "cluster_y_1", "cluster_z_1",
    "cluster_x_2", "cluster_y_2", "cluster_z_2",
    "count_1", "charge_count_1", "loc_eta_1", "loc_phi_1",
    "localDir0_1", "localDir1_1", "localDir2_1",
    "lengthDir0_1", "lengthDir1_1", "lengthDir2_1",
    "glob_eta_1", "glob_phi_1", "eta_angle_1", "phi_angle_1",
    "count_2", "charge_count_2", "loc_eta_2", "loc_phi_2",
    "localDir0_2", "localDir1_2", "localDir2_2",
    "lengthDir0_2", "lengthDir1_2", "lengthDir2_2",
    "glob_eta_2", "glob_phi_2", "eta_angle_2", "phi_angle_2",
    "eta",
    "cluster_r_1", "cluster_phi_1", "cluster_eta_1",
    "cluster_r_2", "cluster_phi_2", "cluster_eta_2",
]  # fmt: skip


def build_edges_frnn(embedding: torch.Tensor, r_max: float, k_max: int) -> torch.Tensor:
    """Original FRNN, exactly as in MetricLearning/4/inference.py::build_edges."""
    import frnn

    _, idxs, _, _ = frnn.frnn_grid_points(
        points1=embedding.unsqueeze(0),
        points2=embedding.unsqueeze(0),
        lengths1=None,
        lengths2=None,
        K=k_max,
        r=r_max,
        grid=None,
        return_nn=False,
        return_sorted=True,
    )
    idxs = idxs.squeeze(0).int()
    ind = torch.arange(idxs.shape[0], device=embedding.device).repeat(idxs.shape[1], 1).T.int()
    positive_idxs = idxs >= 0
    edge_list = torch.stack([ind[positive_idxs], idxs[positive_idxs]]).long()
    return edge_list[:, edge_list[0] != edge_list[1]]


def build_edges_libfrnn(embedding: torch.Tensor, r_max: float, k_max: int) -> torch.Tensor:
    """libFRNN with the same semantics as build_edges_frnn.

    The original FRNN keeps K neighbors *including* the point itself and removes the
    self-loop afterwards, while libFRNN applies max_neighbors after excluding self.
    Hence max_neighbors = k_max - 1.
    """
    from frnn_cuda.torch import build_edges

    return build_edges(
        embedding,
        radius=r_max,
        max_neighbors=k_max - 1,
        exclude_self=True,
        directed=True,
    )


def count_different_edges(edges_a: torch.Tensor, edges_b: torch.Tensor, num_nodes: int) -> int:
    """Size of the symmetric difference of two directed edge sets."""
    codes_a = torch.unique(edges_a[0] * num_nodes + edges_a[1])
    codes_b = torch.unique(edges_b[0] * num_nodes + edges_b[1])
    _, counts = torch.unique(torch.cat([codes_a, codes_b]), return_counts=True)
    return int((counts == 1).sum().item())


def timed(fn, *args):
    """Run fn on the GPU and return (result, latency in ms)."""
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = fn(*args)
    torch.cuda.synchronize()
    return result, (time.perf_counter() - start) * 1000.0


def peak_memory_mb(fn, *args, device) -> float:
    """Run fn and return the peak GPU memory in MB it needs on top of what was in use.

    PyTorch's peak counter is exact for memory PyTorch allocates (both edge lists and
    the original FRNN's buffers) but misses raw cudaMalloc, such as libFRNN's
    per-call workspace. That part is sampled from a background thread as the device
    memory in use minus what PyTorch has reserved, so other processes on the same GPU
    add noise. The sampler competes with fn for the GIL, hence the separate pass.
    """
    def non_torch_bytes():
        free, total = torch.cuda.mem_get_info(device)
        return total - free - torch.cuda.memory_reserved(device)

    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    torch_base = torch.cuda.memory_allocated(device)
    non_torch_base = non_torch_peak = non_torch_bytes()
    done = threading.Event()

    def sample():
        nonlocal non_torch_peak
        while not done.is_set():
            non_torch_peak = max(non_torch_peak, non_torch_bytes())

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    try:
        fn(*args)
        torch.cuda.synchronize(device)
    finally:
        done.set()
        sampler.join()
    torch_peak = torch.cuda.max_memory_allocated(device) - torch_base
    return (torch_peak + max(non_torch_peak - non_torch_base, 0)) / 2**20


EVAL_METRICS_COLUMNS = [
    "request_id",
    "num_space_points",
    "num_frnn_edges",
    "num_libfrnn_edges",
    "num_diff_edges",
    "frnn_latency_ms",
    "libfrnn_latency_ms",
    "frnn_peak_mem_mb",
    "libfrnn_peak_mem_mb",
]


class FRNNEval:
    def __init__(self, config: FRNNEvalConfig):
        self.config = config
        print(self.config)

        model_dir = str(self.config.embedding_model_dir.resolve())
        if model_dir not in sys.path:
            sys.path.insert(0, model_dir)
        from metric_learning import MetricLearning
        from torch_model_inference import run_torch_model

        self.run_torch_model = run_torch_model

        device = self.config.device
        embedding_path = self.config.embedding_model_dir / "embedding.ckpt"
        print(f"Loading checkpoint from {embedding_path}")
        checkpoint = torch.load(embedding_path, map_location="cpu")
        self.embedding_model = MetricLearning(checkpoint["hyper_parameters"])
        self.embedding_model.load_state_dict(checkpoint["state_dict"])
        self.embedding_model.to(device).eval()

        self.embedding_scale = torch.tensor(
            self.config.embedding_node_scale, device=device
        ).float()
        self.embedding_columns = [
            INPUT_NODE_FEATURES.index(x) for x in self.config.embedding_node_features
        ]
        self.num_saved = 0
        self.num_data_saved = 0

    def embed(self, node_features: torch.Tensor) -> torch.Tensor:
        node_features = node_features.to(self.config.device).float()
        node_features = torch.nan_to_num(node_features, nan=0.0, posinf=0.0, neginf=0.0)
        embedding_inputs = node_features[:, self.embedding_columns] / self.embedding_scale
        embedding = self.run_torch_model(
            self.embedding_model, self.config.auto_cast, embedding_inputs
        )
        # libFRNN requires contiguous float32; feed both libraries the same tensor.
        return embedding.float().contiguous()

    def save(self, embedding, edges_frnn, edges_libfrnn, num_diff) -> Path:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        out_path = self.config.output_dir / f"frnn_diff_{stamp}_pid{os.getpid()}_{self.num_saved}.pt"
        torch.save(
            {
                "embedding": embedding.cpu(),
                "edges_frnn": edges_frnn.cpu(),
                "edges_libfrnn": edges_libfrnn.cpu(),
                "num_diff_edges": num_diff,
                "r_max": self.config.r_max,
                "k_max": self.config.k_max,
            },
            out_path,
        )
        self.num_saved += 1
        return out_path

    def save_data(self, embedding, edges_frnn, edges_libfrnn, num_diff, request_id) -> Path:
        """Save one event as .npy files for the libFRNN benchmarks.

        output_dir/data/<time>_pid<pid>_<n>/
            embedding.npy       float32 [N, D], C order, the exact input of both libraries
            edges_frnn.npy      int64 [2, E], original FRNN (reference), directed, no self-loops
            edges_libfrnn.npy   int64 [2, E'], libFRNN
            meta.json           request_id, num_nodes, dim, r_max, k_max, edge counts, num_diff_edges
        """
        import numpy as np

        stamp = time.strftime("%Y%m%d-%H%M%S")
        out_dir = self.config.output_dir / "data" / f"{stamp}_pid{os.getpid()}_{self.num_data_saved}"
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "embedding.npy", embedding.cpu().numpy())
        np.save(out_dir / "edges_frnn.npy", edges_frnn.cpu().numpy())
        np.save(out_dir / "edges_libfrnn.npy", edges_libfrnn.cpu().numpy())
        meta = {
            "request_id": request_id,
            "num_nodes": embedding.shape[0],
            "dim": embedding.shape[1],
            "r_max": self.config.r_max,
            "k_max": self.config.k_max,
            "num_frnn_edges": edges_frnn.shape[1],
            "num_libfrnn_edges": edges_libfrnn.shape[1],
            "num_diff_edges": num_diff,
        }
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        self.num_data_saved += 1
        return out_dir

    def save_metrics(self, metrics: dict) -> None:
        # One file per process, so multiple model instances never write to the same file.
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.config.output_dir / f"eval_metrics_pid{os.getpid()}.csv"
        write_header = not out_path.exists()
        with open(out_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=EVAL_METRICS_COLUMNS)
            if write_header:
                writer.writeheader()
            writer.writerow(metrics)

    def __call__(self, node_features: torch.Tensor, request_id: str = "") -> int:
        """Return 0 if both libraries agree, otherwise the number of differing edges."""
        embedding = self.embed(node_features)
        if not torch.isfinite(embedding).all():
            raise ValueError("embedding contains NaN or infinity")

        r_max, k_max = self.config.r_max, self.config.k_max
        edges_frnn, frnn_latency = timed(build_edges_frnn, embedding, r_max, k_max)
        edges_libfrnn, libfrnn_latency = timed(build_edges_libfrnn, embedding, r_max, k_max)
        num_diff = count_different_edges(edges_frnn, edges_libfrnn, embedding.shape[0])
        frnn_mem = libfrnn_mem = None
        if self.config.measure_memory:
            device = embedding.device
            frnn_mem = peak_memory_mb(build_edges_frnn, embedding, r_max, k_max, device=device)
            libfrnn_mem = peak_memory_mb(
                build_edges_libfrnn, embedding, r_max, k_max, device=device
            )

        if self.config.debug:
            print(
                f"{embedding.shape[0]:,} nodes, "
                f"FRNN {edges_frnn.shape[1]:,} edges in {frnn_latency:.2f} ms, "
                f"libFRNN {edges_libfrnn.shape[1]:,} edges in {libfrnn_latency:.2f} ms, "
                f"{num_diff:,} differ"
            )
            if self.config.measure_memory:
                print(f"peak GPU memory: FRNN {frnn_mem:,.1f} MB, libFRNN {libfrnn_mem:,.1f} MB")
        if self.config.save_eval_metrics:
            self.save_metrics(
                {
                    "request_id": request_id,
                    "num_space_points": embedding.shape[0],
                    "num_frnn_edges": edges_frnn.shape[1],
                    "num_libfrnn_edges": edges_libfrnn.shape[1],
                    "num_diff_edges": num_diff,
                    "frnn_latency_ms": f"{frnn_latency:.3f}",
                    "libfrnn_latency_ms": f"{libfrnn_latency:.3f}",
                    "frnn_peak_mem_mb": "" if frnn_mem is None else f"{frnn_mem:.1f}",
                    "libfrnn_peak_mem_mb": "" if libfrnn_mem is None else f"{libfrnn_mem:.1f}",
                }
            )
        if self.config.save_data:
            out_dir = self.save_data(embedding, edges_frnn, edges_libfrnn, num_diff, request_id)
            if self.config.debug:
                print(f"Saved embedding and edge lists to {out_dir}")
        if num_diff:
            out_path = self.save(embedding, edges_frnn, edges_libfrnn, num_diff)
            print(f"FRNN and libFRNN disagree on {num_diff:,} edges; saved to {out_path}")
        return num_diff


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare FRNN and libFRNN edge lists")
    parser.add_argument("-i", "--input", required=True, help="Node features [N, 44] .pt file")
    parser.add_argument(
        "-m", "--model", default="../../MetricLearning/4", help="Embedding model directory"
    )
    parser.add_argument("-o", "--output-dir", default="frnn_eval_outputs")
    parser.add_argument("-a", "--auto_cast", action="store_true", help="Use autocast")
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug mode")
    parser.add_argument(
        "-s", "--save-eval-metrics", action="store_true", help="Save performance metrics"
    )
    parser.add_argument(
        "-d", "--save-data", action="store_true", help="Save embedding and edge lists as .npy"
    )
    parser.add_argument(
        "-M", "--measure-memory", action="store_true", help="Record peak GPU memory"
    )
    args = parser.parse_args()

    evaluator = FRNNEval(
        FRNNEvalConfig(
            embedding_model_dir=args.model,
            output_dir=args.output_dir,
            auto_cast=args.auto_cast,
            debug=args.verbose,
            save_eval_metrics=args.save_eval_metrics,
            save_data=args.save_data,
            measure_memory=args.measure_memory,
        )
    )
    result = evaluator(torch.load(args.input), request_id=Path(args.input).name)
    print("result", result)
    raise SystemExit(0 if result == 0 else 1)
