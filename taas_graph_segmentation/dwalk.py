from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch
from scipy.sparse.csgraph import connected_components
from torch_geometric.data import Data
from torch_geometric.utils import to_scipy_sparse_matrix
from torch_scatter import scatter_max


REPO_ROOT = Path.cwd()
if (REPO_ROOT / "taas_cuda_ext").exists() and str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from taas_cuda_ext import (
        connected_components_available,
        connected_components_weak,
        connected_components_weak_int32,
        process_components_cuda,
        reverse_topological_dp,
        reverse_topological_dp_available,
        trace_selected_paths_available,
        trace_selected_paths_cuda,
    )
except ImportError:
    connected_components_weak = None
    connected_components_weak_int32 = None
    process_components_cuda = None
    reverse_topological_dp = None
    trace_selected_paths_cuda = None

    def connected_components_available() -> bool:
        return False

    def reverse_topological_dp_available() -> bool:
        return False

    def trace_selected_paths_available() -> bool:
        return False


VALID_CC_BACKENDS = {"auto", "scipy", "custom_cuda"}
SMALL_GRAPH_DP_NODE_THRESHOLD = 2048
SMALL_GRAPH_DP_EDGE_THRESHOLD = 4096


class GroupedTrackTensors:
    def __init__(self, grouped_hit_ids: torch.Tensor, counts: torch.Tensor):
        self.grouped_hit_ids = grouped_hit_ids
        self.counts = counts

    def __iter__(self):
        start = 0
        for track_length in self.counts.detach().cpu().tolist():
            end = start + track_length
            yield self.grouped_hit_ids[start:end]
            start = end


def normalize_cc_backend(cc_backend: str | None) -> str:
    backend = (cc_backend or "auto").lower()
    if backend not in VALID_CC_BACKENDS:
        raise ValueError(
            f"Unsupported connected-components backend '{cc_backend}'. "
            "Expected one of: auto, scipy, custom_cuda."
        )
    return backend


def should_use_custom_cuda(device: torch.device, use_gpu: bool, cc_backend: str) -> bool:
    backend = normalize_cc_backend(cc_backend)
    if backend == "custom_cuda":
        if not (use_gpu and device.type == "cuda" and connected_components_available()):
            raise RuntimeError(
                "cc_backend='custom_cuda' requires use_gpu=True, a CUDA graph, "
                "and a built TAAS CUDA connected-components extension."
            )
        return True
    return False


def get_dwalk_metric_weights(graph: Data, score_name: str, path_metrics: str) -> torch.Tensor:
    if path_metrics == "score_weighted_length":
        return graph[score_name]
    if path_metrics == "length":
        return torch.ones_like(graph[score_name])
    raise ValueError(f"Unsupported D-WALK path metric: {path_metrics}")


def get_dwalk_min_root_score(path_metrics: str) -> float:
    if path_metrics == "score_weighted_length":
        return 0.0
    if path_metrics == "length":
        return 2.0
    raise ValueError(f"Unsupported D-WALK path metric: {path_metrics}")


def edge_mask_subgraph(graph: Data, edge_mask: torch.Tensor, score_name: str) -> Data:
    if edge_mask.dtype != torch.bool:
        edge_mask = edge_mask.bool()
    if edge_mask.numel() != graph.edge_index.shape[1]:
        raise ValueError(
            f"D-WALK edge mask length {edge_mask.numel()} does not match "
            f"edge count {graph.edge_index.shape[1]}."
        )

    src = graph.edge_index[0, edge_mask]
    dst = graph.edge_index[1, edge_mask]
    edge_scores = graph[score_name][edge_mask].float()
    num_old_nodes = int(graph.num_nodes)

    active_node_mask = torch.zeros(
        num_old_nodes,
        dtype=torch.bool,
        device=graph.edge_index.device,
    )
    if src.numel() > 0:
        active_node_mask[src] = True
        active_node_mask[dst] = True

    kept_node_ids = torch.nonzero(active_node_mask, as_tuple=False).flatten()
    num_new_nodes = int(kept_node_ids.numel())
    new_node_ids = torch.full(
        (num_old_nodes,),
        -1,
        dtype=torch.long,
        device=graph.edge_index.device,
    )
    if num_new_nodes > 0:
        new_node_ids[kept_node_ids] = torch.arange(
            num_new_nodes,
            dtype=torch.long,
            device=graph.edge_index.device,
        )

    new_graph = Data(
        edge_index=torch.stack((new_node_ids[src], new_node_ids[dst]), dim=0),
        num_nodes=num_new_nodes,
    )
    new_graph[score_name] = edge_scores
    for key in ("hit_id", "R", "hit_r", "hit_z"):
        if hasattr(graph, key):
            value = graph[key]
            if torch.is_tensor(value) and value.numel() == num_old_nodes:
                new_graph[key] = value[kept_node_ids]
            else:
                new_graph[key] = value
    return new_graph


def filter_graph(graph: Data, score_name: str, threshold: float) -> Data:
    mask = graph[score_name] > threshold
    return edge_mask_subgraph(graph, mask, score_name)


def remove_cycles(graph: Data) -> Data:
    radial_distance = get_radial_order_key(graph)
    edge_flip_mask = (radial_distance[graph.edge_index[0]] > radial_distance[graph.edge_index[1]]) | (
        (radial_distance[graph.edge_index[0]] == radial_distance[graph.edge_index[1]])
        & (graph.edge_index[0] > graph.edge_index[1])
    )
    graph.edge_index[:, edge_flip_mask] = graph.edge_index[:, edge_flip_mask].flip(0)
    return graph


def compute_component_labels_from_backend(
    edge_index: torch.Tensor,
    num_nodes: int,
    use_gpu: bool = False,
    cc_backend: str = "auto",
) -> tuple[torch.Tensor, int]:
    if edge_index.numel() == 0:
        return torch.arange(num_nodes, device=edge_index.device, dtype=torch.long), num_nodes

    if should_use_custom_cuda(edge_index.device, use_gpu, cc_backend):
        labels, num_components = connected_components_weak(
            edge_index[0], edge_index[1], num_nodes
        )
        return labels, num_components

    cpu_edge_index = edge_index.cpu() if edge_index.device.type != "cpu" else edge_index
    adjacency = to_scipy_sparse_matrix(cpu_edge_index, num_nodes=num_nodes)
    num_components, labels = connected_components(
        csgraph=adjacency,
        directed=True,
        connection="weak",
    )
    return torch.from_numpy(labels).long().to(edge_index.device), int(num_components)


def labels_to_lists(labels: torch.Tensor, hit_ids: torch.Tensor) -> tuple[torch.Tensor, ...]:
    if labels.numel() == 0:
        return ()

    _, counts = torch.unique(labels, return_counts=True)
    grouped_order = torch.argsort(labels, stable=True)
    grouped_hit_ids = hit_ids[grouped_order]
    return torch.split(grouped_hit_ids, counts.detach().cpu().tolist())


def labels_to_grouped_tensor(labels: torch.Tensor, hit_ids: torch.Tensor):
    if labels.numel() == 0:
        return None

    _, counts = torch.unique(labels, return_counts=True)
    grouped_order = torch.argsort(labels, stable=True)
    grouped_hit_ids = hit_ids[grouped_order]
    return GroupedTrackTensors(grouped_hit_ids, counts)


def tracks_to_lists(tracks) -> list[list[int]]:
    return [track.detach().cpu().tolist() if isinstance(track, torch.Tensor) else list(track) for track in tracks]


def get_simple_path(
    graph: Data,
    score_name: str,
    threshold: float,
    use_gpu: bool = False,
    cc_backend: str = "auto",
    return_tensors: bool = False,
) -> tuple[list[list[int]], Data]:
    graph = filter_graph(graph, score_name, threshold)
    if graph.edge_index.shape[1] == 0 or graph.num_nodes < 3:
        empty_tracks = [] if return_tensors else []
        return empty_tracks, graph
    graph = remove_cycles(graph)
    labels_int32 = None
    num_nodes = int(graph.num_nodes)

    if should_use_custom_cuda(graph.edge_index.device, use_gpu, cc_backend):
        labels_int32, num_components = connected_components_weak_int32(
            graph.edge_index[0], graph.edge_index[1], num_nodes
        )
        labels = labels_int32
    else:
        labels, num_components = compute_component_labels_from_backend(
            graph.edge_index, num_nodes, use_gpu=use_gpu, cc_backend=cc_backend
        )
    graph.labels = labels

    component_sizes = torch.bincount(labels.long(), minlength=num_components)
    large_component_mask = component_sizes[labels.long()] >= 3

    if labels_int32 is not None:
        large_simple_mask, complex_edge_mask = process_components_cuda(
            graph.edge_index[0],
            graph.edge_index[1],
            labels_int32,
            large_component_mask,
            int(graph.num_nodes),
            num_components,
        )
        subgraph_rest = edge_mask_subgraph(graph, complex_edge_mask, score_name)
        complex_node_mask = large_component_mask & ~large_simple_mask
    else:
        large_simple_mask, complex_edge_mask, subgraph_rest = process_components(
            graph, labels.long(), large_component_mask, num_components, score_name
        )
        complex_node_mask = large_component_mask & ~large_simple_mask

    if return_tensors:
        simple_path_lists = labels_to_grouped_tensor(
            labels.long()[large_simple_mask],
            graph.hit_id[large_simple_mask],
        )
    else:
        simple_path_lists = labels_to_lists(
            labels.long()[large_simple_mask],
            graph.hit_id[large_simple_mask],
        )

    if complex_node_mask.any():
        kept_node_mask = torch.zeros_like(complex_node_mask)
        kept_node_mask[graph.edge_index[0, complex_edge_mask]] = True
        kept_node_mask[graph.edge_index[1, complex_edge_mask]] = True
        cached_labels = labels.long()[kept_node_mask]
        _, cached_component_labels = torch.unique(
            cached_labels, sorted=True, return_inverse=True
        )
        subgraph_rest.cached_component_labels = cached_component_labels
        subgraph_rest.cached_num_components = (
            int(cached_component_labels.max().item()) + 1
            if cached_component_labels.numel() > 0
            else 0
        )

    if return_tensors:
        return ([] if simple_path_lists is None else [simple_path_lists]), subgraph_rest
    return tracks_to_lists(simple_path_lists), subgraph_rest


def process_components(
    graph: Data,
    labels: torch.Tensor,
    large_component_mask: torch.Tensor,
    num_components: int,
    score_name: str,
) -> tuple[torch.Tensor, torch.Tensor, Data]:
    in_degrees = torch.bincount(graph.edge_index[1], minlength=graph.num_nodes)
    out_degrees = torch.bincount(graph.edge_index[0], minlength=graph.num_nodes)

    bad_nodes = torch.maximum(in_degrees, out_degrees) > 1
    bad_components = scatter_max(
        bad_nodes.to(torch.int32), labels, dim=0, dim_size=num_components
    )[0].bool()
    simple_path_components = ~bad_components

    simple_path_mask = simple_path_components[labels]
    large_simple_mask = simple_path_mask & large_component_mask
    large_complex_mask = ~simple_path_mask & large_component_mask

    src, dst = graph.edge_index
    complex_edge_mask = large_complex_mask[src] & large_complex_mask[dst]
    subgraph_rest = edge_mask_subgraph(graph, complex_edge_mask, score_name)
    return large_simple_mask, complex_edge_mask, subgraph_rest


def dwalk(
    graph: Data,
    score_name: str,
    th_min: float,
    th_add: float,
    path_metrics: str = "score_weighted_length",
    use_gpu: bool = False,
    cc_backend: str = "auto",
    return_tensors: bool = False,
) -> list[list[int]]:
    graph = dp_max_add_cuts(graph, score_name, th_min, th_add)
    if graph.edge_index.shape[1] == 0 or graph.num_nodes < 3:
        return []

    dp_state = dp(graph, score_name, path_metrics)
    tracks = get_dp_tracks(
        graph,
        score_name,
        dp_state,
        path_metrics=path_metrics,
        use_gpu=use_gpu,
        cc_backend=cc_backend,
        return_grouped=return_tensors,
    )
    if return_tensors:
        return tracks
    return tracks_to_lists(tracks)


def dp_max_add_cuts(graph: Data, score_name: str, th_min: float, th_add: float) -> Data:
    edge_scores = graph[score_name]
    edge_index = graph.edge_index

    mask_add = edge_scores > th_add

    out_scores, out_argmax = scatter_max(edge_scores, edge_index[0], dim=0)
    outgoing_keep = torch.zeros_like(mask_add, dtype=torch.bool)
    outgoing_keep[out_argmax[out_scores >= th_min]] = True
    final_mask = outgoing_keep | mask_add

    in_scores, in_argmax = scatter_max(edge_scores, edge_index[1], dim=0)
    incoming_keep = torch.zeros_like(mask_add, dtype=torch.bool)
    incoming_keep[in_argmax[in_scores >= th_min]] = True
    final_mask = final_mask & (incoming_keep | mask_add)

    return edge_mask_subgraph(graph, final_mask, score_name)


def get_radial_order_key(graph: Data) -> torch.Tensor:
    if hasattr(graph, "R"):
        return graph.R
    if hasattr(graph, "hit_r") and hasattr(graph, "hit_z"):
        return graph.hit_r.square() + graph.hit_z.square()
    raise ValueError("D-WALK requires graph.R or graph.hit_r and graph.hit_z.")


def get_reverse_topological_order(graph: Data) -> torch.Tensor:
    if hasattr(graph, "_dp_topological_order"):
        cached_order = graph._dp_topological_order
        if int(cached_order.numel()) == int(graph.num_nodes):
            return cached_order

    order = torch.argsort(get_radial_order_key(graph), stable=True)
    graph._dp_topological_order = order.detach().cpu()
    return graph._dp_topological_order


def build_csr_graph_tensors(
    src: torch.Tensor,
    dst: torch.Tensor,
    weight: torch.Tensor,
    num_nodes: int,
) -> dict[str, torch.Tensor]:
    stable_order = torch.argsort(src, stable=True)
    sorted_src = src[stable_order]
    sorted_dst = dst[stable_order]
    sorted_weight = weight[stable_order]
    row_counts = torch.bincount(sorted_src, minlength=num_nodes)
    row_ptr = torch.empty(num_nodes + 1, dtype=torch.int32, device=src.device)
    row_ptr[0] = 0
    row_ptr[1:] = torch.cumsum(row_counts, dim=0).to(torch.int32)

    incoming_order = torch.argsort(dst, stable=True)
    incoming_dst = dst[incoming_order]
    incoming_src = src[incoming_order]
    incoming_counts = torch.bincount(incoming_dst, minlength=num_nodes)
    incoming_row_ptr = torch.empty(num_nodes + 1, dtype=torch.int32, device=src.device)
    incoming_row_ptr[0] = 0
    incoming_row_ptr[1:] = torch.cumsum(incoming_counts, dim=0).to(torch.int32)

    return {
        "row_ptr": row_ptr,
        "col_idx": sorted_dst.to(torch.int32),
        "edge_weight": sorted_weight.to(torch.float32),
        "incoming_row_ptr": incoming_row_ptr,
        "incoming_col_idx": incoming_src.to(torch.int32),
    }


def reverse_topological_dp_numpy(
    num_nodes: int,
    edge_starts: np.ndarray,
    sorted_dst: np.ndarray,
    sorted_weight: np.ndarray,
    topo_order: np.ndarray,
    active_nodes: np.ndarray,
    in_degree: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    best_score = np.zeros(num_nodes, dtype=sorted_weight.dtype)
    best_child = np.full(num_nodes, -1, dtype=np.int64)
    source_mask = np.zeros(num_nodes, dtype=np.bool_)

    for order_idx in range(topo_order.shape[0] - 1, -1, -1):
        node = topo_order[order_idx]
        if not active_nodes[node]:
            continue

        best_value = 0.0
        best_next = -1
        for edge_idx in range(edge_starts[node], edge_starts[node + 1]):
            child = sorted_dst[edge_idx]
            if not active_nodes[child]:
                continue

            candidate = sorted_weight[edge_idx] + best_score[child]
            if candidate > best_value:
                best_value = candidate
                best_next = child

        best_score[node] = best_value
        best_child[node] = best_next

    for node in range(num_nodes):
        if active_nodes[node] and in_degree[node] == 0 and best_score[node] > 0:
            source_mask[node] = True

    return best_score, best_child, source_mask


def run_reverse_topological_dp(
    src: torch.Tensor,
    dst: torch.Tensor,
    weight: torch.Tensor,
    num_nodes: int,
    topo_order: torch.Tensor,
    active_nodes: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    use_small_graph_fallback = (
        src.device.type == "cuda"
        and num_nodes <= SMALL_GRAPH_DP_NODE_THRESHOLD
        and int(weight.numel()) <= SMALL_GRAPH_DP_EDGE_THRESHOLD
    )

    if (
        src.device.type == "cuda"
        and reverse_topological_dp_available()
        and not use_small_graph_fallback
    ):
        csr_graph = build_csr_graph_tensors(src, dst, weight, num_nodes)
        topo_order_cuda = torch.as_tensor(topo_order, device=src.device, dtype=torch.int32)
        native_active_nodes = (
            None if active_nodes is None else active_nodes.to(device=src.device, dtype=torch.bool)
        )
        best_score, best_child, source_mask = reverse_topological_dp(
            csr_graph["row_ptr"],
            csr_graph["col_idx"],
            csr_graph["edge_weight"],
            csr_graph["incoming_row_ptr"],
            csr_graph["incoming_col_idx"],
            topo_order_cuda,
            native_active_nodes,
        )
        return {
            "best_score": best_score.to(dtype=weight.dtype),
            "best_child": best_child.to(dtype=torch.long),
            "source_mask": source_mask,
        }

    if src.numel() == 0:
        return {
            "best_score": torch.zeros(num_nodes, dtype=weight.dtype, device=weight.device),
            "best_child": torch.full((num_nodes,), -1, dtype=torch.long, device=src.device),
            "source_mask": torch.zeros(num_nodes, dtype=torch.bool, device=src.device),
        }

    src_cpu = src.detach().cpu().numpy().astype(np.int64, copy=False)
    dst_cpu = dst.detach().cpu().numpy().astype(np.int64, copy=False)
    weight_cpu = weight.detach().cpu().numpy()
    topo_order_cpu = np.asarray(topo_order, dtype=np.int64)
    active_nodes_cpu = (
        np.ones(num_nodes, dtype=np.bool_)
        if active_nodes is None
        else active_nodes.detach().cpu().numpy().astype(np.bool_, copy=False)
    )

    sort_order = np.argsort(src_cpu, kind="stable")
    sorted_src = src_cpu[sort_order]
    sorted_dst = dst_cpu[sort_order]
    sorted_weight = weight_cpu[sort_order]
    counts = np.bincount(sorted_src, minlength=num_nodes)
    edge_starts = np.zeros(num_nodes + 1, dtype=np.int64)
    edge_starts[1:] = np.cumsum(counts)
    in_degree = np.bincount(dst_cpu, minlength=num_nodes)

    best_score_cpu, best_child_cpu, source_mask_cpu = reverse_topological_dp_numpy(
        num_nodes,
        edge_starts,
        sorted_dst,
        sorted_weight,
        topo_order_cpu,
        active_nodes_cpu,
        in_degree,
    )

    return {
        "best_score": torch.from_numpy(best_score_cpu).to(device=weight.device, dtype=weight.dtype),
        "best_child": torch.from_numpy(best_child_cpu).to(device=src.device, dtype=torch.long),
        "source_mask": torch.from_numpy(source_mask_cpu).to(device=src.device, dtype=torch.bool),
    }


def dp(graph: Data, score_name: str, path_metrics: str = "score_weighted_length") -> dict[str, torch.Tensor]:
    src, dst = graph.edge_index
    weight = get_dwalk_metric_weights(graph, score_name, path_metrics)
    return run_reverse_topological_dp(
        src,
        dst,
        weight,
        int(graph.num_nodes),
        topo_order=get_reverse_topological_order(graph),
    )


def compute_component_labels(
    graph: Data,
    use_gpu: bool = False,
    cc_backend: str = "auto",
) -> tuple[torch.Tensor, int]:
    num_nodes = int(graph.num_nodes)
    num_edges = int(graph.edge_index.shape[1])
    if use_gpu and cc_backend == "custom_cuda" and num_nodes <= 10000 and num_edges <= 10000:
        return compute_component_labels_from_backend(
            graph.edge_index, num_nodes, use_gpu=False, cc_backend="scipy"
        )
    return compute_component_labels_from_backend(
        graph.edge_index, num_nodes, use_gpu=use_gpu, cc_backend=cc_backend
    )


def select_component_roots(
    dp_state: dict[str, torch.Tensor],
    component_labels: torch.Tensor,
    num_components: int,
    path_metrics: str = "score_weighted_length",
) -> torch.Tensor:
    source_mask = dp_state["source_mask"]
    component_index = component_labels if component_labels.dtype == torch.long else component_labels.long()
    masked_scores = dp_state["best_score"].clone()
    masked_scores.masked_fill_(~source_mask, torch.finfo(masked_scores.dtype).min)
    component_scores, root_nodes = scatter_max(
        masked_scores,
        component_index,
        dim=0,
        dim_size=num_components,
    )
    valid_components = torch.nonzero(
        component_scores > get_dwalk_min_root_score(path_metrics), as_tuple=False
    ).flatten()
    return root_nodes[valid_components]


def trace_selected_paths_python(
    best_child: torch.Tensor,
    selected_roots: torch.Tensor,
    num_nodes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_paths = selected_roots.shape[0]
    if num_paths == 0:
        return selected_roots, selected_roots

    current_nodes = selected_roots.clone()
    path_ids = torch.arange(num_paths, device=selected_roots.device, dtype=torch.long)
    selected_nodes = torch.empty(num_nodes, dtype=selected_roots.dtype, device=selected_roots.device)
    selected_track_labels = torch.empty(num_nodes, dtype=torch.long, device=selected_roots.device)
    num_selected = 0
    step = 0

    while current_nodes.numel() > 0 and step < num_nodes:
        next_num_selected = num_selected + current_nodes.numel()
        selected_nodes[num_selected:next_num_selected] = current_nodes
        selected_track_labels[num_selected:next_num_selected] = path_ids
        num_selected = next_num_selected

        next_nodes = best_child[current_nodes]
        active_mask = next_nodes != -1
        current_nodes = next_nodes[active_mask]
        path_ids = path_ids[active_mask]
        step += 1

    return selected_track_labels[:num_selected], selected_nodes[:num_selected]


def trace_selected_paths(
    best_child: torch.Tensor,
    selected_roots: torch.Tensor,
    num_nodes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if best_child.device.type == "cuda" and trace_selected_paths_available():
        return trace_selected_paths_cuda(best_child, selected_roots, num_nodes)
    return trace_selected_paths_python(best_child, selected_roots, num_nodes)


def convert_paths_to_hit_ids(
    hit_id: torch.Tensor,
    selected_nodes: torch.Tensor,
    selected_track_labels: torch.Tensor,
    selected_roots: torch.Tensor,
    return_grouped: bool = False,
) -> tuple[torch.Tensor, ...]:
    num_tracks = int(selected_roots.numel())
    if num_tracks == 0 or selected_nodes.numel() == 0:
        return ()

    selected_hit_ids = hit_id[selected_nodes]
    grouped_order = torch.argsort(selected_track_labels, stable=True)
    grouped_hit_ids = selected_hit_ids[grouped_order]
    grouped_track_labels = selected_track_labels[grouped_order]
    track_sizes = torch.bincount(grouped_track_labels, minlength=num_tracks)
    if return_grouped:
        return GroupedTrackTensors(grouped_hit_ids, track_sizes[track_sizes > 0])
    non_empty_track_sizes = track_sizes[track_sizes > 0].detach().cpu().tolist()
    return torch.split(grouped_hit_ids, non_empty_track_sizes)


def update_active_edges(
    graph: Data,
    selected_node_mask: torch.Tensor,
    score_name: str,
) -> Data:
    if selected_node_mask.numel() == 0 or not selected_node_mask.any():
        return graph

    keep_node_mask = ~selected_node_mask
    keep_node_ids = torch.nonzero(keep_node_mask, as_tuple=False).flatten()
    num_nodes = int(keep_node_ids.numel())
    if num_nodes == int(graph.num_nodes):
        return graph

    src, dst = graph.edge_index
    keep_edge_mask = keep_node_mask[src] & keep_node_mask[dst]
    new_node_ids = torch.full(
        (selected_node_mask.numel(),),
        -1,
        dtype=torch.long,
        device=selected_node_mask.device,
    )
    new_node_ids[keep_node_ids] = torch.arange(num_nodes, device=selected_node_mask.device)

    updated_graph = Data(
        edge_index=torch.stack(
            (new_node_ids[src[keep_edge_mask]], new_node_ids[dst[keep_edge_mask]]),
            dim=0,
        ),
        num_nodes=num_nodes,
    )
    for key in ("hit_id", "R", "hit_r", "hit_z"):
        if hasattr(graph, key):
            updated_graph[key] = graph[key][keep_node_ids]
    updated_graph[score_name] = graph[score_name][keep_edge_mask]
    return updated_graph


def get_dp_tracks(
    graph: Data,
    score_name: str,
    dp_state: dict[str, torch.Tensor],
    path_metrics: str = "score_weighted_length",
    use_gpu: bool = False,
    cc_backend: str = "auto",
    return_grouped: bool = False,
) -> list[torch.Tensor]:
    tracks = []
    current_graph = graph
    current_state = dp_state
    use_cached_components_once = hasattr(graph, "cached_component_labels")

    while True:
        if use_cached_components_once:
            use_cached_components_once = False
            component_labels = current_graph.cached_component_labels
            num_components = current_graph.cached_num_components
        else:
            component_labels, num_components = compute_component_labels(
                current_graph, use_gpu=use_gpu, cc_backend=cc_backend
            )

        selected_roots = select_component_roots(
            current_state, component_labels, num_components, path_metrics=path_metrics
        )
        if selected_roots.numel() == 0:
            break

        selected_track_labels, selected_nodes = trace_selected_paths(
            current_state["best_child"], selected_roots, int(current_graph.num_nodes)
        )
        new_tracks = convert_paths_to_hit_ids(
            current_graph.hit_id.long(),
            selected_nodes,
            selected_track_labels,
            selected_roots,
            return_grouped=return_grouped,
        )
        if return_grouped and new_tracks:
            tracks.append(new_tracks)
        else:
            tracks.extend(new_tracks)

        selected_node_mask = torch.zeros(
            int(current_graph.num_nodes),
            dtype=torch.bool,
            device=current_graph.edge_index.device,
        )
        selected_node_mask[selected_nodes] = True
        updated_graph = update_active_edges(current_graph, selected_node_mask, score_name)
        if updated_graph.num_nodes == current_graph.num_nodes:
            break

        current_graph = updated_graph
        current_state = dp(current_graph, score_name, path_metrics)

    return tracks
