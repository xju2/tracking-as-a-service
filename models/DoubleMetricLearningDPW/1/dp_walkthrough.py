from __future__ import annotations

import torch
from torch_geometric.data import Data
from torch_geometric.transforms import RemoveIsolatedNodes
from torch_geometric.utils import to_scipy_sparse_matrix
from torch_scatter import scatter_max


def dp_walk_through(
    graph: Data,
    score_name: str,
    th_min: float,
    th_add: float,
    path_metrics: str = "score_weighted_length",
):
    if path_metrics != "score_weighted_length":
        raise ValueError(f"Unsupported DP walkthrough path metric: {path_metrics}")

    graph = dp_max_add_cuts(graph, score_name, th_min, th_add)
    if graph.edge_index.shape[1] == 0 or graph.num_nodes < 3:
        return []

    dp_state = run_dp(graph, score_name)
    return get_dp_tracks(graph, score_name, dp_state)


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

    subgraph = graph.edge_subgraph(final_mask)
    subgraph = RemoveIsolatedNodes()(subgraph)
    return subgraph


def run_dp(graph: Data, score_name: str) -> dict[str, torch.Tensor]:
    src, dst = graph.edge_index
    weight = graph[score_name]
    num_nodes = int(graph.num_nodes)

    in_degree, out_degree = get_active_degrees(src, dst, num_nodes)
    remaining_out_degree = out_degree.clone()
    best_score = torch.zeros(num_nodes, dtype=weight.dtype, device=weight.device)
    best_child = torch.full((num_nodes,), -1, dtype=torch.long, device=src.device)

    current = remaining_out_degree == 0
    processed = current.clone()

    while True:
        current_edge_mask = current[dst]
        if current_edge_mask.any():
            released_parents = torch.bincount(
                src[current_edge_mask], minlength=num_nodes
            ).to(remaining_out_degree.dtype)
            remaining_out_degree = remaining_out_degree - released_parents

        ready_parents = (remaining_out_degree == 0) & (~processed) & (out_degree > 0)
        if not ready_parents.any():
            break

        parent_edge_indices = torch.nonzero(ready_parents[src], as_tuple=False).flatten()
        candidate_scores = weight[parent_edge_indices] + best_score[dst[parent_edge_indices]]
        parent_scores, parent_argmax = scatter_max(
            candidate_scores,
            src[parent_edge_indices],
            dim=0,
            dim_size=num_nodes,
        )

        ready_parent_indices = torch.nonzero(ready_parents, as_tuple=False).flatten()
        valid_ready = parent_argmax[ready_parent_indices] >= 0
        ready_parent_indices = ready_parent_indices[valid_ready]
        if ready_parent_indices.numel() > 0:
            best_score[ready_parent_indices] = parent_scores[ready_parent_indices]
            chosen_edge_indices = parent_edge_indices[parent_argmax[ready_parent_indices]]
            best_child[ready_parent_indices] = dst[chosen_edge_indices]

        processed |= ready_parents
        current = ready_parents

    source_mask = (in_degree == 0) & (best_score > 0)
    return {
        "best_score": best_score,
        "best_child": best_child,
        "source_mask": source_mask,
    }


def get_active_degrees(
    src: torch.Tensor, dst: torch.Tensor, num_nodes: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if src.numel() > 0:
        in_degree = torch.bincount(dst, minlength=num_nodes)
        out_degree = torch.bincount(src, minlength=num_nodes)
    else:
        in_degree = torch.zeros(num_nodes, dtype=torch.long, device=src.device)
        out_degree = torch.zeros(num_nodes, dtype=torch.long, device=src.device)
    return in_degree, out_degree


def get_dp_tracks(
    graph: Data, score_name: str, dp_state: dict[str, torch.Tensor]
) -> list[list[int]]:
    tracks = []
    current_graph = graph
    current_state = dp_state

    while True:
        component_labels, num_components = compute_component_labels(current_graph)
        selected_roots = select_component_roots(current_state, component_labels, num_components)
        if selected_roots.numel() == 0:
            break

        track_labels = trace_selected_paths(
            current_state["best_child"], selected_roots, int(current_graph.num_nodes)
        )
        selected_node_mask = track_labels != -1
        tracks.extend(convert_paths_to_hit_ids(current_graph.hit_id.long(), track_labels))

        updated_graph = update_active_edges(current_graph, selected_node_mask)
        if updated_graph.num_nodes == current_graph.num_nodes:
            break

        current_graph = updated_graph
        current_state = run_dp(current_graph, score_name)

    return tracks


def compute_component_labels(graph: Data) -> tuple[torch.Tensor, int]:
    from scipy.sparse.csgraph import connected_components

    adj_matrix = to_scipy_sparse_matrix(graph.edge_index.detach().cpu(), num_nodes=graph.num_nodes)
    num_components, labels = connected_components(
        csgraph=adj_matrix,
        directed=True,
        connection="weak",
    )
    labels = torch.from_numpy(labels).long().to(graph.edge_index.device)
    return labels, num_components


def select_component_roots(
    dp_state: dict[str, torch.Tensor],
    component_labels: torch.Tensor,
    num_components: int,
) -> torch.Tensor:
    source_nodes = torch.nonzero(dp_state["source_mask"], as_tuple=False).flatten()
    if source_nodes.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=component_labels.device)

    source_scores = dp_state["best_score"][source_nodes]
    source_components = component_labels[source_nodes].long()
    component_scores, root_argmax = scatter_max(
        source_scores,
        source_components,
        dim=0,
        dim_size=num_components,
    )
    valid_components = torch.nonzero(
        (component_scores > 0) & (root_argmax < source_nodes.numel()),
        as_tuple=False,
    ).flatten()
    return source_nodes[root_argmax[valid_components]]


def trace_selected_paths(
    best_child: torch.Tensor,
    selected_roots: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    num_paths = selected_roots.shape[0]
    track_labels = torch.full(
        (num_nodes,),
        -1,
        dtype=torch.long,
        device=selected_roots.device,
    )
    if num_paths == 0:
        return track_labels

    current_nodes = selected_roots.clone()
    path_ids = torch.arange(num_paths, device=selected_roots.device, dtype=torch.long)
    active_paths = torch.ones(num_paths, dtype=torch.bool, device=selected_roots.device)
    step = 0

    while active_paths.any() and step < num_nodes:
        active_nodes = current_nodes[active_paths]
        active_path_ids = path_ids[active_paths]
        track_labels[active_nodes] = active_path_ids

        next_nodes = torch.full_like(current_nodes, -1)
        next_nodes[active_paths] = best_child[active_nodes]
        current_nodes = next_nodes
        active_paths = current_nodes != -1
        step += 1

    return track_labels


def convert_paths_to_hit_ids(
    hit_id: torch.Tensor, track_labels: torch.Tensor
) -> list[list[int]]:
    selected_node_mask = track_labels != -1
    if not selected_node_mask.any():
        return []

    selected_hit_ids = hit_id[selected_node_mask]
    selected_track_labels = track_labels[selected_node_mask]

    grouped_order = torch.argsort(selected_track_labels, stable=True)
    grouped_hit_ids = selected_hit_ids[grouped_order].detach().cpu()
    grouped_track_labels = selected_track_labels[grouped_order]

    track_sizes = torch.bincount(grouped_track_labels)
    non_empty_track_sizes = track_sizes[track_sizes > 0].detach().cpu().tolist()
    grouped_tracks = torch.split(grouped_hit_ids, non_empty_track_sizes)
    return [track.tolist() for track in grouped_tracks]


def update_active_edges(graph: Data, selected_node_mask: torch.Tensor) -> Data:
    if selected_node_mask.numel() == 0 or not selected_node_mask.any():
        return graph
    return graph.subgraph(~selected_node_mask)