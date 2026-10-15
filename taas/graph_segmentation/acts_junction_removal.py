from __future__ import annotations

import torch
from torch_geometric.data import Data
from torch_scatter import scatter_max

from .dwalk import GroupedTrackTensors, compute_component_labels_from_backend


def _labels_to_grouped_tracks(
    labels: torch.Tensor,
    hit_ids: torch.Tensor,
    min_candidate_size: int,
):
    if labels.numel() == 0:
        return []

    valid_mask = labels >= 0
    if not valid_mask.any():
        return []

    valid_labels = labels[valid_mask].long()
    valid_hit_ids = hit_ids[valid_mask].long()
    unique_labels, inverse = torch.unique(valid_labels, sorted=True, return_inverse=True)
    if unique_labels.numel() == 0:
        return []

    counts = torch.bincount(inverse, minlength=unique_labels.numel())
    keep_components = counts >= min_candidate_size
    if not keep_components.any():
        return []

    keep_hits = keep_components[inverse]
    filtered_inverse = inverse[keep_hits]
    filtered_hit_ids = valid_hit_ids[keep_hits]
    filtered_counts = torch.bincount(
        filtered_inverse,
        minlength=unique_labels.numel(),
    )[keep_components]

    grouped_order = torch.argsort(filtered_inverse, stable=True)
    grouped_hit_ids = filtered_hit_ids[grouped_order]
    return [GroupedTrackTensors(grouped_hit_ids, filtered_counts)]


def acts_junction_removal(
    graph: Data,
    score_name: str,
    min_candidate_size: int = 3,
    use_gpu: bool = False,
    cc_backend: str = "auto",
):
    """ACTS-style junction removal followed by connected components.

    ACTS removes all but the highest-score incoming edge at every node with two
    or more incoming edges, and all but the highest-score outgoing edge at every
    node with two or more outgoing edges. Track candidates smaller than
    ``min_candidate_size`` are then discarded after connected components.
    """

    if not hasattr(graph, "hit_id"):
        graph.hit_id = torch.arange(
            int(graph.num_nodes),
            device=graph.edge_index.device,
            dtype=torch.long,
        )

    edge_index = graph.edge_index
    edge_scores = graph[score_name]
    num_nodes = int(graph.num_nodes)

    if edge_index.numel() == 0:
        return []

    src = edge_index[0].long()
    dst = edge_index[1].long()
    num_edges = edge_index.size(1)
    device = edge_index.device

    out_degree = torch.bincount(src, minlength=num_nodes)
    in_degree = torch.bincount(dst, minlength=num_nodes)

    keep_out = torch.ones(num_edges, dtype=torch.bool, device=device)
    out_junction_nodes = out_degree >= 2
    out_junction_edges = out_junction_nodes[src]
    if out_junction_edges.any():
        keep_out[out_junction_edges] = False
        out_argmax = scatter_max(edge_scores, src, dim=0, dim_size=num_nodes)[1]
        out_argmax = out_argmax[out_junction_nodes]
        out_argmax = out_argmax[(out_argmax >= 0) & (out_argmax < num_edges)]
        if out_argmax.numel() > 0:
            keep_out[out_argmax.long()] = True

    keep_in = torch.ones(num_edges, dtype=torch.bool, device=device)
    in_junction_nodes = in_degree >= 2
    in_junction_edges = in_junction_nodes[dst]
    if in_junction_edges.any():
        keep_in[in_junction_edges] = False
        in_argmax = scatter_max(edge_scores, dst, dim=0, dim_size=num_nodes)[1]
        in_argmax = in_argmax[in_junction_nodes]
        in_argmax = in_argmax[(in_argmax >= 0) & (in_argmax < num_edges)]
        if in_argmax.numel() > 0:
            keep_in[in_argmax.long()] = True

    pruned_edges = edge_index[:, keep_in & keep_out]
    if pruned_edges.numel() == 0:
        return []

    labels, _ = compute_component_labels_from_backend(
        pruned_edges,
        num_nodes,
        use_gpu=use_gpu,
        cc_backend=cc_backend,
    )
    return _labels_to_grouped_tracks(
        labels.long(),
        graph.hit_id,
        min_candidate_size=max(int(min_candidate_size), 1),
    )
