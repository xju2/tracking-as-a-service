from __future__ import annotations

import torch
from torch_geometric.data import Data
from torch_scatter import scatter_max

from .dwalk import (
    GroupedTrackTensors,
    compute_component_labels_from_backend,
    remove_cycles,
)


def compute_component_labels(graph: Data, use_gpu: bool = False, cc_backend: str = "auto"):
    return compute_component_labels_from_backend(
        graph.edge_index,
        int(graph.num_nodes),
        use_gpu=use_gpu,
        cc_backend=cc_backend,
    )


def labels_to_grouped_tracks(labels: torch.Tensor, hit_ids: torch.Tensor):
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
    grouped_order = torch.argsort(inverse, stable=True)
    grouped_hit_ids = valid_hit_ids[grouped_order]
    return [GroupedTrackTensors(grouped_hit_ids, counts)]


def junction_removal(
    graph: Data,
    score_name: str,
    score_cut: float = 0.9,
    junction_cut: float = 0.2,
    min_chain_length: int = 5,
    skip_remove_cycles: bool = True,
    use_gpu: bool = False,
    cc_backend: str = "auto",
):
    if not hasattr(graph, "hit_id"):
        graph.hit_id = torch.arange(
            int(graph.num_nodes),
            device=graph.edge_index.device,
            dtype=torch.long,
        )

    working_graph = graph if skip_remove_cycles else remove_cycles(graph)
    device = working_graph.edge_index.device
    num_hits = int(working_graph.num_nodes)

    track_id = torch.full(
        (num_hits,),
        -1,
        dtype=torch.long,
        device=device,
    )
    to_keep = torch.ones(num_hits, dtype=torch.bool, device=device)

    edge_scores = working_graph[score_name]
    chain_edges = working_graph.edge_index[:, edge_scores > score_cut]

    if chain_edges.numel() > 0:
        chain_nodes = torch.unique(chain_edges)
        chain_node_map = torch.full(
            (num_hits,),
            -1,
            dtype=torch.long,
            device=device,
        )
        chain_node_map[chain_nodes] = torch.arange(
            chain_nodes.numel(),
            device=device,
            dtype=torch.long,
        )
        compact_chain_edges = chain_node_map[chain_edges]
        compact_track_id, num_components = compute_component_labels(
            Data(edge_index=compact_chain_edges, num_nodes=int(chain_nodes.numel())),
            use_gpu=use_gpu,
            cc_backend=cc_backend,
        )

        out_degree = torch.bincount(compact_chain_edges[0], minlength=chain_nodes.numel())
        in_degree = torch.bincount(compact_chain_edges[1], minlength=chain_nodes.numel())

        tracks_to_remove = torch.zeros(
            num_components,
            dtype=torch.bool,
            device=compact_track_id.device,
        )
        tracks_to_remove[compact_track_id[in_degree > 1].long()] = True
        tracks_to_remove[compact_track_id[out_degree > 1].long()] = True
        nhits = torch.bincount(compact_track_id.long(), minlength=num_components)
        tracks_to_remove |= nhits < min_chain_length

        compact_track_id = compact_track_id.long()
        compact_track_id[tracks_to_remove[compact_track_id]] = -1
        track_id[chain_nodes] = compact_track_id.to(device=device)

    to_keep[track_id >= 0] = False

    junction_mask = edge_scores > junction_cut
    junction_edges = working_graph.edge_index[:, junction_mask]
    junction_score = edge_scores[junction_mask]

    if junction_edges.numel() > 0:
        unresolved_edge_mask = to_keep[junction_edges[0]] & to_keep[junction_edges[1]]
        junction_edges = junction_edges[:, unresolved_edge_mask]
        junction_score = junction_score[unresolved_edge_mask]

    if junction_edges.numel() > 0:
        in_max = scatter_max(junction_score, junction_edges[1])[1]
        in_max = in_max[in_max < junction_edges.size(1)]
        out_max = scatter_max(junction_score, junction_edges[0])[1]
        out_max = out_max[out_max < junction_edges.size(1)]
        in_keep_mask = torch.zeros_like(junction_edges[0], dtype=torch.bool)
        out_keep_mask = torch.zeros_like(junction_edges[0], dtype=torch.bool)
        if in_max.numel() > 0:
            in_keep_mask[in_max] = True
        if out_max.numel() > 0:
            out_keep_mask[out_max] = True
        junction_edges = junction_edges[:, in_keep_mask & out_keep_mask]

    unresolved_nodes = torch.nonzero(to_keep, as_tuple=False).flatten()
    if unresolved_nodes.numel() > 0:
        unresolved_node_map = torch.full(
            (num_hits,),
            -1,
            dtype=torch.long,
            device=device,
        )
        unresolved_node_map[unresolved_nodes] = torch.arange(
            unresolved_nodes.numel(),
            device=device,
            dtype=torch.long,
        )
        compact_junction_edges = unresolved_node_map[junction_edges]
        labels, num_components = compute_component_labels(
            Data(
                edge_index=compact_junction_edges,
                num_nodes=int(unresolved_nodes.numel()),
            ),
            use_gpu=use_gpu,
            cc_backend=cc_backend,
        )
        labels = labels.long()
        if labels.numel() > 0:
            nhits = torch.bincount(labels, minlength=num_components)
            keep_unresolved_mask = nhits[labels] > 1
            if keep_unresolved_mask.any():
                next_track_id = torch.clamp_min(track_id.max() + 1, 0)
                track_id[unresolved_nodes[keep_unresolved_mask]] = (
                    labels[keep_unresolved_mask].to(device=device) + next_track_id
                )

    return labels_to_grouped_tracks(track_id, working_graph.hit_id)
