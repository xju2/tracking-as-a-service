from __future__ import annotations

import torch
from torch_geometric.data import Data

from .dwalk import GroupedTrackTensors


def _load_edge_layer_connector_cuda():
    try:
        from taas_cuda_ext import _edge_layer_connector_cuda
    except ImportError as exc:
        raise RuntimeError(
            "TAAS EdgeLayerConnector CUDA extension is not available. "
            "Start Triton with `scripts/start-tritonserver.sh -b` or run "
            "`./scripts/build_taas_cuda_ext.sh` in the target container."
        ) from exc
    return _edge_layer_connector_cuda


def edge_layer_connector(
    graph: Data,
    score_name: str,
    weights_cut: float = 0.01,
    min_hits: int = 3,
    block_size: int = 512,
    max_hits_per_track: int = 30,
):
    if graph.edge_index.numel() == 0:
        return []
    if not graph.edge_index.is_cuda:
        raise RuntimeError("EdgeLayerConnector requires CUDA graph tensors")

    if not hasattr(graph, "hit_id"):
        graph.hit_id = torch.arange(
            int(graph.num_nodes),
            device=graph.edge_index.device,
            dtype=torch.long,
        )

    ext = _load_edge_layer_connector_cuda()
    device = graph.edge_index.device
    hit_ids = graph.hit_id.to(device=device, dtype=torch.int32).contiguous()
    edge_src = graph.edge_index[0].to(device=device, dtype=torch.int32).contiguous()
    edge_dst = graph.edge_index[1].to(device=device, dtype=torch.int32).contiguous()
    edge_scores = graph[score_name].to(device=device, dtype=torch.float32).contiguous()

    padded_hits, counts, _ = ext.edge_layer_connector_cuda(
        hit_ids,
        edge_src,
        edge_dst,
        edge_scores,
        float(weights_cut),
        int(min_hits),
        int(block_size),
        int(max_hits_per_track),
    )
    if counts.numel() == 0:
        return []

    counts = counts.long()
    num_tracks = int(counts.numel())
    if padded_hits.numel() % num_tracks != 0:
        raise RuntimeError(
            "EdgeLayerConnector returned an inconsistent padded track tensor: "
            f"{padded_hits.numel()} hits for {num_tracks} tracks"
        )
    padded_width = int(padded_hits.numel() // num_tracks)
    if counts.numel() and int(counts.max().item()) > padded_width:
        raise RuntimeError(
            "EdgeLayerConnector returned a track count larger than its padded "
            f"width: max count {int(counts.max().item())}, width {padded_width}"
        )
    padded_hits = padded_hits.long().view(num_tracks, padded_width)
    hit_mask = (
        torch.arange(padded_width, device=device).unsqueeze(0)
        < counts.unsqueeze(1)
    )
    grouped_hit_ids = padded_hits[hit_mask]
    if grouped_hit_ids.numel() == 0:
        return []
    return [GroupedTrackTensors(grouped_hit_ids, counts)]
