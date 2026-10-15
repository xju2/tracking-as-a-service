"""Shared Torch edge-filter runners with explicit compatibility profiles."""

from __future__ import annotations

from enum import Enum
from typing import Callable

import torch
from torch_geometric.utils import sort_edge_index

from taas.edge_classification.base import EdgeClassifier


class FilterProfile(str, Enum):
    METRIC_LEGACY = "metric_legacy"
    METRIC_CURRENT = "metric_current"
    DOUBLE_METRIC_LEGACY = "double_metric_legacy"
    DOUBLE_METRIC_CURRENT = "double_metric_current"


_DTYPE = torch.float16


def _supports_autocast(device_type: str) -> bool:
    if torch.__version__ < "2.4.0":
        return device_type == "cuda"
    return torch.amp.autocast_mode.is_autocast_available(device_type)


def _clone_output(output):
    if isinstance(output, tuple):
        return tuple(item.clone() for item in output)
    return output.clone()


def run_torch_model(
    model: torch.nn.Module,
    auto_cast: bool,
    *inputs,
    profile: FilterProfile = FilterProfile.METRIC_CURRENT,
):
    """Run a Torch model while retaining each historical execution policy."""
    if not inputs:
        raise AssertionError("At least one input must be provided.")

    legacy_metric = profile is FilterProfile.METRIC_LEGACY
    hardcoded_device = profile in {
        FilterProfile.METRIC_CURRENT,
        FilterProfile.DOUBLE_METRIC_LEGACY,
    }
    clone_output = not legacy_metric
    device_type = "cuda:0" if hardcoded_device else inputs[0].device.type
    context = torch.no_grad if legacy_metric else torch.inference_mode

    with context():
        if auto_cast and _supports_autocast(device_type):
            with torch.autocast(device_type, dtype=_DTYPE):
                output = model(*inputs)
        else:
            output = model(*inputs)
    return _clone_output(output) if clone_output else output


def _filter_policy(profile: FilterProfile, batches: int):
    if profile is FilterProfile.METRIC_LEGACY:
        return True, batches, False, torch.no_grad
    if profile in {
        FilterProfile.METRIC_CURRENT,
        FilterProfile.DOUBLE_METRIC_LEGACY,
    }:
        return True, 4, True, torch.inference_mode
    return False, batches, True, torch.inference_mode


def run_gnn_filter(
    model: torch.nn.Module,
    auto_cast: bool,
    batches: int,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    *,
    profile: FilterProfile = FilterProfile.METRIC_CURRENT,
):
    """Run the GNN filter using the selected version-compatible behavior."""
    sort_edges, effective_batches, clone_output, context = _filter_policy(profile, batches)
    active_edges = sort_edge_index(edge_index, sort_by_row=False) if sort_edges else edge_index

    with context():
        device_type = x.device.type
        with torch.autocast(
            device_type,
            dtype=_DTYPE,
            enabled=auto_cast and _supports_autocast(device_type),
        ):
            embedding = model.gnn(x, active_edges)
            if clone_output:
                embedding = embedding.clone()
            score_parts = []
            for subset in torch.tensor_split(active_edges, effective_batches, dim=1):
                scores = model.net(
                    torch.cat([embedding[subset[0]], embedding[subset[1]]], dim=-1)
                ).squeeze(-1)
                score_parts.append(scores.clone() if clone_output else scores)

    return torch.cat(score_parts).sigmoid(), active_edges, embedding


def run_gnn_filter_optimized(
    model: torch.nn.Module,
    auto_cast: bool,
    batches: int,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    *,
    profile: FilterProfile = FilterProfile.METRIC_CURRENT,
):
    """Run the historical 65,536-edge chunked filter implementation."""
    del batches, profile
    chunk_size = 2**16
    with torch.inference_mode():
        active_edges = sort_edge_index(edge_index, sort_by_row=False)
        device_type = x.device.type if hasattr(x, "device") else str(x.device)
        score_parts = []
        with torch.autocast(
            device_type,
            dtype=_DTYPE,
            enabled=auto_cast and _supports_autocast(device_type),
        ):
            embedding = model.gnn(x, active_edges)
            for start in range(0, active_edges.shape[1], chunk_size):
                edge_batch = active_edges[:, start : start + chunk_size]
                edge_features = torch.cat(
                    [embedding[edge_batch[0]], embedding[edge_batch[1]]], dim=-1
                )
                score_parts.append(model.net(edge_features).squeeze(-1))

    return torch.cat(score_parts).sigmoid(), active_edges, embedding.clone()


class TorchEdgeFilter(EdgeClassifier):
    def __init__(self, runner: Callable[..., torch.Tensor]) -> None:
        self._runner = runner

    def score(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if edge_features is None:
            return self._runner(node_features, edge_index)
        return self._runner(node_features, edge_index, edge_features)
