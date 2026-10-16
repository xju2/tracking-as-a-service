"""Torch InteractionGNN adapter shared by ML, DML, and GNN4Pixel."""

from __future__ import annotations

from typing import Callable

import torch

from taas.edge_classification.base import EdgeClassifier


class InteractionGNNTorch(EdgeClassifier):
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
