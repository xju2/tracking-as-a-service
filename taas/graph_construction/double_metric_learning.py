"""Double-metric-learning graph construction."""

from __future__ import annotations

from typing import Callable, Tuple

import torch

from taas.graph import TrackingGraph
from taas.graph_construction._frnn import build_edges
from taas.graph_construction.base import GraphConstructor


class DoubleMetricLearningGraphConstructor(GraphConstructor):
    """Build cross-embedding-space FRNN graphs."""

    def __init__(
        self,
        embed: Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
        r_max: float,
        k_max: int,
    ) -> None:
        self.embed = embed
        self.r_max = r_max
        self.k_max = k_max

    def build(self, features: torch.Tensor) -> TrackingGraph:
        query, database = self.embed(features)
        edge_index = self.build_edges(query, database, self.r_max, self.k_max)
        return TrackingGraph(node_features=features, edge_index=edge_index)

    @staticmethod
    def build_edges(
        query: torch.Tensor,
        database: torch.Tensor,
        r_max: float = 0.1,
        k_max: int = 1000,
    ) -> torch.Tensor:
        return build_edges(query, database, r_max=r_max, k_max=k_max)
