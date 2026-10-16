"""Metric-learning graph construction."""

from __future__ import annotations

from typing import Callable

import torch

from taas.graph import TrackingGraph
from taas.graph_construction._frnn import build_edges
from taas.graph_construction.base import GraphConstructor


class MetricLearningGraphConstructor(GraphConstructor):
    """Build same-embedding-space FRNN graphs with the established TAAS semantics."""

    def __init__(
        self,
        embed: Callable[[torch.Tensor], torch.Tensor],
        r_max: float,
        k_max: int,
    ) -> None:
        self.embed = embed
        self.r_max = r_max
        self.k_max = k_max

    def build(self, features: torch.Tensor) -> TrackingGraph:
        embedding = self.embed(features)
        edge_index = self.build_edges(embedding, embedding, self.r_max, self.k_max)
        return TrackingGraph(node_features=features, edge_index=edge_index)

    @staticmethod
    def build_edges(
        query: torch.Tensor,
        database: torch.Tensor,
        r_max: float = 0.1,
        k_max: int = 1000,
    ) -> torch.Tensor:
        return build_edges(query, database, r_max=r_max, k_max=k_max)
