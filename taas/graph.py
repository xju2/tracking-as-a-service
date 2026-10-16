"""Common graph data exchanged between pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class TrackingGraph:
    """Tensor-only graph representation that preserves device placement."""

    node_features: torch.Tensor
    edge_index: torch.Tensor
    edge_features: Optional[torch.Tensor] = None
    edge_scores: Optional[torch.Tensor] = None
    hit_ids: Optional[torch.Tensor] = None

    @property
    def device(self) -> torch.device:
        return self.node_features.device

    @property
    def num_nodes(self) -> int:
        return int(self.node_features.shape[0])

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])
