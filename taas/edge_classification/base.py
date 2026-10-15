"""Interfaces for edge-classification stages."""

from abc import ABC, abstractmethod

import torch


class EdgeClassifier(ABC):
    @abstractmethod
    def score(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return one score per edge on the active tensor device."""
