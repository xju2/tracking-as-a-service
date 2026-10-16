"""Interfaces for graph-construction stages."""

from abc import ABC, abstractmethod

import torch

from taas.graph import TrackingGraph


class GraphConstructor(ABC):
    @abstractmethod
    def build(self, features: torch.Tensor) -> TrackingGraph:
        """Build a graph without moving tensors away from their input device."""
