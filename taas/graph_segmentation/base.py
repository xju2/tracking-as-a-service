"""Object-oriented graph-segmentation interfaces."""

from abc import ABC, abstractmethod
from typing import Any


class GraphSegmenter(ABC):
    @abstractmethod
    def segment(self, graph: Any):
        """Convert a scored graph into track candidates."""


class FunctionalGraphSegmenter(GraphSegmenter):
    """Bind an existing tested segmentation function and its fixed parameters."""

    def __init__(self, implementation, *args, **kwargs):
        self.implementation = implementation
        self.args = args
        self.kwargs = kwargs

    def segment(self, graph):
        return self.implementation(graph, *self.args, **self.kwargs)
