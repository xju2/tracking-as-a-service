"""ModuleMap CUDA graph-construction adapter."""

from __future__ import annotations

import torch

from taas.cuda_ext.module_map_graph import ModuleMapGraphCuda
from taas.graph import TrackingGraph
from taas.graph_construction.base import GraphConstructor


class ModuleMapGraphConstructor(GraphConstructor):
    def __init__(self, module_map_path: str, gpu_blocks: int, gpu_device: int) -> None:
        self._implementation = ModuleMapGraphCuda(
            module_map_path,
            gpu_blocks=gpu_blocks,
            gpu_device=gpu_device,
        )

    def build(self, features: torch.Tensor) -> TrackingGraph:
        node_features, edge_index, edge_features, hit_ids = self._implementation.build(
            features.contiguous()
        )
        return TrackingGraph(
            node_features=node_features,
            edge_index=edge_index,
            edge_features=edge_features,
            hit_ids=hit_ids,
        )
