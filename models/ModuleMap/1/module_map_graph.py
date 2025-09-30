# Copyright (C) 2023 CERN for the benefit of the ATLAS collaboration

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# 3rd party imports
import os
import logging
import torch
from tqdm import tqdm
import pymmg.mmg_wrapper as mmg
import torch.cuda.nvtx as nvtx
# Local imports
# from ..graph_construction_stage import GraphConstructionStage
# from . import utils
# from acorn.utils.loading_utils import remove_variable_name_prefix_in_pyg


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# class ModuleMapGraph(GraphConstructionStage):
#     def __init__(self, hparams):
#         super().__init__()
#         """
#         Initialise the PyModuleMap - a python implementation of the Triplet Module Map.
#         """
#         self.hparams = hparams
#         self.gpu_available = torch.cuda.is_available()
#         if self.gpu_available:
#             self.device = "cuda"
#         else:
#             self.device = "cpu"

#         # Logging config
#         self.log = logging.getLogger("ModuleMapGraph")
#         log_level = (
#             self.hparams["log_level"].upper()
#             if "log_level" in self.hparams
#             else "WARNING"
#         )

#         if log_level == "WARNING":
#             self.log.setLevel(logging.WARNING)
#         elif log_level == "INFO":
#             self.log.setLevel(logging.INFO)
#         elif log_level == "DEBUG":
#             self.log.setLevel(logging.DEBUG)
#         else:
#             raise ValueError(f"Unknown logging level {log_level}")

#         mmg.init_graph_builder(self.hparams["module_map_pattern_path"])

#     def to(self, device):
#         return self

#     def build_graphs(self, dataset, data_name):
#         """
#         Build the graphs for the data.
#         """

#         output_dir = os.path.join(self.hparams["stage_dir"], data_name)
#         os.makedirs(output_dir, exist_ok=True)
#         self.log.info(f"Building graphs for {data_name}")
#         for graph in tqdm(dataset):
#             event_id = graph.event_id
#             if graph is None:
#                 continue
#             if os.path.exists(os.path.join(output_dir, f"event{event_id}.pyg")):
#                 print(f"Graph {event_id} already exists, skipping...")
#                 continue

#             graph = self.build_graph(graph)
#             if not self.hparams.get("variable_with_prefix"):
#                 graph = remove_variable_name_prefix_in_pyg(graph)
#             torch.save(graph, os.path.join(output_dir, f"event{event_id}.pyg"))

#     def build_graph(self, graph):
#         nvtx.range_push("build_edge_index")
#         graph.edge_index = mmg.build_edge_index(
#             graph.event_id,
#             graph.hit_id,
#             graph.hit_module_id,
#             graph.hit_x,
#             graph.hit_y,
#             graph.hit_z,
#             graph.hit_id.shape[0],
#         )
#         nvtx.range_pop()
#         nvtx.range_push("graph_intersection")
#         y, truth_map = utils.graph_intersection(
#             graph.edge_index.to(device),
#             graph.track_edges.to(device),
#             return_y_pred=True,
#             return_truth_to_pred=True,
#         )
#         nvtx.range_pop()
#         nvtx.range_push("Move to cpu")
#         graph.edge_y = y.cpu()
#         graph.track_to_edge_map = truth_map.cpu()
#         nvtx.range_pop()
#         return graph

def build_graph(self, graph):
    nvtx.range_push("build_edge_index")
    graph.edge_index = mmg.build_edge_index(
        graph.event_id,
        graph.hit_id,
        graph.hit_module_id,
        graph.hit_x,
        graph.hit_y,
        graph.hit_z,
        graph.hit_id.shape[0],
    )
    nvtx.range_pop()
    # nvtx.range_push("graph_intersection")
    # y, truth_map = utils.graph_intersection(
    #     graph.edge_index.to(device),
    #     graph.track_edges.to(device),
    #     return_y_pred=True,
    #     return_truth_to_pred=True,
    # )
    # nvtx.range_pop()
    # nvtx.range_push("Move to cpu")
    # graph.edge_y = y.cpu()
    # graph.track_to_edge_map = truth_map.cpu()
    # nvtx.range_pop()
    return graph.edge_index
