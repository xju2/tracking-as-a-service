import pymmg.mmg_wrapper as mmg
import torch.cuda.nvtx as nvtx


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
    return graph.edge_index
