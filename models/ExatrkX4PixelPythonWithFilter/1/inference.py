from __future__ import annotations

import operator
from functools import partial
from pathlib import Path

import frnn
import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx


def build_edges(
    query: torch.Tensor,
    database: torch.Tensor,
    r_max: float = 0.1,
    k_max: int = 1000,
) -> torch.Tensor:
    # Compute edges
    _, idxs, _, _ = frnn.frnn_grid_points(
        points1=query.unsqueeze(0),
        points2=database.unsqueeze(0),
        lengths1=None,
        lengths2=None,
        K=k_max,
        r=r_max,
        grid=None,
        return_nn=False,
        return_sorted=True,
    )

    idxs: torch.Tensor = idxs.squeeze().int()
    ind = torch.arange(idxs.shape[0], device=query.device).repeat(idxs.shape[1], 1).T.int()
    positive_idxs = idxs >= 0
    edge_list = torch.stack([ind[positive_idxs], idxs[positive_idxs]]).long()

    # Remove self-loops
    edge_list = edge_list[:, edge_list[0] != edge_list[1]]
    return edge_list


def find_next_hits(
    G: nx.DiGraph,
    current_hit: int,
    used_hits: set,
    score_name: str,
    th_min: float,
    th_add: float,
):
    """
    Find what are the next hits we keep to build trakc candidates
    G : the graph (usually pre-filtered)
    current_hit : index of the current_hit considered
    used_hits : a set of already used hits (to avoid re-using them)
    th_min : minimal threshold required to build at least one track candidate
             (we take the hit with the highest score)
    th_add : additional threshold above which we keep all hit neighbors, and not only the
             the one with the highest score. It results in several track candidates
             (th_add should be larger than th_min)
    path is previous hits.
    """
    # Sanity check
    if th_add < th_min:
        print(
            f"WARNING : the minimal threshold {th_min} is above the additional"
            f" threshold {th_add},               this is not how the walkthrough is"
            " supposed to be run."
        )

    # Check if current hit still have unused neighbor(s) hit(s)
    neighbors = [n for n in G.neighbors(current_hit) if n not in used_hits]
    if not neighbors:
        return None

    neighbors_scores = [(n, G.edges[(current_hit, n)][score_name]) for n in neighbors]

    # Stop here if none of the remaining neighbors are above the minimal threshold
    best_neighbor = max(neighbors_scores, key=operator.itemgetter(1))
    if best_neighbor[1] <= th_min:
        return None

    # Always add the neighoors with a score above th_add
    next_hits = [n for n, score in neighbors_scores if score > th_add]

    # Always add the highest scoring neighbor above th_min if it's not already in next_hits
    if not next_hits:
        best_hit = best_neighbor[0]
        if best_hit not in next_hits:
            next_hits.append(best_hit)

    return next_hits


def build_roads(G, starting_node, next_hit_fn, used_hits: set) -> list[tuple]:
    """Build roads starting from a given node.

    Args
    ----
        G : nx.DiGraph, the input graph after GNN.
        starting_node : int, the starting node.
        next_hit_fn : callable, the function to find the next hit.
        used_hits : set, the set of used hits.

    Returns
    -------
        path : list of list, the list of possible paths starting from the given node.
    """
    # Initialize the path with the starting node
    path = [(starting_node,)]

    while True:
        new_path = []
        is_all_none = True

        # Loop on all paths and see if we can find more hits to be added
        for pp in path:
            start = pp[-1]

            # Case where we are at the end of an interesting path
            if start is None:
                new_path.append(pp)
                continue

            # Call next hits function
            current_used_hits = used_hits | set(pp)
            next_hits = next_hit_fn(G, start, current_used_hits)
            if next_hits is None:
                new_path.append((*pp, None))
            else:
                is_all_none = False
                # branching the paths for the next hits.
                new_path.append((*pp, *next_hits))

        path = new_path
        # If all paths have reached the end, break
        if is_all_none:
            break

    return path


def get_tracks(G, th_min, th_add, score_name):
    """Run walkthrough and return subgraphs."""
    used_nodes = set()
    sub_graphs = []
    next_hit_fn = partial(find_next_hits, th_min=th_min, th_add=th_add, score_name=score_name)

    # Rely on the fact the graph was already topologically sorted
    # to start looking first on nodes without incoming edges
    for node in nx.topological_sort(G):
        # Ignore already used nodes
        if node in used_nodes:
            continue

        road = build_roads(G, node, next_hit_fn, used_nodes)
        a_road = max(road, key=len)[:-1]  # [:-1] is to remove the last None

        # Case where there is only one hit: a_road = (<node>,None)
        if len(a_road) < 3:
            used_nodes.add(node)
            continue

        # Need to drop the last item of the a_road tuple, since it is None
        sub_graphs.append([G.nodes[n]["hit_id"] for n in a_road])
        used_nodes.update(a_road)

    return sub_graphs


def get_simple_path(G, do_copy: bool = False):
    in_graph = G.copy() if not do_copy else G
    final_tracks = []
    connected_components = sorted(nx.weakly_connected_components(in_graph))
    for i in range(len(connected_components)):
        is_signal_path = True
        sub_graph = connected_components[i]
        for node in sub_graph:
            if not (in_graph.out_degree(node) <= 1 and in_graph.in_degree(node) <= 1):
                is_signal_path = False
        if is_signal_path:
            track = [in_graph.nodes[node]["hit_id"] for node in sub_graph]
            if len(track) > 2:
                final_tracks.append(track)
            in_graph.remove_nodes_from(sub_graph)
    return final_tracks, in_graph


class MetricLearningInference:
    def __init__(
        self,
        model_path: str,
        r_max: float = 0.1,
        k_max: int = 300,
        filter_cut: float = 0.2,
        filter_batches: int = 10,
        cc_cut: float = 0.01,
        walk_min: float = 0.1,
        walk_max: float = 0.6,
        device: str = "cuda",
        r_index: int = 0,
        z_index: int = 2,
        debug: bool = False,
    ):
        model_path = Path(model_path) if isinstance(model_path, str) else model_path
        self.r_max = r_max
        self.k_max = k_max
        self.filter_cut = filter_cut
        self.filter_batches = filter_batches
        self.cc_cut = cc_cut
        self.walk_min = walk_min
        self.walk_max = walk_max
        self.device = device
        self.r_index = r_index
        self.z_index = z_index
        self.debug = debug

        embedding_path = model_path / "embedding.pt"
        filtering_path = model_path / "filter.pt"
        gnn_path = model_path / "gnn.pt"

        self.embedding_model = torch.jit.load(embedding_path).to(device)
        self.filter_model = torch.jit.load(filtering_path).to(device)
        self.gnn_model = torch.jit.load(gnn_path).to(device)

    def forward(self, node_features: torch.Tensor, hit_id: torch.Tensor | None = None):
        node_features = node_features.to(self.device)

        if hit_id is None:
            hit_id = torch.arange(node_features.shape[0], device=self.device)

        # Metric Learning
        with torch.no_grad():
            embedding = self.embedding_model(node_features)

        if self.debug:
            print(f"after embedding, shape = {embedding.shape}")

        # Build edges
        edge_index = build_edges(embedding, embedding, r_max=self.r_max, k_max=self.k_max)
        del embedding

        if self.debug:
            print(f"Number of edges after embedding: {edge_index.shape[1]:,}")

        # Filtering
        with torch.no_grad():
            edge_score = [
                self.filter_model(node_features, subset).squeeze(-1)
                for subset in torch.tensor_split(edge_index, self.filter_batches, dim=1)
            ]

        edge_score = torch.cat(edge_score).sigmoid()
        edge_index = edge_index[:, edge_score > self.filter_cut]
        if self.debug:
            print(f"Number of edges after filtering: {edge_index.shape[1]:,}")

        # GNN
        # rearrange the edges by their distances from collision point.
        R = node_features[:, self.r_index] ** 2 + node_features[:, self.z_index] ** 2
        edge_flip_mask = R[edge_index[0]] > R[edge_index[1]]
        edge_index[:, edge_flip_mask] = edge_index[:, edge_flip_mask].flip(0)

        # apply GNN
        with torch.no_grad():
            edge_score = self.gnn_model(node_features, edge_index)
        edge_score = edge_score.sigmoid()

        # CC and Walkthrough
        if self.debug:
            print("After GNN...")

        score_name = "edge_score"
        graph = Data(
            x=node_features,
            edge_index=edge_index,
            hit_id=hit_id,
            edge_score=edge_score,
        )
        G = to_networkx(graph, ["hit_id"], [score_name], to_undirected=False)

        # Remove edges below threshold
        list_fake_edges = [(u, v) for u, v, e in G.edges(data=True) if e[score_name] <= self.cc_cut]
        G.remove_edges_from(list_fake_edges)

        G.remove_nodes_from(list(nx.isolates(G)))
        # print(f"After removing fake edges; Number of nodes
        # = {G.number_of_nodes():,}, number of edges = {G.number_of_edges():,}")

        all_trkx = {}
        all_trkx["cc"], G = get_simple_path(G)
        all_trkx["walk"] = get_tracks(G, self.walk_min, self.walk_max, score_name)
        if self.debug:
            print(f"Number of tracks found by CC: {len(all_trkx['cc'])}")
            print(f"Number of tracks found by Walkthrough: {len(all_trkx['walk'])}")

        # label each hits.
        track_candidates = np.array([
            item for track in all_trkx["cc"] + all_trkx["walk"] for item in [*track, -1]
        ])
        return track_candidates

    def __call__(self, node_features: torch.Tensor, hit_id: torch.Tensor | None = None):
        return self.forward(node_features, hit_id)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inference for Metric Learning")
    add_arg = parser.add_argument("-i", "--input", type=str, default="node_features.pt")

    args = parser.parse_args()
    inference = MetricLearningInference(model_path="./", r_max=0.11, debug=True)
    node_features = torch.load(args.input)
    track_ids = inference(node_features)
    print(track_ids)
