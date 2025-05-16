from __future__ import annotations

import operator
from functools import partial

import networkx as nx


def remove_cycles(graph):
    """
    Remove cycles from the graph, simply by pointing all edges outwards
    """

    R = graph.hit_r**2 + graph.hit_z**2
    edge_flip_mask = (R[graph.edge_index[0]] > R[graph.edge_index[1]]) | (
        (R[graph.edge_index[0]] == R[graph.edge_index[1]])
        & (graph.edge_index[0] > graph.edge_index[1])
    )
    graph.edge_index[:, edge_flip_mask] = graph.edge_index[:, edge_flip_mask].flip(0)

    return graph


def topological_sort_graph(G):
    """
    Sort Topologcially the graph such node u appears befroe v if the connection is u->v
    This ordering is valid only if the graph has no directed cycles
    """
    H = nx.DiGraph()
    # Add nodes w/o any features attached
    # maybe this is not needed given line 48?
    H.add_nodes_from(nx.topological_sort(G))

    # put it after the add nodes
    H.add_edges_from(G.edges(data=True))
    sorted_nodes = []

    # Add corresponding nodes features
    for i in list(nx.topological_sort(G)):
        sorted_nodes.append((i, G.nodes[i]))
    H.add_nodes_from(sorted_nodes)

    return H


def find_next_hits(
    G: nx.DiGraph,
    current_hit: int,
    used_hits: set,
    score_name: str,
    th_min: float,
    th_add: float,
):
    """Find what are the next hits we keep to build trakc candidates
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

    Args:
    ----
        G : nx.DiGraph, the input graph after GNN.
        starting_node : int, the starting node.
        next_hit_fn : callable, the function to find the next hit.
        used_hits : set, the set of used hits.

    Returns:
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
