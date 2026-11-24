"""Check the deterministic message sums against the atomic ones: python test_deterministic_sum.py"""

import copy

import torch
from interaction_gnn import SAGEConv, make_deterministic, scatter_sum, sort_by_index


def test_deterministic_sum(device="cuda" if torch.cuda.is_available() else "cpu"):
    torch.manual_seed(0)
    num_nodes, num_edges = 500, 20000
    x = torch.randn(num_nodes, 16, device=device)
    edge_index = torch.randint(0, num_nodes, (2, num_edges), device=device)
    edge_index[1, :10] = 0  # a node with many messages
    messages = torch.randn(num_edges, 32, device=device)

    # Interaction GNN sum: same values as the atomic scatter-add, bit-identical on repeat.
    order = sort_by_index(edge_index[1], num_nodes)
    det = scatter_sum(messages, edge_index[1], num_nodes, order)
    assert torch.allclose(det, scatter_sum(messages, edge_index[1], num_nodes), atol=1e-4)
    assert torch.equal(det, scatter_sum(messages, edge_index[1], num_nodes, order))

    # Filter SAGEConv (mean aggregation): same output with the deterministic aggregation.
    conv = SAGEConv(16, 8).to(device)
    det_conv = make_deterministic(copy.deepcopy(conv))
    assert torch.allclose(det_conv(x, edge_index), conv(x, edge_index), atol=1e-5)
    assert torch.equal(det_conv(x, edge_index), det_conv(x, edge_index))
    # Edges already sorted by target, as run_gnn_filter passes them: the sort is reused.
    sorted_edges = edge_index[:, torch.argsort(edge_index[1], stable=True)]
    assert torch.allclose(det_conv(x, sorted_edges), conv(x, sorted_edges), atol=1e-5)
    assert torch.equal(det_conv(x, sorted_edges), det_conv(x, edge_index))
    print(f"deterministic sums OK on {device}")


if __name__ == "__main__":
    test_deterministic_sum()
