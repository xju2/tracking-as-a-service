import torch
from torch_geometric.utils import sort_edge_index

dtype = torch.float16


def _as_half(x):
    if isinstance(x, torch.Tensor):
        # Only convert floating-point tensors to half precision.
        # Leave integer/boolean/index tensors unchanged so they remain valid indices.
        try:
            return x.half() if x.is_floating_point() else x
        except Exception:
            return x
    if isinstance(x, (list, tuple)):
        return type(x)(_as_half(v) for v in x)
    return x


def run_torch_model(model: torch.nn.Module, auto_cast: bool, *inputs):
    assert len(inputs) > 0, "At least one input must be provided."
    with torch.inference_mode():
        if auto_cast:
            # convert inputs to half precision (in-place for floats), leave integer/index tensors unchanged
            half_inputs = tuple(_as_half(x) for x in inputs)
            output = model(*half_inputs)
        else:
            output = model(*inputs)
    return output



def run_gnn_filter(
    model: torch.nn.Module,
    auto_cast: bool,
    batches: int,
    x: torch.Tensor,
    edge_index: torch.Tensor,
):
    batches = 4
    with torch.inference_mode():
        sorted_edge_index = sort_edge_index(edge_index, sort_by_row=False)
        # Use explicit half-precision conversion when requested (no autocast)
        if auto_cast:
            x_half = _as_half(x)
            gnn_embedding = model.gnn(x_half, sorted_edge_index).clone()
            filter_scores = [
                model.net(
                    torch.cat([gnn_embedding[subset[0]], gnn_embedding[subset[1]]], dim=-1)
                ).squeeze(-1).clone()
                for subset in torch.tensor_split(sorted_edge_index, batches, dim=1)
            ]
        else:
            gnn_embedding = model.gnn(x, sorted_edge_index).clone()
            filter_scores = [
                model.net(
                    torch.cat([gnn_embedding[subset[0]], gnn_embedding[subset[1]]], dim=-1)
                ).squeeze(-1).clone()
                for subset in torch.tensor_split(sorted_edge_index, batches, dim=1)
            ]
    filter_scores = torch.cat(filter_scores).sigmoid()
    return filter_scores, sorted_edge_index.clone().long(), gnn_embedding.clone()

import torch

# Assume sort_edge_index, check_autocast_support, dtype are defined as before

def run_gnn_filter_optimized(
    model: torch.nn.Module,
    auto_cast: bool,
    batches: int,  # Tune this value based on your GPU memory
    x: torch.Tensor,
    edge_index: torch.Tensor,
    # Add a batch_size parameter to control memory usage

):
    batches = 2**16
    with torch.inference_mode():
        sorted_edge_index = sort_edge_index(edge_index, sort_by_row=False)
        device_type = x.device.type if hasattr(x, 'device') else str(x.device)
        num_edges = sorted_edge_index.shape[1]
       # If requested, convert model components and inputs to half precision
        if auto_cast:
            gnn_embedding = model.gnn(_as_half(x), sorted_edge_index).clone()
        else:
            gnn_embedding = model.gnn(x, sorted_edge_index).clone()

        # Process edges in batches to control memory
        filter_scores = []
        for i in range(0, num_edges, batches):
            # 1. Get the current batch of edges
            edge_batch = sorted_edge_index[:, i : i + batches]

            # 2. Gather embeddings for the current batch
            source_nodes = gnn_embedding[edge_batch[0]]
            target_nodes = gnn_embedding[edge_batch[1]]

            # 3. Concatenate features for the batch (this tensor is now much smaller)
            edge_features = torch.cat([source_nodes, target_nodes], dim=-1)

            # 4. Run the filter network on the batch
            scores_batch = model.net(edge_features).squeeze(-1).clone()
            filter_scores.append(scores_batch)

            # Explicitly free memory (optional, but can help in tight situations)
            del edge_batch, source_nodes, target_nodes, edge_features, scores_batch

        # 5. Concatenate the results from all batches
        filter_scores = torch.cat(filter_scores)

    # Return integer edge indices to avoid accidental dtype promotion
    return filter_scores.sigmoid(), sorted_edge_index.clone().long(), gnn_embedding.clone()
