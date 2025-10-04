import torch
from torch_geometric.utils import sort_edge_index

dtype = torch.float16


def check_autocast_support(device_type: str):
    if torch.__version__ < "2.4.0":
        return device_type == "cuda"
    return torch.amp.autocast_mode.is_autocast_available(device_type)


def run_torch_model(model: torch.nn.Module, auto_cast: bool, *inputs):
    assert len(inputs) > 0, "At least one input must be provided."
    with torch.inference_mode():
        # device = next(model.parameters()).device
        device_type = "cuda:0"  # device.type if hasattr(device, 'type') else str(device)
        if auto_cast and check_autocast_support(device_type):
            with torch.autocast(device_type, dtype=dtype):
                output = model(*inputs)
                if isinstance(output, tuple):
                    return tuple(o.clone() for o in output)
                else:
                    return output.clone()  # .to(torch.float32)
                # print("compiled amp:", timed(lambda: model(*inputs))[1])
        else:
            output = model(*inputs)
            if isinstance(output, tuple):
                return tuple(o.clone() for o in output)
            else:
                return output.clone()  # .to(torch.float32)
            # print("compiled:", timed(lambda: model(*inputs))[1])
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
        device_type = x.device.type
        if auto_cast and check_autocast_support(device_type):
            with torch.autocast(device_type, dtype=dtype):
                gnn_embedding = model.gnn(x, sorted_edge_index).clone()
                filter_scores = [
                    model.net(
                        torch.cat([gnn_embedding[subset[0]], gnn_embedding[subset[1]]], dim=-1)
                    )
                    .squeeze(-1)
                    .clone()
                    for subset in torch.tensor_split(sorted_edge_index, batches, dim=1)
                ]
        else:
            gnn_embedding = model.gnn(x, sorted_edge_index).clone()
            filter_scores = [
                model.net(torch.cat([gnn_embedding[subset[0]], gnn_embedding[subset[1]]], dim=-1))
                .squeeze(-1)
                .clone()
                for subset in torch.tensor_split(sorted_edge_index, batches, dim=1)
            ]
    filter_scores = torch.cat(filter_scores).sigmoid()
    return filter_scores, sorted_edge_index, gnn_embedding


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
        device_type = x.device.type if hasattr(x, "device") else str(x.device)
        num_edges = sorted_edge_index.shape[1]

        filter_scores = []

        with torch.autocast(
            device_type, dtype=dtype, enabled=auto_cast and check_autocast_support(device_type)
        ):
            # GNN embeddings are calculated only once
            gnn_embedding = model.gnn(x, sorted_edge_index)

            # Process edges in batches to control memory
            for i in range(0, num_edges, batches):
                # 1. Get the current batch of edges
                edge_batch = sorted_edge_index[:, i : i + batches]

                # 2. Gather embeddings for the current batch
                source_nodes = gnn_embedding[edge_batch[0]]
                target_nodes = gnn_embedding[edge_batch[1]]

                # 3. Concatenate features for the batch (this tensor is now much smaller)
                edge_features = torch.cat([source_nodes, target_nodes], dim=-1)

                # 4. Run the filter network on the batch
                scores_batch = model.net(edge_features).squeeze(-1)
                filter_scores.append(scores_batch)

                # Explicitly free memory (optional, but can help in tight situations)
                del edge_batch, source_nodes, target_nodes, edge_features, scores_batch

        # 5. Concatenate the results from all batches
        filter_scores = torch.cat(filter_scores)

    return filter_scores.sigmoid(), sorted_edge_index, gnn_embedding.clone()
