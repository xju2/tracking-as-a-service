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
        device_type = inputs[0].device.type
        if auto_cast and check_autocast_support(device_type):
            with torch.autocast(device_type, dtype=dtype):
                output = model(*inputs).clone() # .to(torch.float32)
        else:
            output = model(*inputs).clone()
    return output


def run_gnn_filter(
    model: torch.nn.Module,
    auto_cast: bool,
    batches: int,
    x: torch.Tensor,
    edge_index: torch.Tensor,
):
    with torch.inference_mode():
        sorted_edge_index = sort_edge_index(edge_index, sort_by_row=False)
        device_type = x.device.type
        if auto_cast and check_autocast_support(device_type):
            with torch.autocast(device_type, dtype=dtype):
                gnn_embedding = model.gnn(x, sorted_edge_index).clone()
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
    return filter_scores, sorted_edge_index, gnn_embedding


def run_gnn_filter_optimized(
    model: torch.nn.Module,
    auto_cast: bool,
    batches: int,
    x: torch.Tensor,
    edge_index: torch.Tensor,
):
    """GNN filter scores, running the edge MLP on fixed-size chunks of edges.

    Restored from MetricLearning version 4, which inference.py came from; it was lost when
    the versions were merged. As there, `batches` is ignored: chunks are 2**16 edges.
    """
    chunk_size = 2**16
    with torch.inference_mode():
        sorted_edge_index = sort_edge_index(edge_index, sort_by_row=False)
        device_type = x.device.type
        filter_scores = []
        with torch.autocast(
            device_type, dtype=dtype, enabled=auto_cast and check_autocast_support(device_type)
        ):
            gnn_embedding = model.gnn(x, sorted_edge_index)
            for i in range(0, sorted_edge_index.shape[1], chunk_size):
                edge_batch = sorted_edge_index[:, i : i + chunk_size]
                edge_features = torch.cat(
                    [gnn_embedding[edge_batch[0]], gnn_embedding[edge_batch[1]]], dim=-1
                )
                filter_scores.append(model.net(edge_features).squeeze(-1))
        filter_scores = torch.cat(filter_scores)
    return filter_scores.sigmoid(), sorted_edge_index, gnn_embedding.clone()
