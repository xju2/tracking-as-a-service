import torch
from torch_geometric.utils import sort_edge_index

dtype = torch.float16


def check_autocast_support(device_type: str):
    if torch.__version__ < "2.4.0":
        return device_type == "cuda"
    return torch.amp.autocast_mode.is_autocast_available(device_type)


def run_torch_model(model: torch.nn.Module, auto_cast: bool, *inputs):
    assert len(inputs) > 0, "At least one input must be provided."
    with torch.no_grad():
        device_type = inputs[0].device.type
        if auto_cast and check_autocast_support(device_type):
            with torch.autocast(device_type, dtype=dtype):
                output = model(*inputs)
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
    with torch.no_grad():
        sorted_edge_index = sort_edge_index(edge_index, sort_by_row=False)
        device_type = x.device.type
        if auto_cast and check_autocast_support(device_type):
            with torch.autocast(device_type, dtype=dtype):
                gnn_embedding = model.gnn(x, sorted_edge_index)
                filter_scores = [
                    model.net(
                        torch.cat([gnn_embedding[subset[0]], gnn_embedding[subset[1]]], dim=-1)
                    ).squeeze(-1)
                    for subset in torch.tensor_split(sorted_edge_index, batches, dim=1)
                ]
        else:
            gnn_embedding = model.gnn(x, sorted_edge_index)
            filter_scores = [
                model.net(
                    torch.cat([gnn_embedding[subset[0]], gnn_embedding[subset[1]]], dim=-1)
                ).squeeze(-1)
                for subset in torch.tensor_split(sorted_edge_index, batches, dim=1)
            ]
    filter_scores = torch.cat(filter_scores).sigmoid()
    return filter_scores, sorted_edge_index, gnn_embedding
