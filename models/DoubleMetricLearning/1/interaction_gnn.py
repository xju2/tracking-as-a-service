from typing import Dict, Optional, Tuple
import importlib

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torch_scatter import scatter_add, scatter_mean, segment_csr
from torch_geometric.nn import SAGEConv
from torch_geometric.nn.aggr import Aggregation


# ---------------------------------------------------------------------------------------
# Deterministic message sums. CUDA scatter-adds use atomics, so the order of the additions,
# and hence the float rounding, changes from run to run, and the same event can give
# different tracks. make_deterministic() switches the GNNs to sorting the messages by
# target and summing each target's messages in that fixed order, at some latency cost.
# ---------------------------------------------------------------------------------------
def sort_by_index(index: torch.Tensor, dim_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Permutation that sorts index (stable), and CSR pointers of the sorted index."""
    perm = torch.argsort(index, stable=True)
    ptr = torch.zeros(dim_size + 1, dtype=torch.long, device=index.device)
    ptr[1:] = torch.cumsum(torch.bincount(index, minlength=dim_size), 0)
    return perm, ptr


def scatter_sum(
    src: torch.Tensor,
    index: torch.Tensor,
    dim_size: int,
    order: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> torch.Tensor:
    """Sum the rows of src [E, F] into [dim_size, F] by index.

    With order = sort_by_index(index, dim_size), the sum is deterministic; without it,
    it uses a CUDA atomic scatter-add.
    """
    if order is not None:
        perm, ptr = order
        return segment_csr(src[perm], ptr, reduce="sum")
    return torch.scatter_add(
        torch.zeros(src.shape[1], dim_size, device=src.device),
        1,
        torch.unsqueeze(index, 0).tile(src.shape[1], 1),
        src.T,
    ).T


class DeterministicAggregation(Aggregation):
    """The reduction of a torch_geometric aggregation ("sum", "mean", ...) in a fixed order."""

    def __init__(self, reduce: str):
        super().__init__()
        self.reduce_op = reduce

    def forward(self, x, index=None, ptr=None, dim_size=None, dim=-2):
        if ptr is None:
            dim_size = int(index.max()) + 1 if dim_size is None else dim_size
            if bool((index[1:] >= index[:-1]).all()):
                # Already sorted by target (run_gnn_filter sorts the edges): reuse that order.
                ptr = torch.zeros(dim_size + 1, dtype=torch.long, device=index.device)
                ptr[1:] = torch.cumsum(torch.bincount(index, minlength=dim_size), 0)
            else:
                perm, ptr = sort_by_index(index, dim_size)
                x = x.index_select(dim, perm)
        # With ptr given, Aggregation.reduce does a segment reduce instead of a scatter.
        return self.reduce(x, None, ptr, dim_size, dim, reduce=self.reduce_op)


def make_deterministic(model: nn.Module) -> nn.Module:
    """Make the message sums of every GNN in model deterministic (see above)."""
    for module in model.modules():
        if isinstance(module, InteractionGNNParams):
            module.deterministic = True
        elif isinstance(module, SAGEConv):
            module.aggr_module = DeterministicAggregation(module.aggr)
    return model


class InteractionGNNParams:
    """handel GNN paramerters"""

    params: Dict[str, bool]

    def __init__(self, hparams):
        super().__init__()
        hparams["batchnorm"] = False if "batchnorm" not in hparams else hparams["batchnorm"]
        hparams["output_batch_norm"] = hparams.get("output_batch_norm", False)
        hparams["edge_output_transform_final_batch_norm"] = hparams.get(
            "edge_output_transform_final_batch_norm", False
        )
        hparams["edge_output_transform_final_batch_norm"] = hparams.get(
            "edge_output_transform_final_batch_norm", False
        )
        hparams["track_running_stats"] = hparams.get("track_running_stats", False)

        # define boolean variables used in the forward method.
        # For non-booleans, treat them as attributes of the class
        # because TorchScript requires the dictionary has consistent types.
        self.params = {}
        self.params["checkpointing"] = hparams.get("checkpointing", False)
        self.params["concat"] = hparams.get("concat", True)
        self.params["use_scatter_add"] = True if hparams.get("aggr", "sum") == "sum" else False
        self.deterministic: bool = False  # set by make_deterministic()
        self.n_graph_iters: int = hparams.get("n_graph_iters", 8)
        self.in_out_diff_agg: bool = hparams.get("in_out_diff_agg", True)
        self.ckpting_reentrant: bool = hparams.get("ckpting_reentrant", False)

        if hparams["concat"]:
            if hparams["in_out_diff_agg"]:
                self.in_node_net = hparams["hidden"] * 4
            else:
                self.in_node_net = hparams["hidden"] * 3
            self.in_edge_net = hparams["hidden"] * 6
        else:
            if hparams["in_out_diff_agg"]:
                self.in_node_net = hparams["hidden"] * 3
            else:
                self.in_node_net = hparams["hidden"] * 2
            self.in_edge_net = hparams["hidden"] * 3
        # node encoder
        self.node_encoder = make_mlp(
            input_size=len(hparams["node_features"]),
            sizes=[hparams["hidden"]] * hparams["n_node_net_layers"],
            output_activation=hparams["output_activation"],
            hidden_activation=hparams["hidden_activation"],
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],
            output_batch_norm=hparams["output_batch_norm"],
            track_running_stats=hparams["track_running_stats"],
        )
        # edge encoder
        if "edge_features" in hparams and len(hparams["edge_features"]) != 0:
            self.edge_encoder = make_mlp(
                input_size=len(hparams["edge_features"]),
                sizes=[hparams["hidden"]] * hparams["n_edge_net_layers"],
                output_activation=hparams["output_activation"],
                hidden_activation=hparams["hidden_activation"],
                layer_norm=hparams["layernorm"],
                batch_norm=hparams["batchnorm"],
                output_batch_norm=hparams["output_batch_norm"],
                track_running_stats=hparams["track_running_stats"],
            )
        else:
            self.edge_encoder = make_mlp(
                input_size=2 * hparams["hidden"],
                sizes=[hparams["hidden"]] * hparams["n_edge_net_layers"],
                output_activation=hparams["output_activation"],
                hidden_activation=hparams["hidden_activation"],
                layer_norm=hparams["layernorm"],
                batch_norm=hparams["batchnorm"],
                output_batch_norm=hparams["output_batch_norm"],
                track_running_stats=hparams["track_running_stats"],
            )

        # edge decoder
        self.edge_decoder = make_mlp(
            input_size=hparams["hidden"],
            sizes=[hparams["hidden"]] * hparams["n_edge_decoder_layers"],
            output_activation=hparams["output_activation"],
            hidden_activation=hparams["hidden_activation"],
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],
            output_batch_norm=hparams["output_batch_norm"],
            track_running_stats=hparams["track_running_stats"],
        )
        # edge output transform layer
        self.edge_output_transform = make_mlp(
            input_size=hparams["hidden"],
            sizes=[hparams["hidden"], 1],
            output_activation=hparams["edge_output_transform_final_activation"],
            hidden_activation=hparams["hidden_activation"],
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],
            output_batch_norm=hparams["edge_output_transform_final_batch_norm"],
            track_running_stats=hparams["track_running_stats"],
        )
        self.dropout = nn.Dropout(p=0.1)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ):
        return self.forward_without_checkpoint(node_features, edge_index, edge_attr)


def make_mlp(
    input_size,
    sizes,
    hidden_activation="ReLU",
    output_activation=None,
    layer_norm=False,  # TODO : change name to hidden_layer_norm while ensuring backward compatibility
    output_layer_norm=False,
    batch_norm=False,  # TODO : change name to hidden_batch_norm while ensuring backward compatibility
    output_batch_norm=False,
    input_dropout=0,
    hidden_dropout=0,
    track_running_stats=False,
):
    """Construct an MLP with specified fully-connected layers."""
    hidden_activation = getattr(nn, hidden_activation)
    if output_activation is not None:
        output_activation = getattr(nn, output_activation)
    layers = []
    n_layers = len(sizes)
    sizes = [input_size] + sizes
    # Hidden layers
    for i in range(n_layers - 1):
        if i == 0 and input_dropout > 0:
            layers.append(nn.Dropout(input_dropout))
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if layer_norm:  # hidden_layer_norm
            layers.append(nn.LayerNorm(sizes[i + 1], elementwise_affine=False))
        if batch_norm:  # hidden_batch_norm
            layers.append(
                nn.BatchNorm1d(
                    sizes[i + 1],
                    eps=6e-05,
                    track_running_stats=track_running_stats,
                    affine=True,
                )  # TODO : Set BatchNorm and LayerNorm parameters in config file ?
            )
        layers.append(hidden_activation())
        if hidden_dropout > 0:
            layers.append(nn.Dropout(hidden_dropout))
    # Final layer
    layers.append(nn.Linear(sizes[-2], sizes[-1]))
    if output_activation is not None:
        if output_layer_norm:
            layers.append(nn.LayerNorm(sizes[-1], elementwise_affine=False))
        if output_batch_norm:
            layers.append(
                nn.BatchNorm1d(
                    sizes[-1],
                    eps=6e-05,
                    track_running_stats=track_running_stats,
                    affine=True,
                )  # TODO : Set BatchNorm and LayerNorm parameters in config file ?
            )
        layers.append(output_activation())
    return nn.Sequential(*layers)


class RecurrentInteractionGNN2(InteractionGNNParams, nn.Module):
    """
    Interaction Network (L2IT version).
    Operates on directed graphs.
    Aggregate and reduce (sum) separately incomming and outcoming edges latents.
    """

    def __init__(self, hparams):
        super().__init__(hparams)

        # edge network
        self.edge_network = make_mlp(
            input_size=self.in_edge_net,
            sizes=[hparams["hidden"]] * hparams["n_edge_net_layers"],
            output_activation=hparams["output_activation"],
            hidden_activation=hparams["hidden_activation"],
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],
            output_batch_norm=hparams["output_batch_norm"],
            track_running_stats=hparams["track_running_stats"],
        )

        # node network
        self.node_network = make_mlp(
            input_size=self.in_node_net,
            sizes=[hparams["hidden"]] * hparams["n_node_net_layers"],
            output_activation=hparams["output_activation"],
            hidden_activation=hparams["hidden_activation"],
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],
            output_batch_norm=hparams["output_batch_norm"],
            track_running_stats=hparams["track_running_stats"],
        )

    @torch.jit.export
    def forward_without_checkpoint(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ):
        src, dst = edge_index[0], edge_index[1]
        x = self.node_encoder(x)
        if edge_attr is not None:
            e = self.edge_encoder(edge_attr)
        else:
            e = self.edge_encoder(torch.cat([x[src], x[dst]], dim=-1))

        input_x = x
        input_e = e
        outputs = []
        dst_order: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        src_order: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        if self.deterministic:
            dst_order = sort_by_index(dst, x.shape[0])
            src_order = sort_by_index(src, x.shape[0])
        for i in range(self.n_graph_iters):
            if self.params["concat"]:
                x = torch.cat([x, input_x], dim=-1)
                e = torch.cat([e, input_e], dim=-1)
            x, e, out = self.recurrent_message_step(x, e, src, dst, dst_order, src_order)
            outputs.append(out)
        return outputs[-1].squeeze(-1)

    @torch.jit.unused
    def forward_with_checkpoint(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ):
        src, dst = edge_index
        x.requires_grad = True

        if edge_attr is not None:
            edge_attr.requires_grad = True

        # Encode nodes and edges features into latent spaces
        x = checkpoint(self.node_encoder, x, use_reentrant=self.ckpting_reentrant)
        if edge_attr is not None:
            e = checkpoint(self.edge_encoder, edge_attr, use_reentrant=self.ckpting_reentrant)
        else:
            e = checkpoint(
                self.edge_encoder,
                torch.cat([x[src], x[dst]], dim=-1),
                use_reentrant=self.ckpting_reentrant,
            )

        # memorize initial encodings for concatenate in the gnn loop if request
        if self.hparams["concat"]:
            input_x = x
            input_e = e
        # Initialize outputs
        outputs = []
        # Loop over gnn layers
        for _ in range(self.n_graph_iters):
            if self.hparams["concat"]:
                x = checkpoint(self.concat, x, input_x, use_reentrant=self.ckpting_reentrant)
                e = checkpoint(self.concat, e, input_e, use_reentrant=self.ckpting_reentrant)

            x, e, out = checkpoint(
                self.recurrent_message_step,
                x,
                e,
                src,
                dst,
                use_reentrant=self.ckpting_reentrant,
            )
            outputs.append(out)

        return outputs[-1].squeeze(-1)

    def recurrent_message_step(
        self,
        x: torch.Tensor,
        e: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        dst_order: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        src_order: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        edge_inputs = torch.cat([e, x[src], x[dst]], dim=-1)
        e_updated = self.edge_network(edge_inputs)

        edge_messages_from_src = scatter_sum(e_updated, dst, x.shape[0], dst_order)
        if not self.params["use_scatter_add"]:
            # use scatter_mean. Counts are sums of 1.0, exact in any order.
            e_ones = torch.ones(dst.shape[0], device=dst.device)
            count = torch.scatter_add(
                torch.zeros(x.shape[0], device=dst.device),
                0,
                dst,
                e_ones,
            ).unsqueeze(-1)
            count[count < 1] = 1
            edge_messages_from_src.div_(count)

        edge_messages_from_dst = scatter_sum(e_updated, src, x.shape[0], src_order)
        if not self.params["use_scatter_add"]:
            # use scatter_mean
            e_ones = torch.ones(dst.shape[0], device=dst.device)
            count = torch.scatter_add(
                torch.zeros(x.shape[0], device=src.device),
                0,
                src,
                e_ones,
            ).unsqueeze(-1)
            count[count < 1] = 1
            edge_messages_from_dst.div_(count)

        if self.in_out_diff_agg:
            node_inputs = torch.cat(
                [edge_messages_from_src, edge_messages_from_dst, x], dim=-1
            )  # to check : the order dst src  x ?
        else:
            # add message from src and dst ?? # edge_messages = edge_messages_from_src + edge_messages_from_dst
            edge_messages = edge_messages_from_src + edge_messages_from_dst
            node_inputs = torch.cat([edge_messages, x], dim=-1)

        x_updated = self.node_network(node_inputs)

        return (
            x_updated,
            e_updated,
            self.edge_output_transform(self.edge_decoder(e_updated)),
        )

    def concat(self, x, y):
        return torch.cat([x, y], dim=-1)


class ChainedInteractionGNN2(InteractionGNNParams, nn.Module):
    """
    Interaction Network (L2IT version).
    Operates on directed graphs.
    Aggregate and reduce (sum) separately incomming and outcoming edges latents.
    """

    def __init__(self, hparams):
        super().__init__(hparams)

        # edge network
        self.edge_network = nn.ModuleList([
            make_mlp(
                input_size=self.in_edge_net,
                sizes=[hparams["hidden"]] * hparams["n_edge_net_layers"],
                output_activation=hparams["output_activation"],
                hidden_activation=hparams["hidden_activation"],
                layer_norm=hparams["layernorm"],
                batch_norm=hparams["batchnorm"],
                output_batch_norm=hparams["output_batch_norm"],
                track_running_stats=hparams["track_running_stats"],
            )
            for _ in range(hparams["n_graph_iters"])
        ])

        # node network
        self.node_network = nn.ModuleList([
            make_mlp(
                input_size=self.in_node_net,
                sizes=[hparams["hidden"]] * hparams["n_node_net_layers"],
                output_activation=hparams["output_activation"],
                hidden_activation=hparams["hidden_activation"],
                layer_norm=hparams["layernorm"],
                batch_norm=hparams["batchnorm"],
                output_batch_norm=hparams["output_batch_norm"],
                track_running_stats=hparams["track_running_stats"],
            )
            for _ in range(hparams["n_graph_iters"])
        ])

    def forward_without_checkpoint(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ):
        src, dst = edge_index[0], edge_index[1]
        x = self.node_encoder(x)
        if edge_attr is not None:
            e = self.edge_encoder(edge_attr)
        else:
            e = self.edge_encoder(torch.cat([x[src], x[dst]], dim=-1))

        input_x = x
        input_e = e
        outputs = []
        dst_order: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        src_order: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        if self.deterministic:
            dst_order = sort_by_index(dst, x.shape[0])
            src_order = sort_by_index(src, x.shape[0])

        for node_net, edge_net in zip(self.node_network, self.edge_network):
            if self.params["concat"]:
                x = torch.cat([x, input_x], dim=-1)
                e = torch.cat([e, input_e], dim=-1)
            edge_inputs = torch.cat([e, x[src], x[dst]], dim=-1)
            e_updated = edge_net(edge_inputs)

            edge_messages_from_src = scatter_sum(e_updated, dst, x.shape[0], dst_order)
            if not self.params["use_scatter_add"]:
                # use scatter_mean. Counts are sums of 1.0, exact in any order.
                e_ones = torch.ones(dst.shape[0], device=dst.device)
                count = torch.scatter_add(
                    torch.zeros(x.shape[0], device=dst.device),
                    0,
                    dst,
                    e_ones,
                ).unsqueeze(-1)
                count[count < 1] = 1
                edge_messages_from_src.div_(count)

            edge_messages_from_dst = scatter_sum(e_updated, src, x.shape[0], src_order)
            if not self.params["use_scatter_add"]:
                # use scatter_mean
                e_ones = torch.ones(dst.shape[0], device=dst.device)
                count = torch.scatter_add(
                    torch.zeros(x.shape[0], device=src.device),
                    0,
                    src,
                    e_ones,
                ).unsqueeze(-1)
                count[count < 1] = 1
                edge_messages_from_dst.div_(count)

            if self.in_out_diff_agg:
                node_inputs = torch.cat(
                    [edge_messages_from_src, edge_messages_from_dst, x], dim=-1
                )
            else:
                edge_messages = edge_messages_from_src + edge_messages_from_dst
                node_inputs = torch.cat([edge_messages, x], dim=-1)
            x_updated = node_net(node_inputs)
            outputs.append(self.edge_output_transform(self.edge_decoder(e_updated)))
            x = x_updated
            e = e_updated

        return outputs[-1].squeeze(-1)

    def message_step(
        self,
        x: torch.Tensor,
        e: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        i: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        edge_inputs = torch.cat([e, x[src], x[dst]], dim=-1)  # order dst src x ?
        # Torch.jit does not like the use of edge_network[i]. Error info from the torch.jit is:
        # Expected integer literal for index but got a variable or non-integer.
        # ModuleList/Sequential indexing is only supported with integer literals.
        # For example, 'i = 4; self.layers[i](x)' will fail because i is not a literal.
        # Enumeration is supported, e.g. 'for index, v in enumerate(self): out = v(inp)':
        # So this function is only used in training mode.
        e_updated = self.edge_network[i](edge_inputs)

        # Update nodes
        if self.params.get("aggr", "sum") != "sum":
            edge_messages_from_src = scatter_mean(e_updated, dst, dim=0, dim_size=x.shape[0])
            edge_messages_from_dst = scatter_mean(e_updated, src, dim=0, dim_size=x.shape[0])
        else:
            edge_messages_from_src = scatter_add(e_updated, dst, dim=0, dim_size=x.shape[0])
            edge_messages_from_dst = scatter_add(e_updated, src, dim=0, dim_size=x.shape[0])
        if self.in_out_diff_agg:
            node_inputs = torch.cat(
                [edge_messages_from_src, edge_messages_from_dst, x], dim=-1
            )  # to check : the order dst src  x ?
        else:
            # add message from src and dst ?? # edge_messages = edge_messages_from_src + edge_messages_from_dst
            edge_messages = edge_messages_from_src + edge_messages_from_dst
            node_inputs = torch.cat([edge_messages, x], dim=-1)

        x_updated = self.node_network[i](node_inputs)

        return (
            x_updated,
            e_updated,
            self.edge_output_transform(self.edge_decoder(e_updated)),
        )

    def concat(self, x, y):
        return torch.cat([x, y], dim=-1)


class GCNEncoderJitable(nn.Module):
    def __init__(self, gnn_config) -> None:
        super().__init__()
        self.gnn_config = gnn_config
        self.layers = nn.ModuleList()

        for conf in gnn_config:
            module = importlib.import_module(conf["module_name"])
            layer = getattr(module, conf["class_name"])(**conf["init_kwargs"])
            self.layers.append(layer)

    def forward(self, x: torch.Tensor, edge_list: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            if isinstance(layer, SAGEConv):
                x = layer(x, edge_list)
            else:
                x = layer(x)
        return x


class GNNFilterJitable(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        hparams["batchnorm"] = False if "batchnorm" not in hparams else hparams["batchnorm"]
        hparams["track_running_stats"] = hparams.get("track_running_stats", False)

        self.net = make_mlp(
            hparams["hidden"] * 2,
            [hparams["hidden"] // (2**i) for i in range(hparams["nb_layer"])] + [1],
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],
            output_activation=None,
            hidden_activation=hparams["hidden_activation"],
            track_running_stats=hparams["track_running_stats"],
        )
        self.gnn = GCNEncoderJitable(hparams["gnn_config"])

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        x = self.gnn(x, edge_index)
        output = self.net(torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=-1))
        return output.squeeze(-1)
