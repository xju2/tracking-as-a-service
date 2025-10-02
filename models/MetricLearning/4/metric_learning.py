import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

# Local imports
from utils import build_edges, graph_intersection, make_mlp


class MetricLearning(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        """
        Initialise the Lightning Module that can scan over different embedding training regimes
        """

        # Construct the MLP architecture
        in_channels = len(hparams["node_features"])

        self.network = make_mlp(
            in_channels,
            [hparams["emb_hidden"]] * hparams["nb_layer"] + [hparams["emb_dim"]],
            hidden_activation=hparams["activation"],
            output_activation=None,
            layer_norm=True,
        )

        self.hparams = hparams

    def forward(self, x):
        x_out = self.network(x)
        return F.normalize(x_out)
