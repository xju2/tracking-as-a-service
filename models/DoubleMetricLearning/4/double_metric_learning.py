import torch.nn as nn

# 3rd party imports
import torch.nn.functional as F

# Local imports
from utils import make_mlp


class DoubleMetricLearning(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        """
        Initialise the Lightning Module that can scan over different embedding training regimes
        """

        # Construct the MLP architecture
        in_channels = len(hparams["node_features"])

        self.network = make_mlp(
            in_channels,
            [hparams["emb_hidden"]] * hparams["nb_layer"],
            hidden_activation=hparams["activation"],
            output_activation=hparams["activation"],
            layer_norm=True,
        )

        self.src_decoder = make_mlp(
            hparams["emb_hidden"],
            [hparams["emb_dim"]],
            hidden_activation=hparams["activation"],
            output_activation=None,
            layer_norm=True,
        )

        self.tgt_decoder = make_mlp(
            hparams["emb_hidden"],
            [hparams["emb_dim"]],
            hidden_activation=hparams["activation"],
            output_activation=None,
            layer_norm=True,
        )

        self.hparams = hparams

    def forward(self, x):
        x = self.network(x)
        x_src = self.src_decoder(x)
        x_tgt = self.tgt_decoder(x)
        return F.normalize(x_src), F.normalize(x_tgt)
