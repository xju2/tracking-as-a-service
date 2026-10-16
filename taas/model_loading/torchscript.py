"""TorchScript loading helpers."""

import torch


def load_torchscript(path, device):
    return torch.jit.load(str(path), map_location=device)
