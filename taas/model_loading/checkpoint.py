"""Torch checkpoint loading helpers."""

import torch


def load_checkpoint(path, map_location=None):
    return torch.load(path, map_location=map_location)
