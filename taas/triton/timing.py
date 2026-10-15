"""CUDA-aware timing helpers."""

import torch


def synchronize_if_cuda(device):
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
