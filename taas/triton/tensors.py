"""Zero-copy Triton/Torch tensor conversion helpers."""

from torch.utils.dlpack import from_dlpack


def from_triton_tensor(tensor):
    return from_dlpack(tensor.to_dlpack())
