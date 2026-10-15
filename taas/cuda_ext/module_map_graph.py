from pathlib import Path
from typing import Sequence, Tuple

import torch


_IMPORT_ERROR = None
_NATIVE_MODULE = None


def _load_native_module():
    global _IMPORT_ERROR, _NATIVE_MODULE

    if _NATIVE_MODULE is not None:
        return _NATIVE_MODULE

    if _IMPORT_ERROR is not None:
        return None

    try:
        from taas.cuda_ext import _module_map_graph_cuda
    except ImportError as exc:
        _IMPORT_ERROR = exc
        return None

    _NATIVE_MODULE = _module_map_graph_cuda
    return _NATIVE_MODULE


def module_map_graph_available() -> bool:
    return _load_native_module() is not None


def load_exact_module_ids(module_map_path) -> torch.Tensor:
    """Load exact ModuleMap module IDs from the ROOT map files.

    Returns a CPU int64 tensor. The underlying ROOT branches are unsigned long,
    but the ITk compact IDs used here fit in int64.
    """

    module = _load_native_module()
    if module is None:
        raise RuntimeError(
            "TAAS ModuleMapGraph CUDA extension is not available. Build it with "
            "`TAAS_BUILD_CUDA_EXT=1 TAAS_BUILD_MODULEMAP_GRAPH=1 "
            "python setup.py build_ext --inplace` in the modulemap image."
        ) from _IMPORT_ERROR

    return module.load_exact_module_ids(str(module_map_path))


def build_module_map_graph_cuda(
    node_features,
    module_ids,
    hit_ids,
    module_map_path,
    *,
    feature_indices: Sequence[int] = (0, 1, 2),
    gpu_blocks: int = 512,
    gpu_device: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build ModuleMapGraph CUDA edges.

    ``node_features`` is currently expected on CPU with columns ordered as
    ``(r, phi, z, ...)`` by default. ``module_ids`` and ``hit_ids`` are CPU
    int64 tensors; their values are cast to uint64 in the native wrapper.
    """

    module = _load_native_module()
    if module is None:
        raise RuntimeError(
            "TAAS ModuleMapGraph CUDA extension is not available. Build it with "
            "`TAAS_BUILD_CUDA_EXT=1 TAAS_BUILD_MODULEMAP_GRAPH=1 "
            "python setup.py build_ext --inplace` in the modulemap image."
        ) from _IMPORT_ERROR

    if len(feature_indices) != 3:
        raise ValueError("feature_indices must contain (r_index, phi_index, z_index)")

    node_features = node_features.detach().to(
        device="cpu", dtype=torch.float32
    ).contiguous()
    module_ids = module_ids.detach().to(device="cpu", dtype=torch.long).contiguous()
    if hit_ids is None:
        hit_ids = torch.arange(node_features.shape[0], dtype=torch.long)
    else:
        hit_ids = hit_ids.detach().to(device="cpu", dtype=torch.long).contiguous()

    r_index, phi_index, z_index = (int(i) for i in feature_indices)
    return tuple(
        module.build_module_map_graph_cuda(
            node_features,
            module_ids,
            hit_ids,
            str(module_map_path),
            r_index,
            phi_index,
            z_index,
            int(gpu_blocks),
            int(gpu_device),
        )
    )


class ModuleMapGraphCuda:
    """Persistent ACTS-like CUDA ModuleMap graph builder.

    The native object reads the ROOT ModuleMap once, uploads the module maps to
    the selected GPU, and then accepts per-event ``FEATURES`` tensors directly
    on that GPU.
    """

    def __init__(self, module_map_path, *, gpu_blocks: int = 512, gpu_device: int = 0):
        module = _load_native_module()
        if module is None:
            raise RuntimeError(
                "TAAS ModuleMapGraph CUDA extension is not available. Build it with "
                "`TAAS_BUILD_CUDA_EXT=1 TAAS_BUILD_MODULEMAP_GRAPH=1 "
                "python setup.py build_ext --inplace` in the modulemap image."
            ) from _IMPORT_ERROR

        self._native = module.ModuleMapGraphCuda(
            str(module_map_path), int(gpu_blocks), int(gpu_device)
        )

    def build(self, features) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features = features.detach().to(dtype=torch.float32).contiguous()
        return tuple(self._native.build(features))


__all__ = [
    "build_module_map_graph_cuda",
    "load_exact_module_ids",
    "ModuleMapGraphCuda",
    "module_map_graph_available",
]
