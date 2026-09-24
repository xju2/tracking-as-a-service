# TAAS CUDA Extensions

This package vendors the CUDA/C++ kernels used by DoubleMetricLearning D-WALK:

- weak connected components
- reverse-topological dynamic programming
- selected path tracing
- ModuleMapGraph EdgeLayerConnector

Build the extensions in the target runtime environment, after PyTorch and CUDA/NVCC
are available:

```bash
./scripts/build_taas_cuda_ext.sh
```

This runs:

```bash
TAAS_BUILD_CUDA_EXT=1 python setup.py build_ext --inplace
```

The compiled `.so` files are written into this directory. If they are absent,
`dwalk.py` still imports successfully and uses the CPU/SciPy fallbacks unless
`dwalk_cc_backend=custom_cuda` is requested.

The EdgeLayerConnector extension vendors the ModuleMapGraph GPU/CPU headers and
source snippets under `modulemapgraph/` and exposes the CUDA track-building route as
`taas_cuda_ext._edge_layer_connector_cuda`. It is required when
`graph_segmentation_method=edge_layer_connector`.

The vendored ModuleMapGraph/MMG tree is documented in
`modulemapgraph/UPSTREAM.md`. TAAS-local patches for that tree are stored
outside the vendored source in `mmg_patches/`; keep new Triton/PyTorch adapter
logic in the wrappers outside `modulemapgraph/` whenever possible.
