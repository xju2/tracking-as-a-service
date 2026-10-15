# TAAS Runtime Package

`taas` contains implementation code shared by the Triton model repositories in
`models/`. The versioned `models/<name>/<version>/` directories remain the
composition boundary: `model.py` handles Triton integration and `inference.py`
owns pipeline initialization and execution.

## Layout

- `graph.py`: tensor-only graph interchange type
- `graph_construction/`: ModuleMap, metric-learning, and double-metric-learning
  graph builders; `_frnn.py` is their shared nearest-neighbor primitive
- `edge_classification/`: one profile-driven Torch filter runner and
  Torch/ONNX InteractionGNN adapters
- `graph_segmentation/`: one NetworkX Walkthrough implementation, one
  profile-driven fastWalkthrough implementation, D-WALK, junction removal,
  ACTS junction removal, and EdgeLayerConnector
- `torch_models/`: reusable checkpoint-compatible model definitions and layers
- `model_loading/`: explicit TorchScript, checkpoint, and ONNX loaders
- `cuda_ext/`: Python wrappers, CUDA/C++ sources, vendored MMG snapshot, and
  TAAS patch ledger
- `triton/`: small backend configuration, tensor, response, and timing helpers

## Compatibility Policy

Refactoring a model into this package must not change its configuration names,
defaults, feature ordering/scaling, threshold comparison operators, edge
ordering, graph-cleaning order, device placement, or flattened track output.
Historical differences are represented by explicit profiles when they can share
an implementation; genuinely different algorithms remain separate. Shared
objects are initialized once by a Triton model instance and must not introduce
per-request model/session loading or implicit device transfers.

See `COMPATIBILITY.md` for the version-to-implementation map.
