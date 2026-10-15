# Pipeline Compatibility Map

The files in `models/` remain the public Triton model repository. This map
records the shared implementation selected by each deployed version after the
package refactor. Each model now exposes only version 1.

| Triton model | Version | Graph construction | Edge classification | Segmentation |
| --- | --- | --- | --- | --- |
| MetricLearning | 1 | shared FRNN, same-space embedding | shared filter, `METRIC_CURRENT`, and shared Torch definitions | shared fastWalkthrough, `CURRENT` |
| DoubleMetricLearning | 1 | shared FRNN, cross-space embedding | shared filter, `DOUBLE_METRIC_CURRENT` | shared fastWalkthrough, D-WALK, JR, ACTS JR, or ELC |
| GNN4Pixel | 1 | shared FRNN, same-space embedding | TorchScript filter and InteractionGNN | shared fastWalkthrough, `LEGACY_LONGEST` |
| ModuleMap | 1 | `ModuleMapGraphConstructor` and native MMG CUDA | `InteractionGNNOnnx` with CUDA I/O binding | shared fastWalkthrough (`CURRENT`), D-WALK, ACTS JR, or ELC |

The explicit compatibility profiles retain historical differences in autocast
handling, edge sorting, batching, cloning, graph cleanup, cut behavior, and
fastWalkthrough path resolution without duplicating complete modules.

## Stable External Surface

- Every deployed model uses version 1.
- Each Triton `model.py` is an adapter; pipeline execution lives in its
  adjacent `inference.py`.
- Parameter names, instance groups, score cuts,
  feature lists/scales, and output contracts remain config/version owned.
- Checkpoints and ModuleMap artifacts remain at their existing paths. Moving
  them is deliberately deferred because paths are part of operational scripts
  and deployed configuration.
- CUDA extension import names are `taas.cuda_ext.*`; rebuild them only when
  their C++/CUDA sources or extension names change.
