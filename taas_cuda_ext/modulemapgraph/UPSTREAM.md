# ModuleMapGraph / MMG Upstream Snapshot

This directory vendors the ModuleMapGraph/MMG CPU and CUDA source used by the
TAAS ModuleMap and EdgeLayerConnector CUDA extensions.

## Current Status

The tree is organized like the upstream MMG source layout:

- `CPU/`: CPU-side data structures, ROOT readers, graph and module-map helpers
- `GPU/`: CUDA graph construction, ModuleMap, graph, edge, track, and
  EdgeLayerConnector helpers

The TAAS wrappers are intentionally kept outside this vendored tree:

- `../module_map_graph.cu`
- `../edge_layer_connector.cu`
- `../module_map_graph.py`

Those wrappers adapt the vendored MMG code to Triton/PyTorch tensors, Athena
input layout, CUDA device placement, and TAAS output formatting.

## Provenance

The exact upstream source URL and commit/tag for this snapshot were not recorded
when the code was first vendored.

Known local version clue:

- MMG module-map compatibility version: `1.3`
- Source: `CPU/include/MMG/module_map_triplet.hpp` defines `_vs = 1.3`
- The reader checks ROOT branches named `MMG_min_version` in the `.triplets.root`
  and `.doublets.root` module-map files against this value.

Before the next refresh, fill in:

- Upstream repository:
- Upstream commit/tag:
- Snapshot date:
- Person performing refresh:

## Local TAAS Patches

This directory is intended to remain an upstream-style source snapshot. TAAS
patches are stored outside this tree and applied to a temporary build copy when
needed:

- `../mmg_patches/README.md`

Keep new TAAS-specific behavior in the wrapper files outside this directory when
possible. If a change must be made inside `modulemapgraph/`, add an entry to the
patch ledger in `../mmg_patches/`.

## Refresh Procedure

When updating to a newer MMG snapshot:

1. Save the current tree or work on a branch.
2. Replace the upstream-owned files in `CPU/` and `GPU/` with the new MMG source.
3. Reapply the documented TAAS patches from `../mmg_patches/README.md`.
4. Rebuild the CUDA extensions in the target Triton image:

   ```bash
   MAX_JOBS=1 TAAS_BUILD_CUDA_EXT=1 TAAS_BUILD_MODULEMAP_GRAPH=1 \
     TAAS_BUILD_MODULEMAP_PROBE=0 python setup.py build_ext --inplace
   ```

5. Run fixed-event regression checks:

   - ModuleMap graph edge count
   - kept edge count after ONNX
   - EdgeLayerConnector track count
   - one-event Athena/Triton smoke test

6. Record the new upstream repository, commit/tag, date, local patches, and
   regression numbers in this file.
