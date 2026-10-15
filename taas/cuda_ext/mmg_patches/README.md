# TAAS Patches For Vendored MMG

Patch files in this directory are applied to a temporary copy of
`../modulemapgraph/` during selected TAAS extension builds. They are kept
outside the vendored source tree so `modulemapgraph/` can remain an
upstream-style MMG snapshot.

## Patches

- `no_root_guards.patch`: adds `TAAS_MMG_NO_ROOT` guards around MMG helpers that
  depend on ROOT/CPU track and edge types. This lets the lightweight standalone
  EdgeLayerConnector extension build against only the CUDA graph/connector path.

## Build Behavior

`setup.py` copies `modulemapgraph/` into:

```text
build/taas_modulemapgraph_patched/
```

and applies these patches there before compiling
`taas.cuda_ext._edge_layer_connector_cuda`.

The source tree under `modulemapgraph/` is not modified by the build.
