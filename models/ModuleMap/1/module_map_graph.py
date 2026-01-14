import torch
import numpy as np
from pymmg import GraphBuilder
import torch.cuda.nvtx as nvtx
import warnings
import contextlib
import io
from pathlib import Path


# Global instance of GraphBuilder
_graph_builder = None


@contextlib.contextmanager
def _suppress_output():
    """Silence stdout/stderr (used to hide verbose C++ prints)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        yield


def _resolve_base_path(module_map_pattern_path: str) -> Path:
    """Resolve the base path (without extension) for ModuleMap files.

    Accepts base name, directory, or a path with `.triplets.root`/`.doublets.root`/`.root`.
    Returns a Path pointing to the base without any extension.
    """
    p = Path(module_map_pattern_path)
    if p.is_dir():
        return p / p.name
    name = p.name
    if name.endswith(".triplets.root"):
        name = name[: -len(".triplets.root")]
        return p.with_name(name)
    if name.endswith(".doublets.root"):
        name = name[: -len(".doublets.root")]
        return p.with_name(name)
    if name.endswith(".root"):
        name = name[: -len(".root")]
        return p.with_name(name)
    return p


def _triplets_file_from_base(base: Path) -> Path:
    return Path(str(base) + ".triplets.root")


def get_graph_builder(module_map_pattern_path=None, device=0):
    """Get or create a GraphBuilder instance for the triplets file.

    Returns False if the file is missing or is a Git LFS pointer.
    """
    global _graph_builder
    if _graph_builder is None:
        if module_map_pattern_path is None or module_map_pattern_path == ".":
            module_map_pattern_path = str(Path(__file__).parent / "ModuleMap_rel24_ttbar_v9_89809evts_tol1e-10")

        base_path = _resolve_base_path(module_map_pattern_path)

        # If relative path, resolve relative to this file's directory
        if not base_path.is_absolute():
            base_path = (Path(__file__).parent / base_path).resolve()

        triplets_path = _triplets_file_from_base(base_path)

        # Check existence
        if not triplets_path.exists():
            warnings.warn(f"Module map triplets file not found: {triplets_path}. Disabling GraphBuilder.")
            _graph_builder = False
            return _graph_builder

        # Detect Git LFS pointer (tiny ascii file)
        try:
            if triplets_path.stat().st_size < 200:
                content = triplets_path.read_text(errors="ignore")
                if "git-lfs" in content:
                    warnings.warn(f"Module map file is a Git LFS pointer, not the data: {triplets_path}")
                    _graph_builder = False
                    return _graph_builder
        except Exception as e:
            warnings.warn(f"Could not inspect module map file: {e}. Disabling GraphBuilder.")
            _graph_builder = False
            return _graph_builder

        try:
            # GraphBuilder expects the base path; it will resolve required files internally
            with _suppress_output():
                _graph_builder = GraphBuilder(str(base_path), device)
        except Exception as e:
            warnings.warn(f"Failed to initialize GraphBuilder with {triplets_path}: {e}")
            _graph_builder = False

    return _graph_builder


def build_graph(graph, module_map_path=None, device=0):
    """Build edge index using pymmg GraphBuilder.

    If the module map files are not available (e.g., Git LFS not checked out),
    returns an empty edge index and logs a warning.
    """
    builder = get_graph_builder(module_map_pattern_path=module_map_path, device=device)

    if builder is False:
        warnings.warn("GraphBuilder not available. Returning empty edge index.")
        return torch.zeros((2, 0), dtype=torch.long)

    nvtx.range_push("build_edge_index")
    try:
        # Convert to CPU tensors with exact dtypes expected by pymmg
        # pymmg internally calls .numpy() on the tensors
        def to_tensor_uint64(x):
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().to(torch.int64)
            return torch.tensor(x, dtype=torch.int64)

        def to_tensor_float64(x):
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().to(torch.float64)
            return torch.tensor(x, dtype=torch.float64)

        hit_id_t = to_tensor_uint64(graph.hit_id)
        hit_module_id_t = to_tensor_uint64(graph.hit_module_id)
        hit_x_t = to_tensor_float64(graph.hit_x)
        hit_y_t = to_tensor_float64(graph.hit_y)
        hit_z_t = to_tensor_float64(graph.hit_z)
        nb_hits = int(hit_id_t.shape[0])

        # Call pymmg builder with CPU tensors - it will convert to numpy internally
        with _suppress_output():
            graph.edge_index = builder.build_edge_index(
                hit_id_t,
                hit_module_id_t,
                hit_x_t,
                hit_y_t,
                hit_z_t,
                nb_hits,
            )
    except Exception as e:
        warnings.warn(f"Error building edge index: {e}. Returning empty edge index.")
        graph.edge_index = torch.zeros((2, 0), dtype=torch.long)
    finally:
        nvtx.range_pop()

    if not isinstance(graph.edge_index, torch.Tensor):
        if isinstance(graph.edge_index, np.ndarray):
            graph.edge_index = torch.from_numpy(graph.edge_index)
    
    graph.edge_index = graph.edge_index.to(torch.long)
    return graph.edge_index
