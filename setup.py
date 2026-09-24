import os
import shutil
import subprocess

from setuptools import setup


def env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def prepare_patched_modulemapgraph(repo_root, source_root):
    """Create a patched build copy of the vendored MMG tree for TAAS ELC."""

    patch_path = os.path.join(
        repo_root,
        "taas_cuda_ext",
        "mmg_patches",
        "no_root_guards.patch",
    )
    patched_root = os.path.join(repo_root, "build", "taas_modulemapgraph_patched")

    if os.path.exists(patched_root):
        shutil.rmtree(patched_root)
    shutil.copytree(source_root, patched_root)

    try:
        subprocess.run(
            ["patch", "-p1", "-i", patch_path],
            cwd=patched_root,
            check=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "The `patch` executable is required to prepare the TAAS-patched "
            "ModuleMapGraph build tree."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Failed to apply TAAS MMG patch "
            f"{patch_path} to temporary build tree {patched_root}."
        ) from exc

    return patched_root


def get_extension_config():
    build_cuda_ext = env_flag("TAAS_BUILD_CUDA_EXT")
    if not build_cuda_ext:
        return [], {}

    try:
        from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
    except ImportError as exc:
        raise RuntimeError(
            "TAAS_BUILD_CUDA_EXT=1 requires torch to be importable at build time. "
            "Build in the target environment with --no-build-isolation, or run "
            "`TAAS_BUILD_CUDA_EXT=1 python setup.py build_ext --inplace`."
        ) from exc

    compile_args = {
        "cxx": ["-O3"],
        "nvcc": ["-O3", "--use_fast_math", "--std=c++20", "-DTAAS_MMG_NO_ROOT"],
    }
    repo_root = os.path.abspath(os.path.dirname(__file__))
    extension_root = os.path.join(repo_root, "taas_cuda_ext")
    modulemapgraph_root = os.environ.get(
        "TAAS_MODULEMAPGRAPH_ROOT",
        os.path.join(extension_root, "modulemapgraph"),
    )
    modulemapgraph_root = os.path.abspath(modulemapgraph_root)
    elc_modulemapgraph_root = prepare_patched_modulemapgraph(
        repo_root, modulemapgraph_root
    )
    elc_modulemapgraph_include_dirs = [
        os.path.join(elc_modulemapgraph_root, "GPU", "include"),
        os.path.join(elc_modulemapgraph_root, "GPU", "src"),
        os.path.join(elc_modulemapgraph_root, "CPU", "include"),
        os.path.join(elc_modulemapgraph_root, "CPU", "src"),
    ]
    extensions = [
        CUDAExtension(
            name="taas_cuda_ext._connected_components_cuda",
            sources=[
                f"{extension_root}/connected_components.cpp",
                f"{extension_root}/connected_components_kernel.cu",
            ],
            extra_compile_args=compile_args,
        ),
        CUDAExtension(
            name="taas_cuda_ext._reverse_topological_dp_cuda",
            sources=[
                f"{extension_root}/reverse_topological_dp.cpp",
                f"{extension_root}/reverse_topological_dp_kernel.cu",
            ],
            extra_compile_args=compile_args,
        ),
        CUDAExtension(
            name="taas_cuda_ext._trace_selected_paths_cuda",
            sources=[
                f"{extension_root}/trace_selected_paths.cpp",
                f"{extension_root}/trace_selected_paths_kernel.cu",
            ],
            extra_compile_args=compile_args,
        ),
        CUDAExtension(
            name="taas_cuda_ext._edge_layer_connector_cuda",
            sources=[
                f"{extension_root}/edge_layer_connector.cu",
            ],
            include_dirs=elc_modulemapgraph_include_dirs,
            extra_compile_args=compile_args,
        ),
    ]
    rootsys = os.environ.get("ROOTSYS")
    onnxruntime_root = os.environ.get("ONNXRUNTIME_ROOT")
    build_modulemap_probe = env_flag(
        "TAAS_BUILD_MODULEMAP_PROBE",
        default=bool(rootsys and onnxruntime_root),
    )
    if build_modulemap_probe:
        missing = [
            name
            for name, value in {
                "ROOTSYS": rootsys,
                "ONNXRUNTIME_ROOT": onnxruntime_root,
            }.items()
            if not value
        ]
        if missing:
            raise RuntimeError(
                "TAAS_BUILD_MODULEMAP_PROBE requires "
                + ", ".join(missing)
                + " to be set."
            )

        root_include = os.path.join(rootsys, "include")
        root_lib = os.path.join(rootsys, "lib")
        onnxruntime_include = os.path.join(onnxruntime_root, "include")
        onnxruntime_lib = os.path.join(onnxruntime_root, "lib")
        extensions.append(
            CppExtension(
                name="taas_cuda_ext._modulemap_dependency_probe",
                sources=[
                    f"{extension_root}/modulemap_dependency_probe.cpp",
                ],
                include_dirs=[
                    root_include,
                    onnxruntime_include,
                ],
                library_dirs=[
                    root_lib,
                    onnxruntime_lib,
                ],
                libraries=[
                    "Core",
                    "onnxruntime",
                ],
                extra_compile_args={
                    "cxx": ["-O3", "-std=c++17"],
                },
                extra_link_args=[
                    f"-Wl,-rpath,{root_lib}",
                    f"-Wl,-rpath,{onnxruntime_lib}",
                ],
            )
        )
    build_modulemap_graph = env_flag("TAAS_BUILD_MODULEMAP_GRAPH")
    if build_modulemap_graph:
        missing = [
            name
            for name, value in {
                "ROOTSYS": rootsys,
            }.items()
            if not value
        ]
        if missing:
            raise RuntimeError(
                "TAAS_BUILD_MODULEMAP_GRAPH requires "
                + ", ".join(missing)
                + " to be set."
            )

        root_include = os.path.join(rootsys, "include")
        root_lib = os.path.join(rootsys, "lib")
        extensions.append(
            CUDAExtension(
                name="taas_cuda_ext._module_map_graph_cuda",
                sources=[
                    f"{extension_root}/module_map_graph.cu",
                ],
                include_dirs=[
                    root_include,
                    os.path.join(modulemapgraph_root, "GPU", "include"),
                    os.path.join(modulemapgraph_root, "GPU", "src"),
                    os.path.join(modulemapgraph_root, "CPU", "include"),
                    os.path.join(modulemapgraph_root, "CPU", "src"),
                ],
                library_dirs=[
                    root_lib,
                    "/usr/local/lib",
                ],
                libraries=[
                    "Core",
                    "RIO",
                    "Tree",
                    "Hist",
                    "MathCore",
                    "boost_program_options",
                    "boost_regex",
                    "boost_filesystem",
                ],
                extra_compile_args={
                    "cxx": ["-O3", "-std=c++20"],
                    "nvcc": [
                        "-O3",
                        "--use_fast_math",
                        "--std=c++20",
                        "--extended-lambda",
                    ],
                },
                extra_link_args=[
                    f"-Wl,-rpath,{root_lib}",
                    "-Wl,-rpath,/usr/local/lib",
                ],
            )
        )
    return extensions, {"build_ext": BuildExtension}


ext_modules, cmdclass = get_extension_config()

setup(
    packages=["taas_cuda_ext", "taas_graph_segmentation"],
    package_dir={
        "taas_cuda_ext": "taas_cuda_ext",
        "taas_graph_segmentation": "taas_graph_segmentation",
    },
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
