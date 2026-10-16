#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

TAAS_BUILD_CUDA_EXT=1 python setup.py build_ext --inplace
