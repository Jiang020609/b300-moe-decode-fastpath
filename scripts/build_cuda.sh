#!/usr/bin/env bash
set -euo pipefail

# Generic CUDA build. PyTorch selects the visible GPU architecture by default;
# TORCH_CUDA_ARCH_LIST may be set by the caller for a cross-build.
MOE_BUILD_CUDA=1 python -m pip install -e .
