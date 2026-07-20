#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
python_bin="${PYTHON_BIN:-python}"

if ! command -v "${python_bin}" >/dev/null 2>&1; then
  echo "Python interpreter not found: ${python_bin} (set PYTHON_BIN explicitly)" >&2
  exit 1
fi
python_bin="$(command -v "${python_bin}")"

cuda_root="${CUDA_HOME:-${CUDA_PATH:-}}"
if [[ -z "${cuda_root}" ]]; then
  echo "CUDA_HOME is required and must point to CUDA Toolkit 13.0 or newer" >&2
  exit 1
fi
if [[ ! -x "${cuda_root}/bin/nvcc" && ! -x "${cuda_root}/bin/nvcc.exe" ]]; then
  echo "nvcc was not found under ${cuda_root}/bin" >&2
  exit 1
fi
if [[ -z "${CUTLASS_PATH:-}" ]]; then
  echo "CUTLASS_PATH is required and must point to a CUTLASS checkout >=4.3.1" >&2
  exit 1
fi
if [[ ! -f "${CUTLASS_PATH}/include/cutlass/version.h" ]]; then
  echo "CUTLASS_PATH does not contain include/cutlass/version.h: ${CUTLASS_PATH}" >&2
  exit 1
fi
cuda_root="$(cd "${cuda_root}" && pwd)"
CUTLASS_PATH="$(cd "${CUTLASS_PATH}" && pwd)"
export CUDA_HOME="${cuda_root}"
export CUTLASS_PATH

cd "${repo_root}"
echo "Goal 1B interpreter: $(command -v "${python_bin}")"
echo "Goal 1B CUDA_HOME: ${cuda_root}"
echo "Goal 1B CUTLASS_PATH: ${CUTLASS_PATH}"
exec "${python_bin}" -m fastpath.build "$@"
