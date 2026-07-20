#!/usr/bin/env bash
set -euo pipefail

# Legacy dispatch/combine V0 cross-build only. This does not build Goal 1B's
# grouped-GEMM extension; use scripts/build_goal_1b.sh for that exact target.
echo "Building legacy dispatch/combine V0 for sm_103a (not Goal 1B)" >&2

if [[ -n "${CUDA_HOME:-}" ]]; then
  nvcc_bin="${CUDA_HOME}/bin/nvcc"
else
  nvcc_bin="$(command -v nvcc || true)"
  if [[ -n "${nvcc_bin}" ]]; then
    CUDA_HOME="$(cd "$(dirname "${nvcc_bin}")/.." && pwd)"
    export CUDA_HOME
  fi
fi

if [[ -z "${nvcc_bin:-}" || ! -x "${nvcc_bin}" ]]; then
  echo "CUDA_HOME/bin/nvcc is required for the B300 build" >&2
  exit 1
fi

cuda_release="$("${nvcc_bin}" --version | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | tail -n 1)"
if [[ -z "${cuda_release}" ]]; then
  echo "Could not determine the CUDA toolkit version" >&2
  exit 1
fi

python - "${cuda_release}" <<'PY'
import sys
major, minor = (int(part) for part in sys.argv[1].split(".")[:2])
if (major, minor) < (12, 9):
    raise SystemExit("B300 sm_103a requires CUDA 12.9 or newer")
PY

# Keep the default extension portable; only this explicit build adds sm_103a.
MOE_BUILD_CUDA=1 MOE_B300_SM103A=1 python -m pip install -e .
