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
export PYTHON_BIN="${python_bin}"

cd "${repo_root}"
bash "${script_dir}/build_goal_1b.sh"

shopt -s nullglob
goal_tests=(tests/test_goal_1b_*.py)
if (( ${#goal_tests[@]} == 0 )); then
  echo "No Goal 1B tests found under tests/test_goal_1b_*.py" >&2
  exit 1
fi
if [[ ! -f benchmark/benchmark_goal_1b.py ]]; then
  echo "Missing benchmark/benchmark_goal_1b.py" >&2
  exit 1
fi

"${python_bin}" -m pytest "${goal_tests[@]}" -v
"${python_bin}" benchmark/benchmark_goal_1b.py \
  --config configs/quick.yaml \
  --device cuda \
  --backend cutlass_bf16 \
  --dtype bfloat16 \
  --tokens 1,8 \
  --experts 8 \
  --top-k 2 \
  --workloads uniform,hotspot,zipf \
  --warmup 3 \
  --repeats 10 \
  --output results/goal_1b_smoke_b300.csv
