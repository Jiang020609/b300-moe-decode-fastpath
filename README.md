# B300 MoE Decode Fast Path

This repository preserves the Goal 1A PyTorch Local-MoE baseline and adds the
Goal 1B API, stable CUDA routing/permutation kernels, a two-grouped-GEMM BF16
execution path, reusable workspace, correctness tests, and a staged decode
benchmark.

Goal 1B is **not yet hardware-accepted**. The current development host has
PyTorch `2.12.0+cpu`, no NVIDIA GPU, CUDA toolkit, `nvcc`, or CUTLASS checkout.
Only the `torch` backend and CPU tests/benchmark have run here. The
`cutlass_bf16` source path has not been compiled or executed on B300, and the
real NVFP4 adapter is deliberately unavailable. No backend silently falls back.

## Public Goal 1B API

```python
from fastpath import B300MoEWorkspace, b300_moe_forward

workspace = B300MoEWorkspace(capacity_tokens=64, device=hidden_states.device)
output, metadata = b300_moe_forward(
    hidden_states,       # [T, H]
    expert_indices,      # [T, K], int64
    expert_weights,      # [T, K], used exactly as supplied
    gate_up_weight,      # [E, 2I, H]: gate first, then up
    down_weight,         # [E, H, I]
    num_experts=num_experts,
    top_k=top_k,
    backend="torch",    # torch | cutlass_bf16 | cutlass_nvfp4
    quant_mode="none",  # none | bf16 | nvfp4
    workspace=workspace,
    return_metadata=True,
)
```

The result is `[T, H]`. Metadata always includes `backend`, `architecture`,
`quant_mode`, and `used_fallback`; routing and workspace diagnostics are added
when requested. `top_k` supports 1, 2, 4, and 8. Routing weights need not be
normalized and may be negative.

`backend="torch"` is the portable, differentiable correctness path when no
workspace is supplied. The compiled backends are forward-only, require BF16
inputs for `cutlass_bf16`, require an actual CC `(10, 3)` device and a Goal 1B
`sm_103a` extension, and fail explicitly when any requirement is missing.

## Data flow and semantics

For token-major assignment `a = token * K + slot`, routing builds a stable
expert-major order and two exact inverse mappings:

- `permutation`: expert-major row to token-major assignment;
- `reverse_mapping`: token-major assignment to expert-major row.

The intended compiled BF16 path is:

```text
explicit routes
  -> device histogram / offsets / stable mappings
  -> permuted activations
  -> grouped BF16 gate+up GEMM
  -> standalone FP32-compute SwiGLU
  -> grouped BF16 down GEMM
  -> reverse-map, routing-weighted FP32-accumulate combine
  -> output
```

Empty experts retain equal adjacent offsets and issue no per-expert descriptor
from Python. The grouped GEMM wrapper calls PyTorch's internal
`at::cuda::detail::bf16bf16_grouped_mm` output-buffer API. That PyTorch API is
CUTLASS-backed in the targeted CUDA build, but is an internal, version-coupled
dependency. The external CUTLASS checkout is used for SM103/version build gates;
the exact PyTorch binary and target host still have to be verified by compiling
and running the B300 tests.

The NVFP4 source currently returns capability `false`. There is no E2M1 payload,
two-level scale layout, prepacked weight cache, quantization kernel, or NVFP4
grouped GEMM in this revision, so `backend="cutlass_nvfp4"` always fails instead
of substituting BF16.

## Environment audit and build

The complete observed/required environment split is in
[`docs/goal_1b_environment.md`](docs/goal_1b_environment.md). On any machine,
run the non-mutating preflight first:

```bash
python -m fastpath.build --check-only
```

The Goal 1B build is intentionally strict: CUDA 13.x, a CUDA-enabled PyTorch
build, a visible CC `(10, 3)` GPU, CUTLASS 4.3.1 or newer with `Sm103`, and exact
`-gencode=arch=compute_103a,code=sm_103a`. It does not emit another architecture.

```bash
export CUDA_HOME=/path/to/cuda-13.x
export CUTLASS_PATH=/path/to/cutlass
bash scripts/build_goal_1b.sh

# Equivalent Python entry point after setting the same environment:
python -m fastpath.build
```

The older generic dispatch/combine V0 remains available independently through
`MOE_BUILD_CUDA=1 python -m pip install -e .`; it is not evidence that Goal 1B
or `sm_103a` was built.

## Correctness tests

CPU/reference and build-contract tests:

```bash
python -m pytest -q
python -m pytest tests/test_goal_1b_correctness.py \
  tests/test_goal_1b_routing.py tests/test_goal_1b_edge_cases.py -v
```

On a built B300 host:

```bash
python -m pytest tests/test_goal_1b_cuda.py -v
bash scripts/run_goal_1b_smoke.sh

# Opt-in realistic H=4096, I=14336, E=64 correctness case (~21 GiB weights):
MOE_RUN_B300_LARGE=1 python -m pytest \
  tests/test_goal_1b_cuda.py -k b300_config_target_shape -v
```

CUDA tests skip unless all of CUDA, CC `(10, 3)`, and a Goal 1B extension are
present. A skip is not a B300 pass. The test matrix covers `T=1..64`,
`K=1/2/4/8`, uniform/hotspot/Zipf routing, empty experts, concentrated routing,
unnormalized weights, repeatability, workspace reuse, and streams. An irregular
small shape is covered by the torch reference; compiled grouped-GEMM alignment
or padding remains a target-host verification item.

## Staged benchmark

The Goal 1B benchmark writes one P50/P90/P99 CSV row for each required stage:
`routing_us`, `permutation_us`, `quantization_us`, `gate_up_gemm_us`,
`swiglu_us`, `down_gemm_us`, `combine_us`, and `total_us`. It records requested
and actual backends, fallback status, GPU/toolchain metadata, quant mode, timing
method, first device execution, and steady-state percentiles.

Tiny CPU smoke (safe on the current host):

```bash
python benchmark/benchmark_goal_1b.py \
  --config configs/quick.yaml \
  --device cpu --backend torch \
  --tokens 1 --experts 8 --top-k 1 --workloads uniform \
  --hidden-size 8 --intermediate-size 12 \
  --warmup 0 --repeats 3 \
  --output results/goal_1b_cpu_smoke.csv
```

Target-host sweeps:

```bash
python benchmark/benchmark_goal_1b.py \
  --config configs/quick.yaml \
  --device cuda --backend torch,cutlass_bf16 \
  --dtype bfloat16 \
  --output results/goal_1b_quick_b300.csv

python benchmark/benchmark_goal_1b.py \
  --config configs/b300.yaml \
  --device cuda --backend cutlass_bf16 \
  --output results/goal_1b_target_b300.csv
```

`configs/b300.yaml` is not a CPU smoke configuration: its largest expert-weight
set is roughly 42 GiB in BF16. Weight creation is outside all timers. The torch
stage GEMMs include Python per-expert iteration and allocation, while compiled
stages use workspace buffers; the rows are end-to-end stage costs, not identical
low-level GEMM-only experiments. CUDA “cold start” means first measured device
execution after input construction, not full process startup or JIT compilation.

## Current limitations

- B300 compilation, BF16 correctness, and B300 latency are unverified on this
  CPU-only host; therefore the Goal 1B hardware acceptance criteria are open.
- The BF16 adapter depends on an internal PyTorch C++ symbol and on that PyTorch
  build containing a suitable SM103 grouped-GEMM implementation.
- Real NVFP4 quantization, scale handling, weight prepacking, descriptor caching,
  correctness tests, and benchmark are not implemented.
- Compiled irregular-dimension alignment/padding is not yet proven.
- Routing is stable and device-only, but the first implementation scans all
  assignments once per expert; it is correctness-oriented rather than tuned.
- No vLLM/SGLang integration, multi-GPU expert parallelism, communication,
  scheduler, or KV-cache work is in scope.

See [`docs/goal_1b_report.md`](docs/goal_1b_report.md) for the detailed delivery
report and the B300 validation checklist.
