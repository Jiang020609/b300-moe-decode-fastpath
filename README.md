# B300 MoE Decode Fast Path

This repository currently provides Goal 1A: a deterministic, pure-PyTorch
reference and benchmark scaffold for a single-device Local-MoE decode path. It
does **not** contain custom CUDA kernels or published B300 performance numbers.

## Design

The staged reference in `baseline/torch_moe.py` follows this flow:

1. Select `top_k` experts per token and softmax only the selected logits.
2. Flatten token/expert assignments and stably sort them into expert-major order.
3. Run a two-linear SwiGLU FFN independently for every non-empty expert.
4. Multiply each assignment by its routing weight and `index_add_` it back to the
   original token row.

Weights use PyTorch linear layout: gate/up are `[experts, intermediate, hidden]`
and down is `[experts, hidden, intermediate]`. `local_moe` returns the output and
`LocalMoEMetadata`, whose routing record includes top-k decisions, expert counts
and offsets, the forward permutation, and its reverse mapping. A deliberately
direct token/rank-loop `naive_local_moe` serves as an independent correctness
oracle.

`baseline/routing.py` generates three seeded workloads:

- `uniform`: assignments cycle evenly through all experts.
- `skewed`: assignments follow a long-tailed inverse-rank distribution.
- `hot_expert`: assignments heavily favor a small leading set (large enough to
  hold distinct top-k choices).

Random inputs are generated on CPU before transfer, making routing targets
reproducible for a fixed seed across CPU and CUDA devices.

## Setup and tests

Python 3.10+ is recommended.

```bash
python -m pip install -r requirements.txt
python -m pytest -q
```

The tests cover output shapes, routing normalization, permutation round trips,
`top_k=1,2,4`, empty experts, severe imbalance, invalid inputs, and staged versus
naive numerical agreement.

## Benchmark

Run the CPU-friendly quick sweep:

```bash
python benchmark/benchmark.py \
  --config configs/quick.yaml \
  --device cpu \
  --output results/quick_cpu.csv
```

When CUDA is available:

```bash
python benchmark/benchmark.py \
  --config configs/quick.yaml \
  --device cuda \
  --output results/quick_cuda.csv
```

The benchmark separately measures top-k routing, token grouping, expert FFN,
expert combine, and the total layer. It performs configured warmups, prints P50,
P90, and P99 latency, and writes every measured iteration to CSV. CPU timing uses
`perf_counter_ns`; CUDA timing uses CUDA Events and explicitly synchronizes before
measurement and after every stop event.

`configs/b300.yaml` records the requested target shapes but is never selected by
default. It is intentionally only a workload description, not a claim about B300
performance. Its fp32 expert tensors can require tens of GiB, and the full sweep
is costly.

## Current limitations and next stage

This is a correctness-first reference. It loops over experts in Python, creates
intermediate tensors, uses fp32 by default, and does not model communication or
multi-GPU expert parallelism. Stage timings are isolated microbenchmarks, so their
sum need not equal end-to-end latency. CUDA results describe ordinary PyTorch
operators on the machine used; they are not fused-kernel results.

The next stage can preserve these APIs and CSV fields while replacing routing,
grouping, grouped GEMM, and combine with validated sm_103a CUDA fast paths, using
the naive and staged references as correctness oracles.
