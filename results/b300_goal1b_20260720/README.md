# Goal 1B B300 Validation

Validation date: 2026-07-20

## Environment

- GPU: NVIDIA B300 SXM6 AC
- Compute capability: 10.3
- Target architecture: `sm_103a`
- CUDA Toolkit: 13.0.88
- PyTorch: 2.12.1+cu130
- CUTLASS: 4.3.1
- Dtype: BF16

## Correctness

- Full CUDA correctness suite with large case: 94 passed
- No backend fallback
- Empty-expert and repeated-offset grouped-MM contract passed

## End-to-end benchmark

- 42 cases
- 336 stage summaries
- Tokens: 1, 2, 4, 8, 16, 32, 64
- Experts: 64, 128
- Workloads: uniform, hotspot, zipf
- Backend: cutlass_bf16
- Validation result: PASS

## GEMM-only benchmark

Compared:

- `torch._grouped_mm`
- Per-active-expert `torch.matmul`

Shapes:

- gate-up: K=4096, N=28672
- down: K=14336, N=4096
- M=1,2,4,8,16,32,64
- E=8,64

Results:

- Grouped-MM wins: 2/28
- Per-expert matmul wins: 26/28
- Grouped-MM crossover cases:
  - down, E=64, M=32: 1.0361x
  - down, E=64, M=64: 1.0508x

This establishes the baseline for Goal 1C hybrid dispatch and native
B300-optimized grouped GEMM work.
