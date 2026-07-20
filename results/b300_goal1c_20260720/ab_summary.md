# Goal 1C B300 A/B Results

- GPU: NVIDIA B300 SXM6 AC
- Source commit: `92fea90`
- Backend: `cutlass_bf16`
- Shape: H=4096, I=14336, E=64, top-k=2
- Tokens: 1, 4, 8, 32
- Workloads: uniform, hotspot
- Warmup/repetitions: inherited from `configs/b300.yaml`
- Fallback used: no

## End-to-end latency

| Tokens | Workload | Grouped P50 (us) | Auto P50 (us) | P50 reduction | P90 reduction | P99 reduction | Gate-up | Down |
|---:|---|---:|---:|---:|---:|---:|---|---|
| 1 | hotspot | 447.184 | 413.600 | 7.51% | 3.93% | 6.93% | per_expert | per_expert |
| 1 | uniform | 430.688 | 411.712 | 4.41% | 2.75% | 21.42% | per_expert | per_expert |
| 4 | hotspot | 718.928 | 666.752 | 7.26% | 7.10% | 8.41% | per_expert | per_expert |
| 4 | uniform | 889.056 | 795.776 | 10.49% | 9.73% | 9.27% | per_expert | per_expert |
| 8 | hotspot | 729.360 | 696.096 | 4.56% | 3.43% | -4.69% | per_expert | per_expert |
| 8 | uniform | 1468.832 | 1352.288 | 7.93% | 6.55% | -5.50% | per_expert | per_expert |
| 32 | hotspot | 1361.728 | 1215.184 | 10.76% | 10.48% | 10.12% | per_expert | per_expert |
| 32 | uniform | 5108.224 | 4404.112 | 13.78% | 14.12% | 16.77% | per_expert | grouped |

## Aggregate result

- Auto P50 wins: **8/8 cases**
- Geometric-mean P50 speedup: **1.0916x**
- Mean P50 latency reduction: **8.34%**

## GEMM stage comparison

| Tokens | Workload | Gate grouped (us) | Gate auto (us) | Gate reduction | Down grouped (us) | Down auto (us) | Down reduction |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | hotspot | 169.248 | 100.256 | 40.76% | 98.736 | 63.408 | 35.78% |
| 1 | uniform | 154.048 | 98.624 | 35.98% | 98.768 | 62.704 | 36.51% |
| 4 | hotspot | 344.704 | 262.768 | 23.77% | 194.352 | 154.320 | 20.60% |
| 4 | uniform | 493.024 | 347.712 | 29.47% | 210.816 | 198.464 | 5.86% |
| 8 | hotspot | 352.896 | 268.512 | 23.91% | 195.776 | 153.568 | 21.56% |
| 8 | uniform | 873.600 | 678.544 | 22.33% | 414.704 | 382.944 | 7.66% |
| 32 | hotspot | 787.312 | 603.824 | 23.31% | 390.368 | 328.048 | 15.96% |
| 32 | uniform | 3320.464 | 2642.336 | 20.42% | 1571.056 | 1520.528 | 3.22% |

## Conclusion

The Goal 1C hybrid policy improved end-to-end P50 latency in all eight measured cases. The policy selected per-expert matmul for gate-up in every case, while the down GEMM returned to grouped execution for the T=32 uniform case. No fallback was used.
