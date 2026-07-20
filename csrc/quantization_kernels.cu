#include "moe_fastpath.h"

// PyTorch 2.12 exposes BF16 grouped GEMM through ATen, but its public native
// header does not expose an NVFP4 grouped entry point. Do not label a BF16 or
// FP8 kernel as NVFP4. This capability remains false until an SM103 block-scaled
// CUTLASS adapter with the real E2M1 payload and both scale levels is compiled.
bool nvfp4_grouped_gemm_available_cuda() {
  return false;
}

std::string nvfp4_grouped_gemm_unavailable_reason_cuda() {
  return "SM103 NVFP4 grouped GEMM adapter is not compiled; no fallback was used";
}
