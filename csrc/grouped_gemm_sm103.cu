#include "moe_fastpath.h"

#include <ATen/ATen.h>
#include <ATen/native/cuda/GroupMM.h>
#include <c10/cuda/CUDAGuard.h>

#include <cutlass/arch/arch.h>
#include <cutlass/version.h>

#include <optional>

static_assert(
    CUTLASS_MAJOR > 4 ||
        (CUTLASS_MAJOR == 4 && CUTLASS_MINOR > 3) ||
        (CUTLASS_MAJOR == 4 && CUTLASS_MINOR == 3 && CUTLASS_PATCH >= 1),
    "Goal 1B requires CUTLASS 4.3.1 or newer");
static_assert(
    cutlass::arch::Sm103::kMinComputeCapability == 103,
    "selected CUTLASS checkout does not expose the SM103 architecture tag");

void grouped_gemm_bf16_out_cuda(
    torch::Tensor activations,
    torch::Tensor packed_weights,
    torch::Tensor expert_offsets_i32,
    torch::Tensor output) {
  TORCH_CHECK(activations.is_cuda() && packed_weights.is_cuda() && expert_offsets_i32.is_cuda() && output.is_cuda(), "grouped GEMM tensors must be CUDA tensors");
  TORCH_CHECK(activations.device() == packed_weights.device() && activations.device() == expert_offsets_i32.device() && activations.device() == output.device(), "grouped GEMM tensors must share a CUDA device");
  TORCH_CHECK(activations.scalar_type() == at::kBFloat16 && packed_weights.scalar_type() == at::kBFloat16 && output.scalar_type() == at::kBFloat16, "BF16 grouped GEMM requires bfloat16 activations, weights, and output");
  TORCH_CHECK(expert_offsets_i32.scalar_type() == at::kInt, "CUTLASS grouped GEMM offsets must be int32");
  TORCH_CHECK(activations.dim() == 2, "activations must have shape [A, K]");
  TORCH_CHECK(packed_weights.dim() == 3, "packed_weights must have shape [E, K, N]");
  TORCH_CHECK(expert_offsets_i32.dim() == 1 && expert_offsets_i32.numel() == packed_weights.size(0), "offsets must contain one cumulative end offset per expert");
  TORCH_CHECK(activations.size(1) == packed_weights.size(1), "grouped GEMM K dimension mismatch");
  TORCH_CHECK(
      activations.size(1) % 8 == 0 && packed_weights.size(2) % 8 == 0,
      "BF16 grouped GEMM K and N must be divisible by 8 for 16-byte alignment");
  TORCH_CHECK(output.sizes() == at::IntArrayRef({activations.size(0), packed_weights.size(2)}), "grouped GEMM output shape mismatch");
  TORCH_CHECK(activations.is_contiguous() && expert_offsets_i32.is_contiguous() && output.is_contiguous(), "grouped GEMM activations, offsets, and output must be contiguous");
  TORCH_CHECK(
      packed_weights.stride(1) == 1,
      "packed_weights must be a column-major [E, K, N] transpose view");

  const c10::cuda::CUDAGuard guard(activations.device());
  // PyTorch's CUDA implementation is the CUTLASS BF16 grouped-GEMM path. The
  // detail API is used intentionally because it accepts preallocated output,
  // allowing B300MoEWorkspace to reuse the large intermediate buffers.
  at::cuda::detail::bf16bf16_grouped_mm(
      activations,
      packed_weights,
      std::optional<at::Tensor>(expert_offsets_i32),
      std::nullopt,
      output);
}
