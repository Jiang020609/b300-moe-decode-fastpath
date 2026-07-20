#include "moe_fastpath.h"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>

// Goal 1C fused device-side grouped GEMM for tiny decode M.
//
// Motivation (measured on B300, results/b300_goal1c_20260720): the per-expert
// matmul loop beats the grouped kernel at decode shapes, but paying one
// device->host offsets sync per forward ate ~2/3 of the kernel win. This
// kernel removes the sync entirely: every block reads the expert offsets from
// device memory, so the host never needs to know how tokens were routed.
// One launch replaces up to E cuBLAS launches, and the kernel is CUDA-graph
// capture safe because no host-side decision depends on routing data.
//
// Shape assumptions are the decode fast path's: rows per expert are tiny
// (usually 1-8), K and N are large, so the GEMM is bandwidth-bound on the
// weight matrix. The design therefore optimizes for streaming each weight
// column exactly once while amortizing it over all rows of that expert.

namespace {

constexpr int kThreads = 256;                       // 8 warps per block
constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = kThreads / kWarpSize;
// fp32 accumulators kept in registers per thread. Experts with more rows loop
// in chunks and re-stream the weight column once per chunk; decode routing
// rarely exceeds this per-expert row count, so the common case is one pass.
constexpr int kRowChunk = 8;
// 8 bf16 elements = 16 bytes, one vectorized global load.
constexpr int kElementsPerLoad = 8;
constexpr int kMaxGridX = 65535;

__global__ void fused_grouped_gemm_bf16_kernel(
    const __nv_bfloat16* __restrict__ activations,   // [A, K] row-major
    const __nv_bfloat16* __restrict__ weights,       // [E, N, K] contiguous
    const int64_t* __restrict__ expert_offsets,      // [E + 1] cumulative rows
    __nv_bfloat16* __restrict__ output,              // [A, N] row-major
    int64_t k_dim,
    int64_t n_dim) {
  // grid.y indexes the expert; the offsets read below is the whole trick:
  // it happens on-device, so no host synchronization ever exists.
  const int expert = blockIdx.y;
  const int64_t start = expert_offsets[expert];
  const int64_t end = expert_offsets[expert + 1];
  if (end <= start) {
    return;  // expert received no tokens; the block retires immediately
  }

  const int lane = threadIdx.x % kWarpSize;
  const int warp = threadIdx.x / kWarpSize;
  const __nv_bfloat16* expert_weights = weights + static_cast<int64_t>(expert) * n_dim * k_dim;

  // One warp owns one output column n: the column's K weights are contiguous
  // (the packed layout is [E, N, K]), so the 32 lanes read the column as
  // coalesced 16-byte chunks. A grid-stride loop covers all N columns.
  for (int64_t column = static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp;
       column < n_dim;
       column += static_cast<int64_t>(gridDim.x) * kWarpsPerBlock) {
    const __nv_bfloat16* column_weights = expert_weights + column * k_dim;
    for (int64_t row_base = start; row_base < end; row_base += kRowChunk) {
      const int chunk = static_cast<int>(min(static_cast<int64_t>(kRowChunk), end - row_base));
      float accumulator[kRowChunk] = {};
      // Lanes stride across K in 8-element vectors. The same weight vector is
      // multiplied against every row in the chunk, which is what makes the
      // weight traffic (the bandwidth bottleneck) independent of M.
      for (int64_t k = static_cast<int64_t>(lane) * kElementsPerLoad;
           k < k_dim;
           k += static_cast<int64_t>(kWarpSize) * kElementsPerLoad) {
        const uint4 weight_vector = *reinterpret_cast<const uint4*>(column_weights + k);
        const __nv_bfloat16* weight8 = reinterpret_cast<const __nv_bfloat16*>(&weight_vector);
        for (int m = 0; m < chunk; ++m) {
          const uint4 activation_vector =
              *reinterpret_cast<const uint4*>(activations + (row_base + m) * k_dim + k);
          const __nv_bfloat16* activation8 =
              reinterpret_cast<const __nv_bfloat16*>(&activation_vector);
#pragma unroll
          for (int j = 0; j < kElementsPerLoad; ++j) {
            accumulator[m] += __bfloat162float(activation8[j]) * __bfloat162float(weight8[j]);
          }
        }
      }
      // Each lane holds a partial dot product; a shuffle tree folds the 32
      // partials into lane 0, which writes the bf16 result.
      for (int m = 0; m < chunk; ++m) {
        float value = accumulator[m];
        for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
          value += __shfl_down_sync(0xffffffffU, value, offset);
        }
        if (lane == 0) {
          output[(row_base + m) * n_dim + column] = __float2bfloat16(value);
        }
      }
    }
  }
}

}  // namespace

void fused_grouped_gemm_bf16_out_cuda(
    torch::Tensor activations,
    torch::Tensor packed_weights,
    torch::Tensor expert_offsets,
    torch::Tensor output) {
  TORCH_CHECK(activations.is_cuda() && packed_weights.is_cuda() && expert_offsets.is_cuda() && output.is_cuda(), "fused grouped GEMM tensors must be CUDA tensors");
  TORCH_CHECK(activations.device() == packed_weights.device() && activations.device() == expert_offsets.device() && activations.device() == output.device(), "fused grouped GEMM tensors must share a CUDA device");
  TORCH_CHECK(activations.scalar_type() == at::kBFloat16 && packed_weights.scalar_type() == at::kBFloat16 && output.scalar_type() == at::kBFloat16, "fused grouped GEMM requires bfloat16 activations, weights, and output");
  TORCH_CHECK(expert_offsets.scalar_type() == at::kLong, "fused grouped GEMM offsets must be the int64 [E + 1] routing prefix sum");
  TORCH_CHECK(activations.dim() == 2, "activations must have shape [A, K]");
  TORCH_CHECK(packed_weights.dim() == 3, "packed_weights must have shape [E, K, N]");
  TORCH_CHECK(
      expert_offsets.dim() == 1 && expert_offsets.numel() == packed_weights.size(0) + 1,
      "expert_offsets must contain num_experts + 1 cumulative offsets");
  TORCH_CHECK(activations.size(1) == packed_weights.size(1), "fused grouped GEMM K dimension mismatch");
  TORCH_CHECK(
      activations.size(1) % kElementsPerLoad == 0,
      "fused grouped GEMM K must be divisible by 8 for 16-byte loads");
  TORCH_CHECK(output.sizes() == at::IntArrayRef({activations.size(0), packed_weights.size(2)}), "fused grouped GEMM output shape mismatch");
  TORCH_CHECK(activations.is_contiguous() && expert_offsets.is_contiguous() && output.is_contiguous(), "fused grouped GEMM activations, offsets, and output must be contiguous");
  // The kernel streams each output column's K weights as one contiguous run,
  // which is exactly the [E, N, K]-contiguous storage behind the public
  // weight.transpose(1, 2) view. Enforce that layout instead of guessing.
  TORCH_CHECK(
      packed_weights.stride(1) == 1 &&
          packed_weights.stride(2) == packed_weights.size(1) &&
          packed_weights.stride(0) == packed_weights.size(1) * packed_weights.size(2),
      "packed_weights must be the [E, K, N] transpose view of contiguous [E, N, K] weights");

  const c10::cuda::CUDAGuard guard(activations.device());
  const cudaStream_t stream =
      c10::cuda::getCurrentCUDAStream(activations.get_device()).stream();

  const int64_t k_dim = packed_weights.size(1);
  const int64_t n_dim = packed_weights.size(2);
  const int64_t column_tiles = (n_dim + kWarpsPerBlock - 1) / kWarpsPerBlock;
  const dim3 grid(
      static_cast<unsigned int>(std::min<int64_t>(column_tiles, kMaxGridX)),
      static_cast<unsigned int>(packed_weights.size(0)));

  fused_grouped_gemm_bf16_kernel<<<grid, kThreads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(activations.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(packed_weights.data_ptr<at::BFloat16>()),
      expert_offsets.data_ptr<int64_t>(),
      reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
      k_dim,
      n_dim);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
