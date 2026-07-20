#include "moe_fastpath.h"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>

namespace {

constexpr int kThreads = 256;
constexpr int kMaxBlocks = 65535;

int launch_blocks(int64_t items) {
  return static_cast<int>(
      std::min<int64_t>((items + kThreads - 1) / kThreads, kMaxBlocks));
}

template <typename scalar_t>
__global__ void weighted_combine_kernel(
    const scalar_t* __restrict__ expert_outputs,
    const int64_t* __restrict__ assignment_to_permuted,
    const scalar_t* __restrict__ weights,
    scalar_t* __restrict__ output,
    int64_t tokens,
    int64_t hidden,
    int64_t top_k,
    int64_t assignments) {
  const int64_t items = tokens * hidden;
  for (int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
       linear < items;
       linear += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int64_t token = linear / hidden;
    const int64_t column = linear % hidden;
    float accumulator = 0.0F;
    for (int64_t rank = 0; rank < top_k; ++rank) {
      const int64_t assignment = token * top_k + rank;
      const int64_t row = assignment_to_permuted[assignment];
      if (row >= 0 && row < assignments) {
        accumulator += static_cast<float>(expert_outputs[row * hidden + column]) *
            static_cast<float>(weights[assignment]);
      }
    }
    output[linear] = static_cast<scalar_t>(accumulator);
  }
}

}  // namespace

void combine_out_cuda(
    torch::Tensor expert_outputs,
    torch::Tensor assignment_to_permuted,
    torch::Tensor routing_weights,
    torch::Tensor output) {
  TORCH_CHECK(expert_outputs.is_cuda() && assignment_to_permuted.is_cuda() && routing_weights.is_cuda() && output.is_cuda(), "combine tensors must be CUDA tensors");
  TORCH_CHECK(expert_outputs.device() == assignment_to_permuted.device() && expert_outputs.device() == routing_weights.device() && expert_outputs.device() == output.device(), "combine tensors must share a CUDA device");
  TORCH_CHECK(expert_outputs.dim() == 2 && routing_weights.dim() == 2 && output.dim() == 2, "invalid combine rank");
  const int64_t tokens = routing_weights.size(0);
  const int64_t top_k = routing_weights.size(1);
  const int64_t assignments = tokens * top_k;
  TORCH_CHECK(top_k == 1 || top_k == 2 || top_k == 4 || top_k == 8, "top_k must be 1, 2, 4, or 8");
  TORCH_CHECK(expert_outputs.size(0) == assignments && assignment_to_permuted.numel() == assignments, "combine assignment count mismatch");
  TORCH_CHECK(output.sizes() == at::IntArrayRef({tokens, expert_outputs.size(1)}), "invalid combine output shape");
  TORCH_CHECK(expert_outputs.scalar_type() == routing_weights.scalar_type() && expert_outputs.scalar_type() == output.scalar_type(), "combine dtype mismatch");
  TORCH_CHECK(assignment_to_permuted.scalar_type() == at::kLong, "combine mapping must be int64");
  TORCH_CHECK(expert_outputs.is_contiguous() && assignment_to_permuted.is_contiguous() && routing_weights.is_contiguous() && output.is_contiguous(), "combine tensors must be contiguous");

  const c10::cuda::CUDAGuard guard(expert_outputs.device());
  const cudaStream_t stream =
      c10::cuda::getCurrentCUDAStream(expert_outputs.get_device()).stream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      expert_outputs.scalar_type(),
      "moe_weighted_combine",
      [&] {
        weighted_combine_kernel<scalar_t><<<launch_blocks(output.numel()), kThreads, 0, stream>>>(
            expert_outputs.data_ptr<scalar_t>(),
            assignment_to_permuted.data_ptr<int64_t>(),
            routing_weights.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            tokens,
            expert_outputs.size(1),
            top_k,
            assignments);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor combine_cuda(
    torch::Tensor expert_outputs,
    torch::Tensor assignment_to_permuted,
    torch::Tensor routing_weights) {
  auto output = at::empty(
      {routing_weights.size(0), expert_outputs.size(1)}, expert_outputs.options());
  combine_out_cuda(expert_outputs, assignment_to_permuted, routing_weights, output);
  return output;
}
