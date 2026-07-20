#include "moe_fastpath.h"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace {

constexpr int kThreads = 256;
constexpr int kMaxBlocks = 65535;

int launch_blocks(int64_t items) {
  return static_cast<int>(
      std::min<int64_t>((items + kThreads - 1) / kThreads, kMaxBlocks));
}

template <typename scalar_t>
__global__ void gather_kernel(
    const scalar_t* __restrict__ hidden,
    const int64_t* __restrict__ permuted_to_assignment,
    scalar_t* __restrict__ output,
    int64_t assignments,
    int64_t tokens,
    int64_t hidden_size,
    int64_t top_k) {
  const int64_t items = assignments * hidden_size;
  for (int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
       linear < items;
       linear += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int64_t row = linear / hidden_size;
    const int64_t column = linear % hidden_size;
    const int64_t assignment = permuted_to_assignment[row];
    if (assignment < 0 || assignment >= assignments) {
      output[linear] = static_cast<scalar_t>(0);
      continue;
    }
    const int64_t token = assignment / top_k;
    output[linear] = token < tokens
        ? hidden[token * hidden_size + column]
        : static_cast<scalar_t>(0);
  }
}

void validate_dispatch(
    const torch::Tensor& hidden,
    const torch::Tensor& expert_ids,
    int64_t num_experts) {
  TORCH_CHECK(hidden.is_cuda() && expert_ids.is_cuda(), "dispatch inputs must be CUDA tensors");
  TORCH_CHECK(hidden.device() == expert_ids.device(), "dispatch inputs must share a device");
  TORCH_CHECK(hidden.dim() == 2 && expert_ids.dim() == 2, "dispatch expects [T,H] and [T,K]");
  TORCH_CHECK(hidden.size(0) == expert_ids.size(0), "dispatch tensors must share num_tokens");
  TORCH_CHECK(hidden.size(0) > 0 && hidden.size(1) > 0, "dispatch dimensions must be positive");
  TORCH_CHECK(hidden.is_contiguous() && expert_ids.is_contiguous(), "dispatch inputs must be contiguous");
  TORCH_CHECK(expert_ids.scalar_type() == at::kLong, "expert_ids must be int64");
  TORCH_CHECK(
      hidden.scalar_type() == at::kFloat || hidden.scalar_type() == at::kHalf ||
          hidden.scalar_type() == at::kBFloat16,
      "hidden_states must be float32, float16, or bfloat16");
  TORCH_CHECK(num_experts > 0 && num_experts <= 256, "num_experts must be in [1, 256]");
}

}  // namespace

void permute_out_cuda(
    torch::Tensor hidden_states,
    torch::Tensor permuted_to_assignment,
    int64_t top_k,
    torch::Tensor output) {
  TORCH_CHECK(hidden_states.is_cuda() && output.is_cuda(), "permutation tensors must be CUDA tensors");
  TORCH_CHECK(hidden_states.device() == output.device(), "permutation tensors must share a device");
  TORCH_CHECK(permuted_to_assignment.device() == output.device(), "mapping must share the CUDA device");
  TORCH_CHECK(permuted_to_assignment.scalar_type() == at::kLong, "mapping must be int64");
  TORCH_CHECK(hidden_states.dim() == 2, "hidden_states must have shape [T, H]");
  TORCH_CHECK(hidden_states.size(0) > 0 && hidden_states.size(1) > 0, "hidden dimensions must be positive");
  TORCH_CHECK(top_k == 1 || top_k == 2 || top_k == 4 || top_k == 8, "top_k must be 1, 2, 4, or 8");
  TORCH_CHECK(hidden_states.scalar_type() == output.scalar_type(), "permutation dtype mismatch");
  TORCH_CHECK(hidden_states.is_contiguous() && permuted_to_assignment.is_contiguous() && output.is_contiguous(), "permutation tensors must be contiguous");
  TORCH_CHECK(output.dim() == 2 && output.size(0) == permuted_to_assignment.numel(), "invalid permutation output shape");
  TORCH_CHECK(output.size(1) == hidden_states.size(1), "invalid permutation hidden size");
  TORCH_CHECK(output.size(0) == hidden_states.size(0) * top_k, "permutation rows must equal T * top_k");

  const c10::cuda::CUDAGuard guard(hidden_states.device());
  const int64_t items = output.numel();
  const cudaStream_t stream =
      c10::cuda::getCurrentCUDAStream(hidden_states.get_device()).stream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      hidden_states.scalar_type(),
      "moe_stable_permute",
      [&] {
        gather_kernel<scalar_t><<<launch_blocks(items), kThreads, 0, stream>>>(
            hidden_states.data_ptr<scalar_t>(),
            permuted_to_assignment.data_ptr<int64_t>(),
            output.data_ptr<scalar_t>(),
            output.size(0),
            hidden_states.size(0),
            output.size(1),
            top_k);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::vector<torch::Tensor> dispatch_cuda(
    torch::Tensor hidden_states,
    torch::Tensor expert_ids,
    int64_t num_experts) {
  validate_dispatch(hidden_states, expert_ids, num_experts);
  auto metadata = routing_metadata_cuda(expert_ids, num_experts);
  auto output = at::empty(
      {expert_ids.numel(), hidden_states.size(1)}, hidden_states.options());
  permute_out_cuda(hidden_states, metadata[3], expert_ids.size(1), output);
  return {output, metadata[0], metadata[1], metadata[2], metadata[3]};
}
