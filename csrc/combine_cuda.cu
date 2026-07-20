#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>

namespace {

constexpr int kThreads = 256;
constexpr int kMaxBlocks = 65535;

int launch_blocks(int64_t work_items) {
  return static_cast<int>(
      std::min<int64_t>((work_items + kThreads - 1) / kThreads, kMaxBlocks));
}

template <typename scalar_t>
__global__ void combine_kernel(
    const scalar_t* __restrict__ expert_outputs,
    const int64_t* __restrict__ assignment_to_permuted,
    const scalar_t* __restrict__ routing_weights,
    scalar_t* __restrict__ output,
    int64_t num_tokens,
    int64_t hidden_size,
    int64_t top_k,
    int64_t num_assignments) {
  const int64_t work_items = num_tokens * hidden_size;
  for (int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
       linear < work_items;
       linear += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int64_t token = linear / hidden_size;
    const int64_t hidden = linear % hidden_size;
    float accumulator = 0.0F;
#pragma unroll
    for (int64_t slot = 0; slot < top_k; ++slot) {
      const int64_t assignment = token * top_k + slot;
      const int64_t position = assignment_to_permuted[assignment];
      if (position >= 0 && position < num_assignments) {
        accumulator += static_cast<float>(
                           expert_outputs[position * hidden_size + hidden]) *
            static_cast<float>(routing_weights[assignment]);
      }
    }
    output[linear] = static_cast<scalar_t>(accumulator);
  }
}

void validate_combine(
    const at::Tensor& expert_outputs,
    const at::Tensor& assignment_to_permuted,
    const at::Tensor& routing_weights) {
  TORCH_CHECK(expert_outputs.is_cuda(), "expert_outputs must be a CUDA tensor");
  TORCH_CHECK(
      assignment_to_permuted.is_cuda(),
      "assignment_to_permuted must be a CUDA tensor");
  TORCH_CHECK(routing_weights.is_cuda(), "routing_weights must be a CUDA tensor");
  TORCH_CHECK(expert_outputs.dim() == 2, "expert_outputs must have shape [A, H]");
  TORCH_CHECK(
      assignment_to_permuted.dim() == 1,
      "assignment_to_permuted must have shape [A]");
  TORCH_CHECK(routing_weights.dim() == 2, "routing_weights must have shape [T, K]");
  TORCH_CHECK(routing_weights.size(0) > 0, "num_tokens must be positive");
  TORCH_CHECK(expert_outputs.size(1) > 0, "hidden_size must be positive");
  const int64_t top_k = routing_weights.size(1);
  TORCH_CHECK(
      top_k == 1 || top_k == 2 || top_k == 4 || top_k == 8,
      "top_k must be 1, 2, 4, or 8");
  const int64_t num_assignments = routing_weights.size(0) * top_k;
  TORCH_CHECK(
      expert_outputs.size(0) == num_assignments,
      "expert_outputs first dimension must equal num_tokens * top_k");
  TORCH_CHECK(
      assignment_to_permuted.numel() == num_assignments,
      "assignment_to_permuted length must equal num_tokens * top_k");
  TORCH_CHECK(
      expert_outputs.scalar_type() == at::kFloat ||
          expert_outputs.scalar_type() == at::kHalf ||
          expert_outputs.scalar_type() == at::kBFloat16,
      "expert_outputs dtype must be float32, float16, or bfloat16");
  TORCH_CHECK(
      routing_weights.scalar_type() == expert_outputs.scalar_type(),
      "routing_weights and expert_outputs must have the same dtype");
  TORCH_CHECK(
      assignment_to_permuted.scalar_type() == at::kLong,
      "assignment_to_permuted must have dtype int64");
  TORCH_CHECK(
      expert_outputs.device() == assignment_to_permuted.device() &&
          expert_outputs.device() == routing_weights.device(),
      "all combine inputs must be on the same CUDA device");
  TORCH_CHECK(
      expert_outputs.is_contiguous() && assignment_to_permuted.is_contiguous() &&
          routing_weights.is_contiguous(),
      "all combine inputs must be contiguous");
  TORCH_CHECK(
      !expert_outputs.requires_grad() && !routing_weights.requires_grad(),
      "combine V0 does not support backward");
}

}  // namespace

at::Tensor combine_cuda(
    at::Tensor expert_outputs,
    at::Tensor assignment_to_permuted,
    at::Tensor routing_weights) {
  validate_combine(expert_outputs, assignment_to_permuted, routing_weights);
  const c10::cuda::CUDAGuard device_guard(expert_outputs.device());
  const int64_t num_tokens = routing_weights.size(0);
  const int64_t top_k = routing_weights.size(1);
  const int64_t num_assignments = num_tokens * top_k;
  const int64_t hidden_size = expert_outputs.size(1);

  at::Tensor output = at::empty(
      {num_tokens, hidden_size}, expert_outputs.options());
  const cudaStream_t stream =
      c10::cuda::getCurrentCUDAStream(expert_outputs.get_device()).stream();
  const int64_t work_items = num_tokens * hidden_size;
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      expert_outputs.scalar_type(),
      "moe_combine",
      [&] {
        combine_kernel<scalar_t><<<launch_blocks(work_items), kThreads, 0, stream>>>(
            expert_outputs.data_ptr<scalar_t>(),
            assignment_to_permuted.data_ptr<int64_t>(),
            routing_weights.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            num_tokens,
            hidden_size,
            top_k,
            num_assignments);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
