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
__global__ void swiglu_kernel(
    const scalar_t* __restrict__ gate_up,
    scalar_t* __restrict__ output,
    int64_t rows,
    int64_t intermediate) {
  const int64_t items = rows * intermediate;
  for (int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
       linear < items;
       linear += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int64_t row = linear / intermediate;
    const int64_t column = linear % intermediate;
    const float gate = static_cast<float>(
        gate_up[row * (2 * intermediate) + column]);
    const float up = static_cast<float>(
        gate_up[row * (2 * intermediate) + intermediate + column]);
    const float silu = gate / (1.0F + expf(-gate));
    output[linear] = static_cast<scalar_t>(silu * up);
  }
}

}  // namespace

void swiglu_out_cuda(torch::Tensor gate_up, torch::Tensor output) {
  TORCH_CHECK(gate_up.is_cuda() && output.is_cuda(), "SwiGLU tensors must be CUDA tensors");
  TORCH_CHECK(gate_up.device() == output.device(), "SwiGLU tensors must share a CUDA device");
  TORCH_CHECK(gate_up.dim() == 2 && output.dim() == 2, "SwiGLU tensors must be two-dimensional");
  TORCH_CHECK(gate_up.size(1) == output.size(1) * 2 && gate_up.size(0) == output.size(0), "SwiGLU output must have shape [A, gate_up_width/2]");
  TORCH_CHECK(gate_up.scalar_type() == output.scalar_type(), "SwiGLU dtype mismatch");
  TORCH_CHECK(gate_up.is_contiguous() && output.is_contiguous(), "SwiGLU tensors must be contiguous");

  const c10::cuda::CUDAGuard guard(gate_up.device());
  const cudaStream_t stream =
      c10::cuda::getCurrentCUDAStream(gate_up.get_device()).stream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      gate_up.scalar_type(),
      "moe_swiglu",
      [&] {
        swiglu_kernel<scalar_t><<<launch_blocks(output.numel()), kThreads, 0, stream>>>(
            gate_up.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            output.size(0),
            output.size(1));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
