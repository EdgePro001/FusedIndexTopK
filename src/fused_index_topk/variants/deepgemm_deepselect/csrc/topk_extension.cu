// Build-only SM90 adapter for the frozen DeepSelect v1.0.0 FP32 kernel.
// The algorithm and kernel body are included verbatim from the upstream checkout.

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <cstdint>
#include <limits>

#include "cuda_kernels/config.h"
#include "cuda_kernels/v3_fp32/topk_select.cuh"
#include "structs.h"

namespace {

using Fp32K2048Config = TopkSelectConfig<
    float,
    int32_t,
    false,  // sorted_value
    false,  // sorted_index
    false,  // return_value
    4096,   // max_topk: upstream dispatch tier for topk in (1024, 4096]
    256,    // num_threads
    1,      // target_occupancy
    4096,   // elements_per_round
    4096,   // reconstruct_threshold
    3,      // tma_buffer_depth
    512,    // elements_per_segment
    1       // cluster_size
>;

void topk_out(torch::Tensor input, torch::Tensor end, torch::Tensor output_index) {
  TORCH_CHECK(input.is_cuda() && end.is_cuda() && output_index.is_cuda(),
              "input, end, and output_index must be CUDA tensors");
  TORCH_CHECK(input.device() == end.device() && input.device() == output_index.device(),
              "all tensors must be on the same CUDA device");
  TORCH_CHECK(input.scalar_type() == at::kFloat, "input must be float32");
  TORCH_CHECK(end.scalar_type() == at::kInt, "end must be int32");
  TORCH_CHECK(output_index.scalar_type() == at::kInt, "output_index must be int32");
  TORCH_CHECK(input.dim() == 2 && output_index.dim() == 2 && end.dim() == 1,
              "expected input [Q,N], end [Q], and output_index [Q,K]");
  TORCH_CHECK(input.size(0) == output_index.size(0) && input.size(0) == end.size(0),
              "row counts must match");
  TORCH_CHECK(input.stride(1) == 1 && output_index.stride(1) == 1,
              "last dimensions must be contiguous");
  TORCH_CHECK(end.is_contiguous(), "end must be contiguous");
  TORCH_CHECK(input.storage_offset() == 0, "input must have zero storage offset");
  TORCH_CHECK(output_index.storage_offset() == 0,
              "output_index must have zero storage offset");

  const auto batch_size = input.size(0);
  const auto vocab_size = input.size(1);
  const auto topk = output_index.size(1);
  TORCH_CHECK(batch_size > 0 && batch_size <= std::numeric_limits<uint32_t>::max(),
              "batch size is out of range");
  TORCH_CHECK(vocab_size > topk && vocab_size < MAX_VOCAB_SIZE,
              "DeepSelect requires K < N < 2^23");
  TORCH_CHECK(topk > 1024 && topk <= 4096,
              "this frozen adapter covers the upstream K in (1024, 4096] tier");
  TORCH_CHECK(input.stride(0) * input.element_size() %
                      INPUT_STRIDE_ALIGNMENT_REQUIREMENT ==
                  0,
              "input row stride must be 1024-byte aligned");
  TORCH_CHECK(output_index.stride(0) * output_index.element_size() %
                      OUTPUT_STRIDE_ALIGNMENT_REQUIREMENT ==
                  0,
              "output row stride must be 32-byte aligned");

  cudaDeviceProp* device_prop = at::cuda::getDeviceProperties(at::cuda::current_device());
  TORCH_CHECK(device_prop != nullptr, "unable to query CUDA device properties");
  TORCH_CHECK(device_prop->major == 9 && device_prop->minor == 0,
              "this build adapter is qualified only for SM90");

  TopkSelectArgs args = {
      static_cast<uint32_t>(batch_size),
      static_cast<uint32_t>(vocab_size),
      static_cast<uint32_t>(topk),
      input.data_ptr(),
      nullptr,
      output_index.data_ptr(),
      nullptr,
      end.data_ptr<int>(),
      nullptr,
      static_cast<uint64_t>(input.stride(0)),
      0,
      static_cast<uint64_t>(output_index.stride(0)),
      false,
      false,
      false,
      -1,
      -std::numeric_limits<float>::infinity(),
      true,
      device_prop->sharedMemPerBlockOptin,
      at::cuda::getCurrentCUDAStream().stream(),
  };
  topk_select_fp32::run_topk_select_kernel<Fp32K2048Config>(args);
}
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("topk_out", &topk_out, "DeepSelect FP32 exact Top-K (SM90 build adapter)");
}
