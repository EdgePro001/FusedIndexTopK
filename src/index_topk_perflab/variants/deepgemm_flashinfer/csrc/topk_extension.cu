#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <flashinfer/math.cuh>
#include <flashinfer/topk.cuh>

#include <cstdint>
#include <string>

namespace {

void check_tensor(const torch::Tensor& tensor,
                  at::ScalarType dtype,
                  int64_t rows,
                  int64_t columns,
                  const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.scalar_type() == dtype, name, " has an unexpected dtype");
  TORCH_CHECK(tensor.dim() == 2, name, " must be two dimensional");
  TORCH_CHECK(tensor.size(0) == rows && tensor.size(1) == columns,
              name,
              " has an unexpected shape");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void topk_out(torch::Tensor input,
              torch::Tensor output_indices,
              torch::Tensor output_values,
              torch::Tensor row_states,
              int64_t algorithm) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(input.scalar_type() == at::kFloat, "input must be float32");
  TORCH_CHECK(input.dim() == 2 && input.is_contiguous(),
              "input must be a contiguous two-dimensional physical-row view");
  TORCH_CHECK(input.size(0) > 0 && input.size(1) > 0, "input must not be empty");

  const int64_t rows = input.size(0);
  const int64_t columns = input.size(1);
  const int64_t top_k = output_indices.size(1);
  TORCH_CHECK(rows <= UINT32_MAX && columns <= UINT32_MAX && top_k <= UINT32_MAX,
              "FlashInfer dimensions must fit uint32_t");
  TORCH_CHECK(top_k > 0 && top_k <= columns, "invalid top_k");
  check_tensor(output_indices, at::kInt, rows, top_k, "output_indices");
  check_tensor(output_values, at::kFloat, rows, top_k, "output_values");
  TORCH_CHECK(row_states.is_cuda() && row_states.scalar_type() == at::kByte &&
                  row_states.is_contiguous() && row_states.numel() >= 1024 * 1024,
              "row_states must be a contiguous CUDA uint8 tensor with at least 1 MiB");
  TORCH_CHECK(input.get_device() == output_indices.get_device() &&
                  input.get_device() == output_values.get_device() &&
                  input.get_device() == row_states.get_device(),
              "all tensors must be on the same CUDA device");

  c10::cuda::CUDAGuard guard(input.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(input.get_device()).stream();
  auto* states = reinterpret_cast<flashinfer::sampling::RadixRowState*>(
      row_states.data_ptr<uint8_t>());

  cudaError_t status = cudaSuccess;
  switch (algorithm) {
    case 0:
      status = flashinfer::sampling::TopKDispatch<float, int32_t>(
          input.data_ptr<float>(),
          output_indices.data_ptr<int32_t>(),
          output_values.data_ptr<float>(),
          static_cast<uint32_t>(rows),
          static_cast<uint32_t>(top_k),
          static_cast<uint32_t>(columns),
          states,
          false,
          false,
          flashinfer::sampling::TopKTieBreak::None,
          stream,
          false);
      break;
    case 1:
      TORCH_CHECK(top_k <= flashinfer::sampling::FILTERED_TOPK_MAX_K,
                  "FilteredTopK supports at most FILTERED_TOPK_MAX_K elements");
      TORCH_CHECK(flashinfer::sampling::CanImplementFilteredTopK(),
                  "this GPU cannot provide FilteredTopK's shared memory requirement");
      status = flashinfer::sampling::FilteredTopK<float, int32_t>(
          input.data_ptr<float>(),
          output_indices.data_ptr<int32_t>(),
          output_values.data_ptr<float>(),
          nullptr,
          static_cast<uint32_t>(rows),
          static_cast<uint32_t>(top_k),
          static_cast<uint32_t>(columns),
          false,
          flashinfer::sampling::TopKTieBreak::None,
          stream,
          false);
      break;
    case 2:
      status = flashinfer::sampling::RadixTopKMultiCTA<float, int32_t>(
          input.data_ptr<float>(),
          output_indices.data_ptr<int32_t>(),
          output_values.data_ptr<float>(),
          nullptr,
          static_cast<uint32_t>(rows),
          static_cast<uint32_t>(top_k),
          static_cast<uint32_t>(columns),
          states,
          false,
          stream);
      break;
    default:
      TORCH_CHECK(false, "unknown FlashInfer TopK algorithm id: ", algorithm);
  }
  TORCH_CHECK(status == cudaSuccess,
              "FlashInfer TopK launch failed: ",
              cudaGetErrorString(status));
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("topk_out", &topk_out, "FlashInfer exact TopK into preallocated tensors");
}
