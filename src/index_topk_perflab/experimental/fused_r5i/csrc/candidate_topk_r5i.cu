#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

constexpr int kBlockThreads = 1024;
constexpr int kWarpSize = 32;
constexpr int kTopK = 2048;
constexpr int kCandidateCapacity = 14080;
constexpr int kCandidateWords = 2 * kCandidateCapacity;
constexpr int kCandidateBytes = kCandidateWords * sizeof(uint32_t);
constexpr unsigned kFullMask = 0xffffffffu;

static_assert(kCandidateBytes == 112640);

__device__ __forceinline__ uint32_t ordered_float(float value) {
  const uint32_t bits = __float_as_uint(value);
  const uint32_t mask =
      (static_cast<int32_t>(bits) < 0) ? 0xffffffffu : 0x80000000u;
  return bits ^ mask;
}

__device__ __forceinline__ void warp_histogram_add(int* histogram,
                                                    bool active,
                                                    int bin) {
  const unsigned active_mask = __ballot_sync(kFullMask, active);
  if (!active) return;
  const unsigned peers = __match_any_sync(active_mask, bin);
  const int lane = threadIdx.x & (kWarpSize - 1);
  if (lane == __ffs(static_cast<int>(peers)) - 1) {
    atomicAdd(&histogram[bin], __popc(peers));
  }
}

__device__ __forceinline__ int select_histogram_byte(const int* histogram,
                                                      int rank,
                                                      int* selected_byte,
                                                      int* greater_count) {
  const int lane = threadIdx.x & (kWarpSize - 1);
  int counts[8];
  int lane_total = 0;
  #pragma unroll
  for (int item = 0; item < 8; ++item) {
    counts[item] = histogram[lane * 8 + item];
    lane_total += counts[item];
  }
  int suffix_total = lane_total;
  #pragma unroll
  for (int offset = 1; offset < kWarpSize; offset <<= 1) {
    const int incoming = __shfl_down_sync(kFullMask, suffix_total, offset);
    if (lane + offset < kWarpSize) suffix_total += incoming;
  }
  int greater = suffix_total - lane_total;
  int found = 0;
  #pragma unroll
  for (int item = 7; item >= 0; --item) {
    const int next = greater + counts[item];
    if (greater < rank && rank <= next) {
      *selected_byte = lane * 8 + item;
      *greater_count = greater;
      found = 1;
    }
    greater = next;
  }
  return found;
}

__device__ __forceinline__ uint32_t radix_select_shared(
    const uint32_t* scores,
    int count,
    int requested_rank,
    int* histogram,
    int* selected_byte,
    int* selected_greater,
    int* remaining_rank) {
  const int tx = threadIdx.x;
  uint32_t threshold = 0;
  uint32_t prefix_mask = 0;
  int rank = requested_rank;
  #pragma unroll
  for (int round = 0; round < 4; ++round) {
    const int shift = 24 - round * 8;
    if (tx < 256) histogram[tx] = 0;
    __syncthreads();
    for (int base = 0; base < count; base += kBlockThreads) {
      const int index = base + tx;
      const uint32_t score = index < count ? scores[index] : 0;
      const bool active =
          index < count && (score & prefix_mask) == threshold;
      warp_histogram_add(
          histogram, active, static_cast<int>((score >> shift) & 0xffu));
    }
    __syncthreads();
    if (tx < kWarpSize) {
      int byte = 0;
      int greater = 0;
      if (select_histogram_byte(histogram, rank, &byte, &greater)) {
        *selected_byte = byte;
        *selected_greater = greater;
      }
    }
    __syncthreads();
    rank -= *selected_greater;
    threshold |= static_cast<uint32_t>(*selected_byte) << shift;
    prefix_mask |= 0xffu << shift;
  }
  *remaining_rank = rank;
  return threshold;
}

__device__ __forceinline__ int warp_reserve_output(int* counter,
                                                    bool active) {
  const unsigned active_mask = __ballot_sync(kFullMask, active);
  if (active_mask == 0) return -1;
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int leader = __ffs(static_cast<int>(active_mask)) - 1;
  int base = 0;
  if (lane == leader) base = atomicAdd(counter, __popc(active_mask));
  base = __shfl_sync(kFullMask, base, leader);
  const unsigned lower_lanes = lane == 0 ? 0u : ((1u << lane) - 1u);
  return active ? base + __popc(active_mask & lower_lanes) : -1;
}

__global__ __launch_bounds__(kBlockThreads, 2)
void itk_fused_r5i_candidate_radix(
    const float* __restrict__ candidate_scores,
    const int32_t* __restrict__ candidate_indices,
    const int32_t* __restrict__ candidate_counts,
    int32_t* __restrict__ output_indices,
    uint8_t* __restrict__ failure_flags,
    int rows) {
  const int row = blockIdx.x;
  if (row >= rows) return;
  const int tx = threadIdx.x;
  const int count = candidate_counts[row];
  const bool verified = count >= kTopK && count <= kCandidateCapacity;
  if (tx == 0) failure_flags[row] = verified ? uint8_t{0} : uint8_t{1};
  if (!verified) return;

  extern __shared__ __align__(128) unsigned char dynamic_shared[];
  uint32_t* scores = reinterpret_cast<uint32_t*>(dynamic_shared);
  uint32_t* indices = scores + kCandidateCapacity;
  const int64_t candidate_row =
      static_cast<int64_t>(row) * kCandidateCapacity;
  for (int index = tx; index < count; index += kBlockThreads) {
    scores[index] = ordered_float(candidate_scores[candidate_row + index]);
    indices[index] =
        static_cast<uint32_t>(candidate_indices[candidate_row + index]);
  }

  __shared__ __align__(128) int histogram[256];
  __shared__ int selected_byte;
  __shared__ int selected_greater;
  __shared__ int greater_counter;
  __shared__ int equal_counter;
  __syncthreads();

  int remaining_rank = 0;
  const uint32_t threshold = radix_select_shared(
      scores,
      count,
      kTopK,
      histogram,
      &selected_byte,
      &selected_greater,
      &remaining_rank);

  if (tx == 0) {
    greater_counter = 0;
    equal_counter = 0;
  }
  __syncthreads();

  const int total_greater = kTopK - remaining_rank;
  int32_t* output_row =
      output_indices + static_cast<int64_t>(row) * kTopK;
  for (int base = 0; base < count; base += kBlockThreads) {
    const int index = base + tx;
    const bool in_bounds = index < count;
    const uint32_t score = in_bounds ? scores[index] : 0;
    const bool is_greater = in_bounds && score > threshold;
    const bool is_equal = in_bounds && score == threshold;
    const int greater_position =
        warp_reserve_output(&greater_counter, is_greater);
    const int equal_position =
        warp_reserve_output(&equal_counter, is_equal);

    int output_position = -1;
    if (is_greater) {
      output_position = greater_position;
    } else if (is_equal && equal_position < remaining_rank) {
      output_position = total_greater + equal_position;
    }
    if (output_position >= 0) {
      output_row[output_position] = static_cast<int32_t>(indices[index]);
    }
  }
}

void check_matrix(const torch::Tensor& tensor,
                  at::ScalarType dtype,
                  int64_t rows,
                  int64_t columns,
                  const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(tensor.scalar_type() == dtype, name, " has unexpected dtype");
  TORCH_CHECK(tensor.dim() == 2 && tensor.size(0) == rows &&
                  tensor.size(1) == columns && tensor.is_contiguous(),
              name,
              " has unexpected shape or layout");
}

void configure() {
  const cudaError_t status = cudaFuncSetAttribute(
      itk_fused_r5i_candidate_radix,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      kCandidateBytes);
  TORCH_CHECK(status == cudaSuccess,
              "cannot opt in to fused R5i candidate shared memory: ",
              cudaGetErrorString(status));
}

pybind11::dict resource_report() {
  cudaFuncAttributes attributes{};
  cudaError_t status = cudaFuncGetAttributes(
      &attributes, itk_fused_r5i_candidate_radix);
  TORCH_CHECK(status == cudaSuccess,
              "cannot query fused R5i candidate kernel: ",
              cudaGetErrorString(status));
  int active_blocks_per_sm = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks_per_sm,
      itk_fused_r5i_candidate_radix,
      kBlockThreads,
      kCandidateBytes);
  TORCH_CHECK(status == cudaSuccess,
              "cannot calculate fused R5i candidate occupancy: ",
              cudaGetErrorString(status));

  pybind11::dict report;
  report["candidate_capacity"] = kCandidateCapacity;
  report["dynamic_shared_bytes"] = kCandidateBytes;
  report["static_shared_bytes"] = attributes.sharedSizeBytes;
  report["registers_per_thread"] = attributes.numRegs;
  report["active_blocks_per_sm"] = active_blocks_per_sm;
  return report;
}

void topk_out(torch::Tensor candidate_scores,
              torch::Tensor candidate_indices,
              torch::Tensor candidate_counts,
              torch::Tensor output_indices,
              torch::Tensor failure_flags) {
  TORCH_CHECK(candidate_scores.dim() == 2,
              "candidate_scores must be two dimensional");
  const int64_t rows = candidate_scores.size(0);
  TORCH_CHECK(rows > 0 && rows <= INT32_MAX, "invalid row count");
  check_matrix(candidate_scores,
               at::kFloat,
               rows,
               kCandidateCapacity,
               "candidate_scores");
  check_matrix(candidate_indices,
               at::kInt,
               rows,
               kCandidateCapacity,
               "candidate_indices");
  check_matrix(output_indices, at::kInt, rows, kTopK, "output_indices");
  TORCH_CHECK(candidate_counts.is_cuda() &&
                  candidate_counts.scalar_type() == at::kInt &&
                  candidate_counts.dim() == 1 &&
                  candidate_counts.numel() == rows &&
                  candidate_counts.is_contiguous(),
              "candidate_counts must be contiguous CUDA int32 [rows]");
  TORCH_CHECK(failure_flags.is_cuda() &&
                  failure_flags.scalar_type() == at::kByte &&
                  failure_flags.dim() == 1 &&
                  failure_flags.numel() == rows &&
                  failure_flags.is_contiguous(),
              "failure_flags must be contiguous CUDA uint8 [rows]");
  const int device = candidate_scores.get_device();
  TORCH_CHECK(candidate_indices.get_device() == device &&
                  candidate_counts.get_device() == device &&
                  output_indices.get_device() == device &&
                  failure_flags.get_device() == device,
              "all tensors must share one CUDA device");

  c10::cuda::CUDAGuard guard(candidate_scores.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(device).stream();
  itk_fused_r5i_candidate_radix<<<
      static_cast<unsigned>(rows),
      kBlockThreads,
      kCandidateBytes,
      stream>>>(candidate_scores.data_ptr<float>(),
                candidate_indices.data_ptr<int32_t>(),
                candidate_counts.data_ptr<int32_t>(),
                output_indices.data_ptr<int32_t>(),
                failure_flags.data_ptr<uint8_t>(),
                static_cast<int>(rows));
  const cudaError_t status = cudaGetLastError();
  TORCH_CHECK(status == cudaSuccess,
              "fused R5i candidate radix launch failed: ",
              cudaGetErrorString(status));
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("configure", &configure, "Configure candidate radix shared memory");
  module.def("resource_report", &resource_report, "Report candidate radix resources");
  module.def("topk_out", &topk_out, "Exact TopK over compact candidates");
}
