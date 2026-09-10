#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>

namespace {

constexpr int kBlockThreads = 512;
constexpr int kFinalizeThreads = 256;
constexpr int kWarpSize = 32;
constexpr int kTopK = 2048;
constexpr int kSegments = 16;
constexpr int kSegmentCapacity = 1024;
constexpr int kCandidateCapacity = kSegments * kSegmentCapacity;
constexpr int kMergeCapacity = 2 * kTopK;
constexpr int kLocalSharedBytes = kCandidateCapacity * sizeof(uint64_t);
constexpr int kMergeSharedBytes = kMergeCapacity * sizeof(uint64_t);
constexpr unsigned kFullMask = 0xffffffffu;

static_assert(kCandidateCapacity == 16384);
static_assert(kLocalSharedBytes == 131072);
static_assert(kMergeSharedBytes == 32768);

__device__ __forceinline__ void warp_histogram_add(
    int* histogram, bool active, int bin) {
  const unsigned active_mask = __ballot_sync(kFullMask, active);
  if (!active) return;
  const unsigned peers = __match_any_sync(active_mask, bin);
  const int lane = threadIdx.x & (kWarpSize - 1);
  if (lane == __ffs(static_cast<int>(peers)) - 1) {
    atomicAdd(&histogram[bin], __popc(peers));
  }
}

__device__ __forceinline__ int select_histogram_byte(
    const int* histogram,
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

template <int kThreads>
__device__ __forceinline__ uint32_t radix_select_score(
    const uint64_t* pairs,
    int count,
    int requested_rank,
    int* histogram,
    int* selected_byte,
    int* selected_greater) {
  const int tx = threadIdx.x;
  uint32_t threshold = 0;
  uint32_t prefix_mask = 0;
  int rank = requested_rank;
#pragma unroll
  for (int round = 0; round < 4; ++round) {
    const int shift = 24 - round * 8;
    if (tx < 256) histogram[tx] = 0;
    __syncthreads();
    for (int base = 0; base < count; base += kThreads) {
      const int index = base + tx;
      const uint32_t score = index < count
          ? static_cast<uint32_t>(pairs[index] >> 32) : 0u;
      const bool active = index < count && score != 0 &&
          (score & prefix_mask) == threshold;
      warp_histogram_add(
          histogram, active, static_cast<int>((score >> shift) & 0xffu));
    }
    __syncthreads();
    if (tx < kWarpSize) {
      int byte = 0;
      int greater = 0;
      if (select_histogram_byte(
              histogram, rank, &byte, &greater)) {
        *selected_byte = byte;
        *selected_greater = greater;
      }
    }
    __syncthreads();
    rank -= *selected_greater;
    threshold |= static_cast<uint32_t>(*selected_byte) << shift;
    prefix_mask |= 0xffu << shift;
  }
  return threshold;
}

__device__ __forceinline__ int warp_reserve_output(
    int* counter, bool active) {
  const unsigned active_mask = __ballot_sync(kFullMask, active);
  if (active_mask == 0) return -1;
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int leader = __ffs(static_cast<int>(active_mask)) - 1;
  int base = 0;
  if (lane == leader) base = atomicAdd(counter, __popc(active_mask));
  base = __shfl_sync(kFullMask, base, leader);
  const unsigned lower_lanes = lane == 0
      ? 0u : ((1u << lane) - 1u);
  return active ? base + __popc(active_mask & lower_lanes) : -1;
}

__device__ __forceinline__ uint64_t add_logical_offset(
    uint64_t packed, uint32_t logical_offset) {
  const uint32_t local_id = static_cast<uint32_t>(packed);
  return (packed & 0xffffffff00000000ull) |
      static_cast<uint32_t>(local_id + logical_offset);
}

template <int kThreads>
__device__ __forceinline__ void select_topk_pairs(
    const uint64_t* input,
    int count,
    uint64_t* output,
    uint32_t logical_offset,
    bool apply_offset,
    int* histogram,
    int* selected_byte,
    int* selected_greater,
    int* greater_counter,
    int* equal_counter,
    int* selection_error) {
  const int tx = threadIdx.x;
  if (tx == 0) *selection_error = 0;
  __syncthreads();
  if (count <= kTopK) {
    for (int index = tx; index < kTopK; index += kThreads) {
      uint64_t packed = index < count ? input[index] : 0ull;
      if (index < count && apply_offset) {
        packed = add_logical_offset(packed, logical_offset);
      }
      output[index] = packed;
    }
    __syncthreads();
    return;
  }

  const uint32_t threshold = radix_select_score<kThreads>(
      input,
      count,
      kTopK,
      histogram,
      selected_byte,
      selected_greater);

  if (tx == 0) {
    *greater_counter = 0;
    *equal_counter = 0;
  }
  __syncthreads();

  int local_greater = 0;
  for (int base = 0; base < count; base += kThreads) {
    const int index = base + tx;
    local_greater += index < count &&
        static_cast<uint32_t>(input[index] >> 32) > threshold;
  }
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    local_greater += __shfl_down_sync(kFullMask, local_greater, offset);
  }
  if ((tx & (kWarpSize - 1)) == 0) {
    atomicAdd(greater_counter, local_greater);
  }
  __syncthreads();

  // Preserve the completed count before reusing greater_counter as the output
  // reservation cursor.  Reading and resetting the same shared location
  // without this hand-off is a race: late readers can observe the reset zero.
  if (tx == 0) {
    *selected_greater = *greater_counter;
    *greater_counter = 0;
  }
  __syncthreads();
  const int total_greater = *selected_greater;
  const int equal_needed = kTopK - total_greater;
  for (int base = 0; base < count; base += kThreads) {
    const int index = base + tx;
    uint64_t packed = index < count ? input[index] : 0ull;
    const uint32_t score = static_cast<uint32_t>(packed >> 32);
    const bool is_greater = index < count && score > threshold;
    const bool is_equal = index < count && score != 0 && score == threshold;
    const int greater_position =
        warp_reserve_output(greater_counter, is_greater);
    const int equal_position = warp_reserve_output(equal_counter, is_equal);
    int output_position = -1;
    if (is_greater) {
      output_position = greater_position;
    } else if (is_equal && equal_position < equal_needed) {
      output_position = (kTopK - equal_needed) + equal_position;
    }
    if (output_position >= 0 && output_position < kTopK) {
      if (apply_offset) {
        packed = add_logical_offset(packed, logical_offset);
      }
      output[output_position] = packed;
    }
  }
  __syncthreads();
  if (tx == 0 &&
      (total_greater < 0 || total_greater >= kTopK ||
       *greater_counter != total_greater || equal_needed <= 0 ||
       *equal_counter < equal_needed)) {
    *selection_error = 1;
  }
  __syncthreads();
}

__global__ __launch_bounds__(kBlockThreads, 1)
void itk_r16a_chunk_local_topk(
    const uint64_t* __restrict__ candidate_pairs,
    const int32_t* __restrict__ segment_counts,
    const int32_t* __restrict__ chunk_k_start,
    const int32_t* __restrict__ chunk_k_end,
    const uint8_t* __restrict__ failure_flags,
    uint64_t* __restrict__ local_pairs,
    int32_t* __restrict__ local_counts,
    uint8_t* __restrict__ repair_error_flags,
    int rows,
    uint32_t logical_offset,
    bool first_chunk) {
  const int tx = threadIdx.x;
  extern __shared__ __align__(128) uint64_t shared_pairs[];
  __shared__ int histogram[256];
  __shared__ int segment_offsets[kSegments];
  __shared__ int total_count;
  __shared__ int maximum_segment_count;
  __shared__ int selected_byte;
  __shared__ int selected_greater;
  __shared__ int greater_counter;
  __shared__ int equal_counter;
  __shared__ int selection_error;

  for (int row = blockIdx.x; row < rows; row += gridDim.x) {
    if (failure_flags[row] == 0) continue;
    if (first_chunk && tx == 0) repair_error_flags[row] = uint8_t{0};
    __syncthreads();

    int count = tx < kSegments
        ? segment_counts[static_cast<int64_t>(row) * kSegments + tx]
        : 0;
    int total = count;
    int maximum = count;
    if (tx < kWarpSize) {
#pragma unroll
      for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        total += __shfl_down_sync(kFullMask, total, offset);
        maximum = max(
            maximum, __shfl_down_sync(kFullMask, maximum, offset));
      }
    }
    if (tx == 0) {
      total_count = total;
      maximum_segment_count = maximum;
      int offset = 0;
#pragma unroll
      for (int segment = 0; segment < kSegments; ++segment) {
        segment_offsets[segment] = offset;
        offset += segment_counts[
            static_cast<int64_t>(row) * kSegments + segment];
      }
      const int valid_length = chunk_k_end[row] - chunk_k_start[row];
      if (total != valid_length || maximum > kSegmentCapacity ||
          total > kCandidateCapacity) {
        repair_error_flags[row] = uint8_t{1};
        local_counts[row] = 0;
      }
    }
    __syncthreads();
    if (maximum_segment_count > kSegmentCapacity ||
        total_count > kCandidateCapacity || repair_error_flags[row] != 0) {
      __syncthreads();
      continue;
    }

    const int warp = tx / kWarpSize;
    const int lane = tx & (kWarpSize - 1);
    const int segment = warp;
    const int segment_count = segment_counts[
        static_cast<int64_t>(row) * kSegments + segment];
    for (int local = lane; local < segment_count; local += kWarpSize) {
      shared_pairs[segment_offsets[segment] + local] = candidate_pairs[
          static_cast<int64_t>(row) * kCandidateCapacity +
          segment * kSegmentCapacity + local];
    }
    __syncthreads();

    uint64_t* output = local_pairs + static_cast<int64_t>(row) * kTopK;
    select_topk_pairs<kBlockThreads>(
        shared_pairs,
        total_count,
        output,
        logical_offset,
        true,
        histogram,
        &selected_byte,
        &selected_greater,
        &greater_counter,
        &equal_counter,
        &selection_error);
    if (tx == 0) {
      if (selection_error != 0) {
        repair_error_flags[row] = uint8_t{2};
        local_counts[row] = 0;
      } else {
        local_counts[row] = min(total_count, kTopK);
      }
    }
    __syncthreads();
  }
}

__global__ __launch_bounds__(kBlockThreads, 1)
void itk_r16a_merge_topk_pairs(
    uint64_t* __restrict__ accumulator_pairs,
    int32_t* __restrict__ accumulator_counts,
    const uint64_t* __restrict__ local_pairs,
    const int32_t* __restrict__ local_counts,
    const uint8_t* __restrict__ failure_flags,
    uint8_t* __restrict__ repair_error_flags,
    int rows,
    bool first_chunk) {
  const int tx = threadIdx.x;
  extern __shared__ __align__(128) uint64_t shared_pairs[];
  __shared__ int histogram[256];
  __shared__ int selected_byte;
  __shared__ int selected_greater;
  __shared__ int greater_counter;
  __shared__ int equal_counter;
  __shared__ int selection_error;
  __shared__ int accumulator_count;
  __shared__ int local_count;

  for (int row = blockIdx.x; row < rows; row += gridDim.x) {
    if (failure_flags[row] == 0 || repair_error_flags[row] != 0) continue;
    if (tx == 0) {
      accumulator_count = first_chunk ? 0 : accumulator_counts[row];
      local_count = local_counts[row];
    }
    __syncthreads();
    const int combined_count = accumulator_count + local_count;
    for (int index = tx; index < combined_count; index += kBlockThreads) {
      if (index < accumulator_count) {
        shared_pairs[index] = accumulator_pairs[
            static_cast<int64_t>(row) * kTopK + index];
      } else {
        shared_pairs[index] = local_pairs[
            static_cast<int64_t>(row) * kTopK +
            index - accumulator_count];
      }
    }
    __syncthreads();

    uint64_t* output =
        accumulator_pairs + static_cast<int64_t>(row) * kTopK;
    select_topk_pairs<kBlockThreads>(
        shared_pairs,
        combined_count,
        output,
        0,
        false,
        histogram,
        &selected_byte,
        &selected_greater,
        &greater_counter,
        &equal_counter,
        &selection_error);
    if (tx == 0) {
      if (selection_error != 0) {
        repair_error_flags[row] = uint8_t{3};
      } else {
        accumulator_counts[row] = min(combined_count, kTopK);
      }
    }
    __syncthreads();
  }
}

__global__ void itk_r16a_finalize_repair(
    const uint64_t* __restrict__ accumulator_pairs,
    const int32_t* __restrict__ accumulator_counts,
    int32_t* __restrict__ output_indices,
    uint8_t* __restrict__ failure_flags,
    const uint8_t* __restrict__ repair_error_flags,
    int rows) {
  const int row = blockIdx.x;
  if (row >= rows || failure_flags[row] == 0) return;
  if (repair_error_flags[row] != 0 || accumulator_counts[row] != kTopK) {
    return;
  }
  for (int index = threadIdx.x; index < kTopK; index += blockDim.x) {
    output_indices[static_cast<int64_t>(row) * kTopK + index] =
        static_cast<int32_t>(accumulator_pairs[
            static_cast<int64_t>(row) * kTopK + index]);
  }
  __syncthreads();
  if (threadIdx.x == 0) failure_flags[row] = uint8_t{0};
}

void check_matrix(
    const torch::Tensor& tensor,
    at::ScalarType dtype,
    int64_t rows,
    int64_t columns,
    const char* name) {
  TORCH_CHECK(
      tensor.is_cuda() && tensor.scalar_type() == dtype && tensor.dim() == 2 &&
          tensor.size(0) == rows && tensor.size(1) == columns &&
          tensor.is_contiguous(),
      name,
      " has an unexpected dtype, shape, device, or layout");
}

void check_vector(
    const torch::Tensor& tensor,
    at::ScalarType dtype,
    int64_t rows,
    const char* name) {
  TORCH_CHECK(
      tensor.is_cuda() && tensor.scalar_type() == dtype && tensor.dim() == 1 &&
          tensor.numel() == rows && tensor.is_contiguous(),
      name,
      " has an unexpected dtype, shape, device, or layout");
}

int repair_grid_blocks(int rows, int device) {
  int multiprocessor_count = 0;
  const cudaError_t status = cudaDeviceGetAttribute(
      &multiprocessor_count,
      cudaDevAttrMultiProcessorCount,
      device);
  TORCH_CHECK(
      status == cudaSuccess,
      "cannot query multiprocessor count for R16a repair: ",
      cudaGetErrorString(status));
  return std::min(rows, multiprocessor_count);
}

void local_topk_pairs_out(
    torch::Tensor candidate_pairs,
    torch::Tensor segment_counts,
    torch::Tensor chunk_k_start,
    torch::Tensor chunk_k_end,
    torch::Tensor failure_flags,
    torch::Tensor local_pairs,
    torch::Tensor local_counts,
    torch::Tensor repair_error_flags,
    int64_t logical_offset,
    bool first_chunk) {
  TORCH_CHECK(candidate_pairs.dim() == 2, "candidate_pairs must be [Q, 16384]");
  const int64_t rows64 = candidate_pairs.size(0);
  TORCH_CHECK(rows64 > 0 && rows64 <= INT32_MAX, "invalid R16a row count");
  TORCH_CHECK(
      logical_offset >= 0 && logical_offset <= UINT32_MAX,
      "invalid R16a logical offset");
  check_matrix(candidate_pairs, at::kLong, rows64, kCandidateCapacity, "candidate_pairs");
  check_matrix(segment_counts, at::kInt, rows64, kSegments, "segment_counts");
  check_vector(chunk_k_start, at::kInt, rows64, "chunk_k_start");
  check_vector(chunk_k_end, at::kInt, rows64, "chunk_k_end");
  check_vector(failure_flags, at::kByte, rows64, "failure_flags");
  check_matrix(local_pairs, at::kLong, rows64, kTopK, "local_pairs");
  check_vector(local_counts, at::kInt, rows64, "local_counts");
  check_vector(repair_error_flags, at::kByte, rows64, "repair_error_flags");

  const int device = candidate_pairs.get_device();
  TORCH_CHECK(
      segment_counts.get_device() == device &&
          chunk_k_start.get_device() == device &&
          chunk_k_end.get_device() == device &&
          failure_flags.get_device() == device &&
          local_pairs.get_device() == device &&
          local_counts.get_device() == device &&
          repair_error_flags.get_device() == device,
      "all R16a local reducer tensors must share one CUDA device");
  c10::cuda::CUDAGuard guard(candidate_pairs.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(device).stream();
  const int rows = static_cast<int>(rows64);
  itk_r16a_chunk_local_topk<<<
      repair_grid_blocks(rows, device),
      kBlockThreads,
      kLocalSharedBytes,
      stream>>>(
      reinterpret_cast<const uint64_t*>(candidate_pairs.data_ptr<int64_t>()),
      segment_counts.data_ptr<int32_t>(),
      chunk_k_start.data_ptr<int32_t>(),
      chunk_k_end.data_ptr<int32_t>(),
      failure_flags.data_ptr<uint8_t>(),
      reinterpret_cast<uint64_t*>(local_pairs.data_ptr<int64_t>()),
      local_counts.data_ptr<int32_t>(),
      repair_error_flags.data_ptr<uint8_t>(),
      rows,
      static_cast<uint32_t>(logical_offset),
      first_chunk);
  const cudaError_t status = cudaGetLastError();
  TORCH_CHECK(
      status == cudaSuccess,
      "R16a chunk-local TopK launch failed: ",
      cudaGetErrorString(status));
}

void merge_topk_pairs_out(
    torch::Tensor accumulator_pairs,
    torch::Tensor accumulator_counts,
    torch::Tensor local_pairs,
    torch::Tensor local_counts,
    torch::Tensor failure_flags,
    torch::Tensor repair_error_flags,
    bool first_chunk) {
  TORCH_CHECK(accumulator_pairs.dim() == 2, "accumulator_pairs must be [Q, K]");
  const int64_t rows64 = accumulator_pairs.size(0);
  TORCH_CHECK(rows64 > 0 && rows64 <= INT32_MAX, "invalid R16a row count");
  check_matrix(accumulator_pairs, at::kLong, rows64, kTopK, "accumulator_pairs");
  check_vector(accumulator_counts, at::kInt, rows64, "accumulator_counts");
  check_matrix(local_pairs, at::kLong, rows64, kTopK, "local_pairs");
  check_vector(local_counts, at::kInt, rows64, "local_counts");
  check_vector(failure_flags, at::kByte, rows64, "failure_flags");
  check_vector(repair_error_flags, at::kByte, rows64, "repair_error_flags");

  const int device = accumulator_pairs.get_device();
  TORCH_CHECK(
      accumulator_counts.get_device() == device &&
          local_pairs.get_device() == device &&
          local_counts.get_device() == device &&
          failure_flags.get_device() == device &&
          repair_error_flags.get_device() == device,
      "all R16a merge tensors must share one CUDA device");
  c10::cuda::CUDAGuard guard(accumulator_pairs.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(device).stream();
  const int rows = static_cast<int>(rows64);
  itk_r16a_merge_topk_pairs<<<
      repair_grid_blocks(rows, device),
      kBlockThreads,
      kMergeSharedBytes,
      stream>>>(
      reinterpret_cast<uint64_t*>(accumulator_pairs.data_ptr<int64_t>()),
      accumulator_counts.data_ptr<int32_t>(),
      reinterpret_cast<const uint64_t*>(local_pairs.data_ptr<int64_t>()),
      local_counts.data_ptr<int32_t>(),
      failure_flags.data_ptr<uint8_t>(),
      repair_error_flags.data_ptr<uint8_t>(),
      rows,
      first_chunk);
  const cudaError_t status = cudaGetLastError();
  TORCH_CHECK(
      status == cudaSuccess,
      "R16a hierarchical pair merge launch failed: ",
      cudaGetErrorString(status));
}

void finalize_repair_out(
    torch::Tensor accumulator_pairs,
    torch::Tensor accumulator_counts,
    torch::Tensor output_indices,
    torch::Tensor failure_flags,
    torch::Tensor repair_error_flags) {
  TORCH_CHECK(accumulator_pairs.dim() == 2, "accumulator_pairs must be [Q, K]");
  const int64_t rows64 = accumulator_pairs.size(0);
  TORCH_CHECK(rows64 > 0 && rows64 <= INT32_MAX, "invalid R16a row count");
  check_matrix(accumulator_pairs, at::kLong, rows64, kTopK, "accumulator_pairs");
  check_vector(accumulator_counts, at::kInt, rows64, "accumulator_counts");
  check_matrix(output_indices, at::kInt, rows64, kTopK, "output_indices");
  check_vector(failure_flags, at::kByte, rows64, "failure_flags");
  check_vector(repair_error_flags, at::kByte, rows64, "repair_error_flags");

  const int device = accumulator_pairs.get_device();
  TORCH_CHECK(
      accumulator_counts.get_device() == device &&
          output_indices.get_device() == device &&
          failure_flags.get_device() == device &&
          repair_error_flags.get_device() == device,
      "all R16a finalize tensors must share one CUDA device");
  c10::cuda::CUDAGuard guard(accumulator_pairs.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(device).stream();
  const int rows = static_cast<int>(rows64);
  itk_r16a_finalize_repair<<<
      rows, kFinalizeThreads, 0, stream>>>(
      reinterpret_cast<const uint64_t*>(accumulator_pairs.data_ptr<int64_t>()),
      accumulator_counts.data_ptr<int32_t>(),
      output_indices.data_ptr<int32_t>(),
      failure_flags.data_ptr<uint8_t>(),
      repair_error_flags.data_ptr<uint8_t>(),
      rows);
  const cudaError_t status = cudaGetLastError();
  TORCH_CHECK(
      status == cudaSuccess,
      "R16a repair finalize launch failed: ",
      cudaGetErrorString(status));
}

void configure() {
  cudaError_t status = cudaFuncSetAttribute(
      itk_r16a_chunk_local_topk,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      kLocalSharedBytes);
  TORCH_CHECK(
      status == cudaSuccess,
      "cannot opt in to R16a chunk-local shared memory: ",
      cudaGetErrorString(status));
}

pybind11::dict resource_report() {
  cudaFuncAttributes local_attributes{};
  cudaFuncAttributes merge_attributes{};
  TORCH_CHECK(
      cudaFuncGetAttributes(&local_attributes, itk_r16a_chunk_local_topk) ==
          cudaSuccess,
      "cannot query R16a local reducer resources");
  TORCH_CHECK(
      cudaFuncGetAttributes(&merge_attributes, itk_r16a_merge_topk_pairs) ==
          cudaSuccess,
      "cannot query R16a merge reducer resources");
  pybind11::dict report;
  report["chunk_elements"] = kCandidateCapacity;
  report["top_k"] = kTopK;
  report["local_block_threads"] = kBlockThreads;
  report["local_dynamic_shared_bytes"] = kLocalSharedBytes;
  report["local_static_shared_bytes"] = local_attributes.sharedSizeBytes;
  report["local_registers_per_thread"] = local_attributes.numRegs;
  report["merge_dynamic_shared_bytes"] = kMergeSharedBytes;
  report["merge_static_shared_bytes"] = merge_attributes.sharedSizeBytes;
  report["merge_registers_per_thread"] = merge_attributes.numRegs;
  return report;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("configure", &configure, "Configure R16a long-context repair");
  module.def("resource_report", &resource_report, "Report R16a repair resources");
  module.def(
      "local_topk_pairs_out",
      &local_topk_pairs_out,
      "Exact chunk-local TopK pairs for flagged rows");
  module.def(
      "merge_topk_pairs_out",
      &merge_topk_pairs_out,
      "Merge one chunk TopK into the exact row accumulator");
  module.def(
      "finalize_repair_out",
      &finalize_repair_out,
      "Commit repaired IDs and clear successfully repaired flags");
}
