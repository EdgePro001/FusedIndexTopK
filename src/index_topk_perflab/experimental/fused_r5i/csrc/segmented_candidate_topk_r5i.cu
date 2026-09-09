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
constexpr int kSegments = 16;
constexpr int kFastSegmentCapacity = 880;
constexpr int kRepairSegmentCapacity = 1024;
constexpr int kFastCandidateCapacity = kSegments * kFastSegmentCapacity;
constexpr int kRepairCandidateCapacity = kSegments * kRepairSegmentCapacity;
constexpr int kFastCandidateBytes = 2 * kFastCandidateCapacity * sizeof(uint32_t);
constexpr int kRepairCandidateBytes =
    2 * kRepairCandidateCapacity * sizeof(uint32_t);
constexpr unsigned kFullMask = 0xffffffffu;

static_assert(kFastCandidateCapacity == 14080);
static_assert(kRepairCandidateCapacity == 16384);
static_assert(kFastCandidateBytes == 112640);
static_assert(kRepairCandidateBytes == 131072);

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
      const bool active = index < count && score != 0 &&
          (score & prefix_mask) == threshold;
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

template <int kSegmentCapacity, bool kMaskedRepair>
__global__ __launch_bounds__(kBlockThreads, 2)
void itk_fused_r5i_segmented_candidate_radix(
    const uint64_t* __restrict__ candidate_pairs,
    const int32_t* __restrict__ segment_counts,
    const int32_t* __restrict__ k_start,
    const int32_t* __restrict__ k_end,
    int32_t* __restrict__ output_indices,
    uint8_t* __restrict__ failure_flags,
    int rows) {
  constexpr int kCandidateCapacity = kSegments * kSegmentCapacity;
  const int tx = threadIdx.x;
  const int row_stride = kMaskedRepair ? gridDim.x : rows;
  for (int row = blockIdx.x; row < rows; row += row_stride) {
    if constexpr (kMaskedRepair) {
      if (failure_flags[row] == 0) continue;
    }
    const int valid_length = k_end[row] - k_start[row];

    int count = tx < kSegments
        ? segment_counts[static_cast<int64_t>(row) * kSegments + tx] : 0;
    int total = count;
    int maximum = count;
    if (tx < kWarpSize) {
      #pragma unroll
      for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        total += __shfl_down_sync(kFullMask, total, offset);
        maximum = max(maximum, __shfl_down_sync(kFullMask, maximum, offset));
      }
    }
    __shared__ int total_count;
    __shared__ int maximum_segment_count;
    __shared__ int segment_offsets[kSegments];
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
      const bool short_row = valid_length < kTopK;
      const bool enough = kMaskedRepair
          ? total == valid_length
          : (short_row ? total == valid_length : total >= kTopK);
      failure_flags[row] = enough && maximum <= kSegmentCapacity
          ? uint8_t{0} : uint8_t{1};
    }
    __syncthreads();
    if constexpr (kMaskedRepair) {
      if (total_count != valid_length ||
          maximum_segment_count > kSegmentCapacity) {
        __syncthreads();
        continue;
      }
    } else if (maximum_segment_count > kSegmentCapacity) {
      return;
    }

    int32_t* output_row = output_indices + static_cast<int64_t>(row) * kTopK;
    if (valid_length < kTopK) {
      for (int output = tx; output < kTopK; output += kBlockThreads) {
        output_row[output] = -1;
      }
      for (int index = tx; index < kCandidateCapacity;
           index += kBlockThreads) {
        const int segment = index / kSegmentCapacity;
        const int local = index - segment * kSegmentCapacity;
        const int count_for_segment = segment_counts[
            static_cast<int64_t>(row) * kSegments + segment];
        if (local < count_for_segment) {
          const uint64_t packed = candidate_pairs[
              static_cast<int64_t>(row) * kCandidateCapacity + index];
          const int output = segment_offsets[segment] + local;
          if (output < kTopK) {
            output_row[output] = static_cast<int32_t>(packed);
          }
        }
      }
      if constexpr (kMaskedRepair) {
        __syncthreads();
        continue;
      } else {
        return;
      }
    }
    if (total_count < kTopK) {
      if constexpr (kMaskedRepair) {
        __syncthreads();
        continue;
      } else {
        return;
      }
    }

    extern __shared__ __align__(128) unsigned char dynamic_shared[];
    uint32_t* scores = reinterpret_cast<uint32_t*>(dynamic_shared);
    uint32_t* indices = scores + kCandidateCapacity;
    const int64_t candidate_row =
        static_cast<int64_t>(row) * kCandidateCapacity;
    const int warp = tx / kWarpSize;
    const int lane = tx & (kWarpSize - 1);
    // Two warps cooperatively compact each producer segment. Radix rounds then
    // scan only the valid entries instead of the full capacity.
    const int segment = warp / 2;
    const int segment_warp = warp & 1;
    const int segment_count = segment_counts[
        static_cast<int64_t>(row) * kSegments + segment];
    for (int local = segment_warp * kWarpSize + lane;
         local < segment_count;
         local += 2 * kWarpSize) {
      const int compact = segment_offsets[segment] + local;
      const uint64_t packed = candidate_pairs[
          candidate_row + segment * kSegmentCapacity + local];
      scores[compact] = static_cast<uint32_t>(packed >> 32);
      indices[compact] = static_cast<uint32_t>(packed);
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
        total_count,
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
    for (int base = 0; base < total_count; base += kBlockThreads) {
      const int index = base + tx;
      const uint32_t score = index < total_count ? scores[index] : 0;
      const bool is_greater = score > threshold;
      const bool is_equal = score != 0 && score == threshold;
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
    if constexpr (kMaskedRepair) __syncthreads();
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
  cudaError_t status = cudaFuncSetAttribute(
      itk_fused_r5i_segmented_candidate_radix<
          kFastSegmentCapacity, false>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      kFastCandidateBytes);
  TORCH_CHECK(status == cudaSuccess,
              "cannot opt in to segmented R5i candidate shared memory: ",
              cudaGetErrorString(status));
  status = cudaFuncSetAttribute(
      itk_fused_r5i_segmented_candidate_radix<
          kRepairSegmentCapacity, true>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      kRepairCandidateBytes);
  TORCH_CHECK(status == cudaSuccess,
              "cannot opt in to masked R5i repair shared memory: ",
              cudaGetErrorString(status));
}

pybind11::dict resource_report() {
  cudaFuncAttributes attributes{};
  cudaError_t status = cudaFuncGetAttributes(
      &attributes,
      itk_fused_r5i_segmented_candidate_radix<
          kFastSegmentCapacity, false>);
  TORCH_CHECK(status == cudaSuccess,
              "cannot query segmented R5i candidate kernel: ",
              cudaGetErrorString(status));
  int active_blocks_per_sm = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks_per_sm,
      itk_fused_r5i_segmented_candidate_radix<
          kFastSegmentCapacity, false>,
      kBlockThreads,
      kFastCandidateBytes);
  TORCH_CHECK(status == cudaSuccess,
              "cannot calculate segmented R5i candidate occupancy: ",
              cudaGetErrorString(status));

  cudaFuncAttributes repair_attributes{};
  status = cudaFuncGetAttributes(
      &repair_attributes,
      itk_fused_r5i_segmented_candidate_radix<
          kRepairSegmentCapacity, true>);
  TORCH_CHECK(status == cudaSuccess,
              "cannot query masked R5i repair kernel: ",
              cudaGetErrorString(status));
  int repair_active_blocks_per_sm = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &repair_active_blocks_per_sm,
      itk_fused_r5i_segmented_candidate_radix<
          kRepairSegmentCapacity, true>,
      kBlockThreads,
      kRepairCandidateBytes);
  TORCH_CHECK(status == cudaSuccess,
              "cannot calculate masked R5i repair occupancy: ",
              cudaGetErrorString(status));

  pybind11::dict report;
  report["candidate_capacity"] = kFastCandidateCapacity;
  report["segments"] = kSegments;
  report["segment_capacity"] = kFastSegmentCapacity;
  report["dynamic_shared_bytes"] = kFastCandidateBytes;
  report["static_shared_bytes"] = attributes.sharedSizeBytes;
  report["registers_per_thread"] = attributes.numRegs;
  report["active_blocks_per_sm"] = active_blocks_per_sm;
  report["repair_candidate_capacity"] = kRepairCandidateCapacity;
  report["repair_segment_capacity"] = kRepairSegmentCapacity;
  report["repair_dynamic_shared_bytes"] = kRepairCandidateBytes;
  report["repair_static_shared_bytes"] = repair_attributes.sharedSizeBytes;
  report["repair_registers_per_thread"] = repair_attributes.numRegs;
  report["repair_active_blocks_per_sm"] = repair_active_blocks_per_sm;
  return report;
}

template <bool kMaskedRepair>
void topk_out_impl(torch::Tensor candidate_pairs,
                   torch::Tensor segment_counts,
                   torch::Tensor k_start,
                   torch::Tensor k_end,
                   torch::Tensor output_indices,
                   torch::Tensor failure_flags) {
  constexpr int segment_capacity =
      kMaskedRepair ? kRepairSegmentCapacity : kFastSegmentCapacity;
  constexpr int candidate_capacity = kSegments * segment_capacity;
  constexpr int candidate_bytes = 2 * candidate_capacity * sizeof(uint32_t);
  TORCH_CHECK(candidate_pairs.dim() == 2,
              "candidate_pairs must be two dimensional");
  const int64_t rows = candidate_pairs.size(0);
  TORCH_CHECK(rows > 0 && rows <= INT32_MAX, "invalid row count");
  check_matrix(candidate_pairs,
               at::kLong,
               rows,
               candidate_capacity,
               "candidate_pairs");
  check_matrix(segment_counts,
               at::kInt,
               rows,
               kSegments,
               "segment_counts");
  TORCH_CHECK(k_start.is_cuda() && k_end.is_cuda() &&
                  k_start.scalar_type() == at::kInt &&
                  k_end.scalar_type() == at::kInt &&
                  k_start.dim() == 1 && k_end.dim() == 1 &&
                  k_start.numel() == rows && k_end.numel() == rows &&
                  k_start.is_contiguous() && k_end.is_contiguous(),
              "k_start and k_end must be contiguous CUDA int32 [rows]");
  check_matrix(output_indices, at::kInt, rows, kTopK, "output_indices");
  TORCH_CHECK(failure_flags.is_cuda() &&
                  failure_flags.scalar_type() == at::kByte &&
                  failure_flags.dim() == 1 &&
                  failure_flags.numel() == rows &&
                  failure_flags.is_contiguous(),
              "failure_flags must be contiguous CUDA uint8 [rows]");
  const int device = candidate_pairs.get_device();
  TORCH_CHECK(segment_counts.get_device() == device &&
                  k_start.get_device() == device && k_end.get_device() == device &&
                  output_indices.get_device() == device &&
                  failure_flags.get_device() == device,
              "all tensors must share one CUDA device");

  c10::cuda::CUDAGuard guard(candidate_pairs.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(device).stream();
  int grid_blocks = static_cast<int>(rows);
  if constexpr (kMaskedRepair) {
    int multiprocessor_count = 0;
    const cudaError_t attribute_status = cudaDeviceGetAttribute(
        &multiprocessor_count,
        cudaDevAttrMultiProcessorCount,
        device);
    TORCH_CHECK(attribute_status == cudaSuccess,
                "cannot query multiprocessor count for masked repair: ",
                cudaGetErrorString(attribute_status));
    grid_blocks = multiprocessor_count < rows
        ? multiprocessor_count : static_cast<int>(rows);
  }
  itk_fused_r5i_segmented_candidate_radix<
      segment_capacity, kMaskedRepair><<<
      static_cast<unsigned>(grid_blocks),
      kBlockThreads,
      candidate_bytes,
      stream>>>(reinterpret_cast<const uint64_t*>(
                    candidate_pairs.data_ptr<int64_t>()),
                segment_counts.data_ptr<int32_t>(),
                k_start.data_ptr<int32_t>(),
                k_end.data_ptr<int32_t>(),
                output_indices.data_ptr<int32_t>(),
                failure_flags.data_ptr<uint8_t>(),
                static_cast<int>(rows));
  const cudaError_t status = cudaGetLastError();
  TORCH_CHECK(status == cudaSuccess,
              kMaskedRepair
                  ? "masked R5i repair radix launch failed: "
                  : "segmented R5i candidate radix launch failed: ",
              cudaGetErrorString(status));
}

void topk_out(torch::Tensor candidate_pairs,
              torch::Tensor segment_counts,
              torch::Tensor k_start,
              torch::Tensor k_end,
              torch::Tensor output_indices,
              torch::Tensor failure_flags) {
  topk_out_impl<false>(
      candidate_pairs,
      segment_counts,
      k_start,
      k_end,
      output_indices,
      failure_flags);
}

void repair_topk_out(torch::Tensor candidate_pairs,
                     torch::Tensor segment_counts,
                     torch::Tensor k_start,
                     torch::Tensor k_end,
                     torch::Tensor output_indices,
                     torch::Tensor failure_flags) {
  topk_out_impl<true>(
      candidate_pairs,
      segment_counts,
      k_start,
      k_end,
      output_indices,
      failure_flags);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("configure", &configure, "Configure segmented candidate radix");
  module.def("resource_report", &resource_report, "Report segmented candidate resources");
  module.def("topk_out", &topk_out, "Exact TopK over segmented candidates");
  module.def(
      "repair_topk_out",
      &repair_topk_out,
      "Masked exact TopK over complete-row repair candidates");
}
