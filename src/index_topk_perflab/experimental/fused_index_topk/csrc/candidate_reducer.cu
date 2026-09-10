#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>

#ifndef ITK_R11_PREFIX_FIRST_BITS
#define ITK_R11_PREFIX_FIRST_BITS 9
#endif

#ifndef ITK_R11_KERNEL_NAME
#define ITK_R11_KERNEL_NAME itk_fused_r11d_segmented_candidate_radix
#endif

#ifndef ITK_R11_FAST_WORKING_CAPACITY
#define ITK_R11_FAST_WORKING_CAPACITY 6656
#endif

namespace {

constexpr int kFastBlockThreads = 512;
constexpr int kRepairBlockThreads = 1024;
constexpr int kWarpSize = 32;
constexpr int kTopK = 2048;
constexpr int kSegments = 16;
constexpr int kFastSegmentCapacity = 880;
constexpr int kRepairSegmentCapacity = 1024;
constexpr int kFastCandidateCapacity = kSegments * kFastSegmentCapacity;
constexpr int kRepairCandidateCapacity = kSegments * kRepairSegmentCapacity;
constexpr int kFastWorkingCapacity = ITK_R11_FAST_WORKING_CAPACITY;
constexpr int kWarpTailCapacity = 256;
constexpr int kPrefixFirstBits = ITK_R11_PREFIX_FIRST_BITS;
constexpr int kPrefixHistogramBins = 1 << kPrefixFirstBits;
constexpr int kFastWorkingBytes = 2 * kFastWorkingCapacity * sizeof(uint32_t);
constexpr int kRepairCandidateBytes =
    2 * kRepairCandidateCapacity * sizeof(uint32_t);
constexpr unsigned kFullMask = 0xffffffffu;

static_assert(kFastCandidateCapacity == 14080);
static_assert(kRepairCandidateCapacity == 16384);
static_assert(kFastWorkingCapacity == 6656 || kFastWorkingCapacity == 6400);
static_assert(kFastWorkingBytes == 53248 || kFastWorkingBytes == 51200);
static_assert(kRepairCandidateBytes == 131072);
static_assert(kPrefixFirstBits == 9 || kPrefixFirstBits == 10);

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

// Resolve a 9-bit digit without giving every thread sixteen live counters.
// Each lane still owns eight high-byte groups, but combines the two adjacent
// ninth-bit bins before the warp suffix scan. The winning lane then resolves
// the ninth bit from the two original counters.
__device__ __forceinline__ int select_histogram_nine_bits(
    const int* histogram,
    int rank,
    int* selected_digit,
    int* greater_count) {
  const int lane = threadIdx.x & (kWarpSize - 1);
  int counts[8];
  int lane_total = 0;
  #pragma unroll
  for (int item = 0; item < 8; ++item) {
    const int byte = lane * 8 + item;
    counts[item] = histogram[byte * 2] + histogram[byte * 2 + 1];
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
      const int byte = lane * 8 + item;
      const int high_half = histogram[byte * 2 + 1];
      const int rank_in_byte = rank - greater;
      if (rank_in_byte <= high_half) {
        *selected_digit = byte * 2 + 1;
        *greater_count = greater;
      } else {
        *selected_digit = byte * 2;
        *greater_count = greater + high_half;
      }
      found = 1;
    }
    greater = next;
  }
  return found;
}

__device__ __forceinline__ int select_histogram_seven_bits(
    const int* histogram,
    int rank,
    int* selected_digit,
    int* greater_count) {
  const int lane = threadIdx.x & (kWarpSize - 1);
  int counts[4];
  int lane_total = 0;
  #pragma unroll
  for (int item = 0; item < 4; ++item) {
    counts[item] = histogram[lane * 4 + item];
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
  for (int item = 3; item >= 0; --item) {
    const int next = greater + counts[item];
    if (greater < rank && rank <= next) {
      *selected_digit = lane * 4 + item;
      *greater_count = greater;
      found = 1;
    }
    greater = next;
  }
  return found;
}

// Resolve a 10-bit digit while preserving the same eight high-byte groups per
// lane as the 9-bit selector. The winning group is refined across four bins.
__device__ __forceinline__ int select_histogram_ten_bits(
    const int* histogram,
    int rank,
    int* selected_digit,
    int* greater_count) {
  const int lane = threadIdx.x & (kWarpSize - 1);
  int counts[8];
  int lane_total = 0;
  #pragma unroll
  for (int item = 0; item < 8; ++item) {
    const int byte = lane * 8 + item;
    counts[item] = histogram[byte * 4] + histogram[byte * 4 + 1] +
        histogram[byte * 4 + 2] + histogram[byte * 4 + 3];
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
      const int byte = lane * 8 + item;
      int sub_greater = greater;
      #pragma unroll
      for (int sub = 3; sub >= 0; --sub) {
        const int sub_next = sub_greater + histogram[byte * 4 + sub];
        if (sub_greater < rank && rank <= sub_next) {
          *selected_digit = byte * 4 + sub;
          *greater_count = sub_greater;
          found = 1;
        }
        sub_greater = sub_next;
      }
    }
    greater = next;
  }
  return found;
}

__device__ __forceinline__ int select_histogram_six_bits(
    const int* histogram,
    int rank,
    int* selected_digit,
    int* greater_count) {
  const int lane = threadIdx.x & (kWarpSize - 1);
  int counts[2];
  int lane_total = 0;
  #pragma unroll
  for (int item = 0; item < 2; ++item) {
    counts[item] = histogram[lane * 2 + item];
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
  for (int item = 1; item >= 0; --item) {
    const int next = greater + counts[item];
    if (greater < rank && rank <= next) {
      *selected_digit = lane * 2 + item;
      *greater_count = greater;
      found = 1;
    }
    greater = next;
  }
  return found;
}

template <int kKernelThreads>
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
    for (int base = 0; base < count; base += kKernelThreads) {
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

template <int kKernelThreads>
__device__ __forceinline__ uint32_t radix_select_prefix16_shared(
    const uint32_t* scores,
    int count,
    int requested_rank,
    int* histogram,
    int* selected_byte,
    int* selected_greater,
    int* remaining_rank) {
  const int tx = threadIdx.x;
  static_assert(kKernelThreads == kFastBlockThreads);
  int rank = requested_rank;

  // The first pass resolves either 9 or 10 high bits. Every thread clears one
  // bin; the 10-bit variant clears a second bin at tx + 512.
  histogram[tx] = 0;
  if constexpr (kPrefixFirstBits == 10) histogram[tx + 512] = 0;
  __syncthreads();
  for (int base = 0; base < count; base += kKernelThreads) {
    const int index = base + tx;
    const uint32_t score = index < count ? scores[index] : 0;
    const bool active = index < count && score != 0;
    const int digit = kPrefixFirstBits == 9
        ? static_cast<int>((score >> 23) & 0x1ffu)
        : static_cast<int>((score >> 22) & 0x3ffu);
    warp_histogram_add(histogram, active, digit);
  }
  __syncthreads();
  if (tx < kWarpSize) {
    int digit = 0;
    int greater = 0;
    const int found = kPrefixFirstBits == 9
        ? select_histogram_nine_bits(histogram, rank, &digit, &greater)
        : select_histogram_ten_bits(histogram, rank, &digit, &greater);
    if (found) {
      *selected_byte = digit;
      *selected_greater = greater;
    }
  }
  __syncthreads();
  rank -= *selected_greater;
  uint32_t threshold = static_cast<uint32_t>(*selected_byte) <<
      (32 - kPrefixFirstBits);

  // The second pass resolves the remaining 7 or 6 bits to the same exact
  // 16-bit prefix.
  constexpr int kSecondBits = 16 - kPrefixFirstBits;
  constexpr int kSecondBins = 1 << kSecondBits;
  if (tx < kSecondBins) histogram[tx] = 0;
  __syncthreads();
  for (int base = 0; base < count; base += kKernelThreads) {
    const int index = base + tx;
    const uint32_t score = index < count ? scores[index] : 0;
    constexpr uint32_t kPrefixMask = kPrefixFirstBits == 9
        ? 0xff800000u : 0xffc00000u;
    const bool active = index < count && score != 0 &&
        (score & kPrefixMask) == threshold;
    warp_histogram_add(
        histogram,
        active,
        static_cast<int>((score >> 16) & (kSecondBins - 1)));
  }
  __syncthreads();
  if (tx < kWarpSize) {
    int digit = 0;
    int greater = 0;
    const int found = kPrefixFirstBits == 9
        ? select_histogram_seven_bits(histogram, rank, &digit, &greater)
        : select_histogram_six_bits(histogram, rank, &digit, &greater);
    if (found) {
      *selected_byte = digit;
      *selected_greater = greater;
    }
  }
  __syncthreads();
  rank -= *selected_greater;
  threshold |= static_cast<uint32_t>(*selected_byte) << 16;
  *remaining_rank = rank;
  return threshold;
}

__device__ __forceinline__ int warp_reserve_output(int* counter, bool active);

__device__ __forceinline__ void tail_group_sync() {
  asm volatile("bar.sync 1, 256;" ::: "memory");
}

// Threads [0, 256) call this helper after the CTA has already constructed the
// third-byte histogram while compacting the selected 16-bit-prefix tail. Only
// the fourth radix byte remains here.
__device__ __forceinline__ void group_select_suffix8(
    const uint32_t* scores,
    const uint32_t* indices,
    int count,
    int remaining_rank,
    int requested_rank16,
    uint32_t prefix24_threshold,
    int output_offset,
    int32_t* output,
    int* histogram,
    int* selected_byte,
    int* selected_greater,
    int* greater_counter,
    int* equal_counter) {
  const int tx = threadIdx.x;
  uint32_t threshold = prefix24_threshold;
  int rank = remaining_rank;

  histogram[tx] = 0;
  tail_group_sync();
  const uint32_t score = tx < count ? scores[tx] : 0;
  const bool active = tx < count && score != 0 &&
      (score & 0xffffff00u) == threshold;
  warp_histogram_add(
      histogram, active, static_cast<int>(score & 0xffu));
  tail_group_sync();

  if (tx < kWarpSize) {
    int byte = 0;
    int greater = 0;
    if (select_histogram_byte(histogram, rank, &byte, &greater)) {
      *selected_byte = byte;
      *selected_greater = greater;
    }
  }
  tail_group_sync();
  rank -= *selected_greater;
  threshold |= static_cast<uint32_t>(*selected_byte);

  const int tail_greater = requested_rank16 - rank;
  if (tx == 0) {
    *greater_counter = 0;
    *equal_counter = 0;
  }
  tail_group_sync();
  const bool is_greater = tx < count && score > threshold;
  const bool is_equal = tx < count && score != 0 && score == threshold;
  const int greater_position =
      warp_reserve_output(greater_counter, is_greater);
  const int equal_position = warp_reserve_output(equal_counter, is_equal);
  if (is_greater) {
    output[output_offset + greater_position] =
        static_cast<int32_t>(indices[tx]);
  } else if (is_equal && equal_position < rank) {
    output[output_offset + tail_greater + equal_position] =
        static_cast<int32_t>(indices[tx]);
  }
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

template <int kSegmentCapacity, bool kMaskedRepair,
          int kKernelThreads, int kMinBlocksPerSm>
__global__ __launch_bounds__(kKernelThreads, kMinBlocksPerSm)
void ITK_R11_KERNEL_NAME(
    const uint64_t* __restrict__ candidate_pairs,
    const int32_t* __restrict__ segment_counts,
    const int32_t* __restrict__ k_start,
    const int32_t* __restrict__ k_end,
    int32_t* __restrict__ output_indices,
    uint8_t* __restrict__ failure_flags,
    int rows) {
  constexpr int kCandidateCapacity = kSegments * kSegmentCapacity;
  constexpr int kWorkingCapacity =
      kMaskedRepair ? kCandidateCapacity : kFastWorkingCapacity;
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
      failure_flags[row] = enough && maximum <= kSegmentCapacity &&
              total <= kWorkingCapacity
          ? uint8_t{0} : uint8_t{1};
    }
    __syncthreads();
    if constexpr (kMaskedRepair) {
      if (total_count != valid_length ||
          maximum_segment_count > kSegmentCapacity ||
          total_count > kWorkingCapacity) {
        __syncthreads();
        continue;
      }
    } else if (maximum_segment_count > kSegmentCapacity ||
               total_count > kWorkingCapacity) {
      return;
    }

    int32_t* output_row = output_indices + static_cast<int64_t>(row) * kTopK;
    if (valid_length < kTopK) {
      for (int output = tx; output < kTopK; output += kKernelThreads) {
        output_row[output] = -1;
      }
      for (int index = tx; index < kCandidateCapacity;
           index += kKernelThreads) {
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
    uint32_t* indices = scores + kWorkingCapacity;
    const int64_t candidate_row =
        static_cast<int64_t>(row) * kCandidateCapacity;
    const int warp = tx / kWarpSize;
    const int lane = tx & (kWarpSize - 1);
    constexpr int kWarpsPerSegment =
        kKernelThreads / (kWarpSize * kSegments);
    static_assert(kWarpsPerSegment >= 1);
    const int segment = warp / kWarpsPerSegment;
    const int segment_warp = warp % kWarpsPerSegment;
    const int segment_count = segment_counts[
        static_cast<int64_t>(row) * kSegments + segment];
    for (int local = segment_warp * kWarpSize + lane;
         local < segment_count;
         local += kWarpsPerSegment * kWarpSize) {
      const int compact = segment_offsets[segment] + local;
      const uint64_t packed = candidate_pairs[
          candidate_row + segment * kSegmentCapacity + local];
      scores[compact] = static_cast<uint32_t>(packed >> 32);
      indices[compact] = static_cast<uint32_t>(packed);
    }

    __shared__ __align__(128) int histogram[kPrefixHistogramBins];
    __shared__ int selected_byte;
    __shared__ int selected_greater;
    __shared__ int greater_counter;
    __shared__ int equal_counter;
    __shared__ int selected_counter;
    __syncthreads();

    if constexpr (!kMaskedRepair) {
      int prefix_rank = 0;
      const uint32_t prefix_threshold =
          radix_select_prefix16_shared<kKernelThreads>(
          scores,
          total_count,
          kTopK,
          histogram,
          &selected_byte,
          &selected_greater,
          &prefix_rank);

      if (tx == 0) {
        greater_counter = 0;
        selected_counter = 0;
      }
      if (tx < 256) histogram[tx] = 0;
      __syncthreads();

      const uint32_t selected_prefix = prefix_threshold >> 16;
      for (int base = 0; base < total_count; base += kKernelThreads) {
        const int index = base + tx;
        const uint32_t score = index < total_count ? scores[index] : 0;
        const uint32_t prefix = score >> 16;
        const bool is_greater = index < total_count && score != 0 &&
            prefix > selected_prefix;
        const bool is_selected = index < total_count && score != 0 &&
            prefix == selected_prefix;
        const int greater_position =
            warp_reserve_output(&greater_counter, is_greater);
        const int selected_position =
            warp_reserve_output(&selected_counter, is_selected);
        if (is_selected) {
          atomicAdd(&histogram[(score >> 8) & 0xffu], 1);
        }
        if (is_greater) {
          output_row[greater_position] =
              static_cast<int32_t>(indices[index]);
        }
        if (is_selected && selected_position < kWarpTailCapacity &&
            total_count + selected_position < kWorkingCapacity) {
          scores[total_count + selected_position] = score;
          indices[total_count + selected_position] = indices[index];
        }
      }
      __syncthreads();

      if (tx == 0) {
        const int expected_greater = kTopK - prefix_rank;
        if (greater_counter != expected_greater ||
            selected_counter < prefix_rank ||
            selected_counter > kWarpTailCapacity ||
            total_count + selected_counter > kWorkingCapacity) {
          failure_flags[row] = uint8_t{1};
        }
      }
      __syncthreads();
      if (failure_flags[row] != 0) return;

      if (tx < kWarpSize) {
        int byte = 0;
        int greater = 0;
        if (select_histogram_byte(
                histogram, prefix_rank, &byte, &greater)) {
          selected_byte = byte;
          selected_greater = greater;
        }
      }
      __syncthreads();
      const int prefix24_rank = prefix_rank - selected_greater;
      const uint32_t prefix24_threshold =
          prefix_threshold | (static_cast<uint32_t>(selected_byte) << 8);

      if (tx < kWarpTailCapacity) {
        group_select_suffix8(
            scores + total_count,
            indices + total_count,
            selected_counter,
            prefix24_rank,
            prefix_rank,
            prefix24_threshold,
            kTopK - prefix_rank,
            output_row,
            histogram,
            &selected_byte,
            &selected_greater,
            &greater_counter,
            &equal_counter);
      }
      return;
    } else {
      int remaining_rank = 0;
      const uint32_t threshold = radix_select_shared<kKernelThreads>(
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
      for (int base = 0; base < total_count; base += kKernelThreads) {
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
      __syncthreads();
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
  cudaError_t status = cudaFuncSetAttribute(
      ITK_R11_KERNEL_NAME<
          kFastSegmentCapacity, false, kFastBlockThreads, 4>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      kFastWorkingBytes);
  TORCH_CHECK(status == cudaSuccess,
              "cannot opt in to segmented R11d candidate shared memory: ",
              cudaGetErrorString(status));
  status = cudaFuncSetAttribute(
      ITK_R11_KERNEL_NAME<
          kRepairSegmentCapacity, true, kRepairBlockThreads, 1>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      kRepairCandidateBytes);
  TORCH_CHECK(status == cudaSuccess,
              "cannot opt in to masked R11d repair shared memory: ",
              cudaGetErrorString(status));
}

pybind11::dict resource_report() {
  cudaFuncAttributes attributes{};
  cudaError_t status = cudaFuncGetAttributes(
      &attributes,
      ITK_R11_KERNEL_NAME<
          kFastSegmentCapacity, false, kFastBlockThreads, 4>);
  TORCH_CHECK(status == cudaSuccess,
              "cannot query segmented R11d candidate kernel: ",
              cudaGetErrorString(status));
  int active_blocks_per_sm = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks_per_sm,
      ITK_R11_KERNEL_NAME<
          kFastSegmentCapacity, false, kFastBlockThreads, 4>,
      kFastBlockThreads,
      kFastWorkingBytes);
  TORCH_CHECK(status == cudaSuccess,
              "cannot calculate segmented R11d candidate occupancy: ",
              cudaGetErrorString(status));

  cudaFuncAttributes repair_attributes{};
  status = cudaFuncGetAttributes(
      &repair_attributes,
      ITK_R11_KERNEL_NAME<
          kRepairSegmentCapacity, true, kRepairBlockThreads, 1>);
  TORCH_CHECK(status == cudaSuccess,
              "cannot query masked R11d repair kernel: ",
              cudaGetErrorString(status));
  int repair_active_blocks_per_sm = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &repair_active_blocks_per_sm,
      ITK_R11_KERNEL_NAME<
          kRepairSegmentCapacity, true, kRepairBlockThreads, 1>,
      kRepairBlockThreads,
      kRepairCandidateBytes);
  TORCH_CHECK(status == cudaSuccess,
              "cannot calculate masked R11d repair occupancy: ",
              cudaGetErrorString(status));

  pybind11::dict report;
  report["candidate_capacity"] = kFastCandidateCapacity;
  report["working_capacity"] = kFastWorkingCapacity;
  report["segments"] = kSegments;
  report["segment_capacity"] = kFastSegmentCapacity;
  report["block_threads"] = kFastBlockThreads;
  report["dynamic_shared_bytes"] = kFastWorkingBytes;
  report["static_shared_bytes"] = attributes.sharedSizeBytes;
  report["registers_per_thread"] = attributes.numRegs;
  report["active_blocks_per_sm"] = active_blocks_per_sm;
  report["repair_candidate_capacity"] = kRepairCandidateCapacity;
  report["repair_segment_capacity"] = kRepairSegmentCapacity;
  report["repair_block_threads"] = kRepairBlockThreads;
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
  constexpr int working_capacity =
      kMaskedRepair ? candidate_capacity : kFastWorkingCapacity;
  constexpr int candidate_bytes =
      2 * working_capacity * sizeof(uint32_t);
  constexpr int block_threads =
      kMaskedRepair ? kRepairBlockThreads : kFastBlockThreads;
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
  ITK_R11_KERNEL_NAME<
      segment_capacity,
      kMaskedRepair,
      block_threads,
      kMaskedRepair ? 1 : 4><<<
      static_cast<unsigned>(grid_blocks),
      block_threads,
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
                  ? "masked R11d repair radix launch failed: "
                  : "segmented R11d candidate radix launch failed: ",
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
