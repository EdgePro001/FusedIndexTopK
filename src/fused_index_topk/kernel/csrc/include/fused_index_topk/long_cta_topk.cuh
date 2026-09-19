#pragma once

#include <cstdint>

#ifndef ITK_LONG_TUNING
#define ITK_LONG_TUNING 0
#endif
#ifndef ITK_LONG_DIAGNOSTIC
#define ITK_LONG_DIAGNOSTIC 0
#endif

namespace fused_index_topk::kernel::long_context {

// Diagnostic builds only. Independent Math/TopK writers own disjoint fields.
// clock64 is SM-local: compare only records from the same persistent CTA.
__device__ __forceinline__ uint64_t phase_clock() {
    if constexpr (ITK_LONG_DIAGNOSTIC) return clock64();
    return 0;
}
__device__ __forceinline__ void phase_stamp(int64_t* trace, int field, bool leader) {
    if constexpr (ITK_LONG_DIAGNOSTIC)
        if (leader) trace[field] = phase_clock();
}

// Only the 96 selection threads join this non-aligned named barrier.
// Producer/consumer ownership is handled separately by per-slot mbarriers.
__device__ __forceinline__ void selection_sync() {
    asm volatile("barrier.sync 15, 96;" : : : "memory");
}

constexpr int kTopK = 2048;
constexpr int kSegments = 8;
// Rare overflow only: full score + global ID, never segment-relative IDs.
constexpr int kSpillPerSegment = 256;
constexpr int kSpillPerRow = kSegments * kSpillPerSegment;

template <int Capacity>
struct CandidateStorage {
    uint64_t pairs[2][Capacity];
    __device__ __forceinline__ void store_pair(int row, int index, uint64_t pair) {
        pairs[row][index] = pair;
    }
    __device__ __forceinline__ uint32_t load_score(int row, int index) const {
        return pairs[row][index] >> 32;
    }
    __device__ __forceinline__ uint64_t load_pair(int row, int index) const {
        return pairs[row][index];
    }
    __device__ __forceinline__ int32_t load_index(int row, int index) const {
        return static_cast<int32_t>(pairs[row][index]);
    }
};

// The storage segment supplies ID bits [7:4] (16 segments) or [6:4]
// (8 segments). Store the remaining bits losslessly; FP32 scores are unchanged.
template <>
struct CandidateStorage<5888> {
    uint32_t scores[2][5888];
    uint16_t indices[2][5888];
    __device__ __forceinline__ void store_pair(int row, int index, uint64_t pair) {
        scores[row][index] = pair >> 32;
        const uint32_t id = static_cast<uint32_t>(pair);
        indices[row][index] = static_cast<uint16_t>(((id >> 7) << 4) | (id & 15u));
    }
    __device__ __forceinline__ uint32_t load_score(int row, int index) const {
        return scores[row][index];
    }
    __device__ __forceinline__ uint64_t load_pair(int row, int index) const {
        return static_cast<uint64_t>(scores[row][index]) << 32 | static_cast<uint32_t>(load_index(row, index));
    }
    __device__ __forceinline__ int32_t load_index(int row, int index) const {
        const uint32_t packed = indices[row][index];
        const uint32_t segment = index / (5888 / kSegments);
        return static_cast<int32_t>(((packed >> 4) << 7) | (segment << 4) | (packed & 15u));
    }
};

template <int Capacity>
struct alignas(16) SelectionScratch : CandidateStorage<Capacity> {
    int counts[2][kSegments];
    int histogram[256];
    uint32_t threshold;
    int rank;
    int failed;
    int greater_counter;
    int equal_counter;
    int spill_count;  // Uses existing alignment padding; SMEM allocation is unchanged.
};

static_assert(sizeof(SelectionScratch<7680>) == 124000);
static_assert(sizeof(SelectionScratch<4096>) == 66656);
static_assert(sizeof(SelectionScratch<5888>) == 71776);

// All 32 lanes call; only active lanes receive a unique compact offset.
__device__ __forceinline__ int reserve(int* counter, bool active) {
    const unsigned mask = __ballot_sync(0xffffffffu, active);
    if (!mask) return -1;
    const int lane = threadIdx.x & 31;
    const int leader = __ffs(mask) - 1;
    int base = 0;
    if (lane == leader) base = atomicAdd(counter, __popc(mask));
    base = __shfl_sync(0xffffffffu, base, leader);
    const unsigned lower = (1u << lane) - 1u;
    return active ? base + __popc(mask & lower) : -1;
}

// Called by exactly 96 threads, with tid in [0,96). The complete candidates
// stay in CTA SMEM. A fixed-capacity segment is never read past its count.
// Rank is 1-based and equal scores may be returned in any order.
template <int Capacity>
__device__ __forceinline__ void select_row(
        SelectionScratch<Capacity>* scratch, int slot, int tid,
        int start, int end, int32_t* output, uint8_t* failure, int64_t* trace,
        const uint64_t* spill_pairs) {
    phase_stamp(trace, 0, tid == 0);
    constexpr int kCandidateCapacity = Capacity;
    constexpr int kSegmentCapacity = Capacity / kSegments;
    const int valid = end > start ? end - start : 0;
    if (valid <= kTopK) {
        for (int i = tid; i < kTopK; i += 96)
            output[i] = i < valid ? start + i : -1;
        if (tid == 0) *failure = 0;
        selection_sync();
        return;
    }
    if (tid == 0) {
        int total = 0;
        int failed = 0;
        int maximum = 0;
        int spills = 0;
        for (int s = 0; s < kSegments; ++s) {
            const int count = scratch->counts[slot][s];
            total += count;
            maximum = count > maximum ? count : maximum;
            failed |= count > kSegmentCapacity + kSpillPerSegment;
            spills += count > kSegmentCapacity ? count - kSegmentCapacity : 0;
        }
        scratch->failed = failed || total < kTopK;
        scratch->threshold = 0;
        scratch->rank = kTopK;
        scratch->greater_counter = 0;
        scratch->equal_counter = 0;
        scratch->spill_count = spills;
        *failure = scratch->failed;
        if constexpr (ITK_LONG_DIAGNOSTIC) {
            trace[12] = total;
            trace[13] = maximum;
            trace[14] = total < kTopK;
            trace[15] = failed;
            trace[16] = spills;
        }
    }
    selection_sync();
    if (scratch->failed) {
        for (int i = tid; i < kTopK; i += 96) output[i] = -1;
        selection_sync();
        return;
    }

    phase_stamp(trace, 1, tid == 0);
    uint32_t prefix_mask = 0;
    for (int shift = 24; shift >= 0; shift -= 8) {
        const uint64_t hist_begin = phase_clock();
        for (int bin = tid; bin < 256; bin += 96)
            scratch->histogram[bin] = 0;
        selection_sync();
        const uint32_t prefix = scratch->threshold;
        if constexpr (ITK_LONG_TUNING & 2) {
            // One warp owns one segment at a time. Scan only its live count,
            // with no per-item division/remainder by the segment capacity.
            for (int segment = tid / 32; segment < kSegments; segment += 3) {
                const int count = min(scratch->counts[slot][segment], kSegmentCapacity);
                for (int offset = tid % 32; offset < count; offset += 32) {
                    const uint32_t score = scratch->load_score(
                        slot, segment * kSegmentCapacity + offset);
                    if ((score & prefix_mask) == prefix)
                        atomicAdd(&scratch->histogram[(score >> shift) & 255u], 1);
                }
            }
        } else for (int i = tid; i < kCandidateCapacity; i += 96) {
            const int segment = i / kSegmentCapacity;
            const int offset = i % kSegmentCapacity;
            if (offset < scratch->counts[slot][segment]) {
                const uint32_t score = scratch->load_score(slot, i);
                if ((score & prefix_mask) == prefix)
                    atomicAdd(&scratch->histogram[(score >> shift) & 255u], 1);
            }
        }
        if (scratch->spill_count) {
            for (int segment = tid / 32; segment < kSegments; segment += 3) {
                const int count = scratch->counts[slot][segment] - kSegmentCapacity;
                for (int offset = tid % 32; offset < count; offset += 32) {
                    const uint32_t score = spill_pairs[segment * kSpillPerSegment + offset] >> 32;
                    if ((score & prefix_mask) == prefix)
                        atomicAdd(&scratch->histogram[(score >> shift) & 255u], 1);
                }
            }
        }
        selection_sync();
        const uint64_t bucket_begin = phase_clock();
        if constexpr (ITK_LONG_DIAGNOSTIC)
            if (tid == 0) trace[2 + (24 - shift) / 8] = bucket_begin - hist_begin;
        // One full warp scans 8 consecutive bins per lane. The suffix sum
        // gives each lane the number of candidates in strictly higher lanes.
        // Every shuffle is unconditional within this participating warp.
        if (tid < 32) {
            const int lane = tid;
            int counts[8];
            int total = 0;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                counts[j] = scratch->histogram[lane * 8 + j];
                total += counts[j];
            }
            int suffix = total;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                const int incoming = __shfl_down_sync(0xffffffffu, suffix, offset);
                if (lane + offset < 32) suffix += incoming;
            }
            const int rank = scratch->rank;
            const uint32_t old_threshold = scratch->threshold;
            // All lanes must snapshot rank before the winning lane updates it.
            __syncwarp();
            int greater = suffix - total;
            #pragma unroll
            for (int j = 7; j >= 0; --j) {
                const int next = greater + counts[j];
                if (greater < rank && rank <= next) {
                    scratch->threshold = old_threshold |
                        static_cast<uint32_t>(lane * 8 + j) << shift;
                    scratch->rank = rank - greater;
                }
                greater = next;
            }
        }
        selection_sync();
        if constexpr (ITK_LONG_DIAGNOSTIC)
            if (tid == 0) trace[6 + (24 - shift) / 8] = phase_clock() - bucket_begin;
        prefix_mask |= 0xffu << shift;
    }

    phase_stamp(trace, 10, tid == 0);
    const uint32_t threshold = scratch->threshold;
    const int equal_needed = scratch->rank;
    const int total_greater = kTopK - equal_needed;
    for (int i = tid; i < kCandidateCapacity; i += 96) {
        const bool valid_slot =
            i % kSegmentCapacity < scratch->counts[slot][i / kSegmentCapacity];
        uint64_t pair = 0;
        uint32_t score = 0;
        if constexpr (ITK_LONG_TUNING & 1) {
            if (valid_slot) score = scratch->load_score(slot, i);
        } else {
            pair = valid_slot ? scratch->load_pair(slot, i) : 0;
            score = pair >> 32;
        }
        const bool greater = valid_slot && score > threshold;
        const bool equal = valid_slot && score == threshold;
        const int g = reserve(&scratch->greater_counter, greater);
        const int e = reserve(&scratch->equal_counter, equal);
        if constexpr (ITK_LONG_TUNING & 1) {
            if (greater || (equal && e < equal_needed))
                output[greater ? g : total_greater + e] = scratch->load_index(slot, i);
        } else {
            if (greater) output[g] = static_cast<int32_t>(pair);
            if (equal && e < equal_needed)
                output[total_greater + e] = static_cast<int32_t>(pair);
        }
    }
    if (scratch->spill_count) {
        for (int segment = tid / 32; segment < kSegments; segment += 3) {
            const int count = scratch->counts[slot][segment] - kSegmentCapacity;
            // Loop condition is warp-uniform, including partial final groups.
            for (int base = 0; base < count; base += 32) {
                const int offset = base + tid % 32;
                const bool live = offset < count;
                const uint64_t pair = live ? spill_pairs[segment * kSpillPerSegment + offset] : 0;
                const uint32_t score = pair >> 32;
                const bool greater = live && score > threshold;
                const bool equal = live && score == threshold;
                const int g = reserve(&scratch->greater_counter, greater);
                const int e = reserve(&scratch->equal_counter, equal);
                if (greater) output[g] = static_cast<int32_t>(pair);
                if (equal && e < equal_needed)
                    output[total_greater + e] = static_cast<int32_t>(pair);
            }
        }
    }
    selection_sync();
    phase_stamp(trace, 11, tid == 0);
}

} // namespace fused_index_topk::kernel::long_context
