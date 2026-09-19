#pragma once

// Project-local producer-side fusion derivative of DeepGEMM commit
// 7c95b14aa4a66edd7b682e5acdde62351ca81197.
// The official checkout is not modified. Final FP32 score tiles are filtered
// against per-row sampled thresholds and retained in CTA shared memory.
// Three selection warps produce exact TopK indices in this same kernel.

#include <cstdint>
#include <type_traits>

#ifndef ITK_SHORT_MATH_SCHEDULE
#define ITK_SHORT_MATH_SCHEDULE 2
#endif
#ifndef ITK_SHORT_MATH_REGISTERS
#define ITK_SHORT_MATH_REGISTERS 168
#endif
#ifndef ITK_SHORT_CACHE_WEIGHTS
#define ITK_SHORT_CACHE_WEIGHTS 0
#endif
#include "short_cta_topk.cuh"

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>
#include <cute/arch/mma_sm90_desc.hpp>

#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/common/sm90_utils.cuh>

namespace fused_index_topk::kernel::short_context {

using namespace deep_gemm;
using namespace deep_gemm::sm90;

__device__ __forceinline__ uint64_t pack_score_id(
        float score, uint32_t logical_id) {
    uint32_t bits = __float_as_uint(score);
    if ((bits & 0x7fffffffu) == 0)
        bits = 0;
    const uint32_t ordered =
        (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
    return (static_cast<uint64_t>(ordered) << 32) | logical_id;
}

// ReSharper disable once CppNotAllPathsReturnValue
template <uint32_t kHeadDim>
static constexpr int to_swizzle_cute_type() {
    DG_STATIC_ASSERT(kHeadDim == 32 or kHeadDim == 64 or kHeadDim == 128, "Invalid swizzling");
    if constexpr (kHeadDim == 32)
        return static_cast<int>(cute::SM90::GMMA::LayoutType::B32);
    if constexpr (kHeadDim == 64)
        return static_cast<int>(cute::SM90::GMMA::LayoutType::B64);
    if constexpr (kHeadDim == 128)
        return static_cast<int>(cute::SM90::GMMA::LayoutType::B128);
}

#ifndef ITK_SHORT_KERNEL_NAME
#define ITK_SHORT_KERNEL_NAME itk_fused_index_topk_short
#endif

template <uint32_t kNumHeads, uint32_t kHeadDim,
          uint32_t BLOCK_Q, uint32_t BLOCK_KV,
          uint32_t kNumQStages, uint32_t kNumKVStages,
          uint32_t kNumTMAThreads, uint32_t kNumMathThreads,
          uint32_t TOP_K, uint32_t CANDIDATE_CAPACITY,
          uint32_t kCandidateSlots, bool kOverlap>
__global__ __launch_bounds__(kNumTMAThreads + kNumMathThreads, 1)
void ITK_SHORT_KERNEL_NAME(
                         const uint32_t seq_len, const uint32_t seq_len_kv,
                         uint32_t* cu_seq_len_k_start,
                         uint32_t* cu_seq_len_k_end,
                         const float* sample_thresholds,
                         int32_t* output_indices,
                         uint8_t* failure_flags,
                         int64_t* phase_trace,
                         const __grid_constant__ cute::TmaDescriptor tensor_map_q,
                         const __grid_constant__ cute::TmaDescriptor tensor_map_kv,
                         const __grid_constant__ cute::TmaDescriptor tensor_map_kv_scales,
                         const __grid_constant__ cute::TmaDescriptor tensor_map_weights) {
    // Persistent CTA: Math produces pair i while TopK consumes pair i-1.
    // Per-slot ready/free mbarriers prevent reuse while the consumer reads.
    // Q should be load only at once for a block
    const auto& num_q_blocks = ceil_div(seq_len, BLOCK_Q);

    // Types
    using WGMMA = typename FP8MMASelector<BLOCK_Q * kNumHeads>::type;
    using Barrier = cutlass::arch::ClusterTransactionBarrier;

    static constexpr uint32_t kNumTMAWarpThreads = 32;
    static constexpr uint32_t kNumMathWarps = kNumMathThreads / 32;
    static constexpr uint32_t kNumCandidateSegments = 16;
    static constexpr uint32_t kWarpSegments =
        kNumCandidateSegments / kNumMathWarps;
    static constexpr uint32_t kSegmentCandidateCapacity =
        CANDIDATE_CAPACITY / kNumCandidateSegments;
    // Prefetch TMA descriptors
    DG_STATIC_ASSERT(kNumHeads == 64 and kHeadDim == 128 and
                     (kNumQStages == 2 or kNumQStages == 3) and kNumKVStages == 3,
                     "short-context fused TopK requires Q stages 2/3 and KV stages 3");
    DG_STATIC_ASSERT(
        BLOCK_Q == 2 and
        BLOCK_KV == 128 and kNumMathThreads == 256,
        "short-context fused TopK requires [2,128]/256");
    DG_STATIC_ASSERT(TOP_K == 2048, "short-context fused TopK requires exact Top-2048");
    DG_STATIC_ASSERT((CANDIDATE_CAPACITY == 7680 and kNumQStages == 3 and kCandidateSlots == 1 and not kOverlap) or
                     ((CANDIDATE_CAPACITY == 4096 or CANDIDATE_CAPACITY == 5888) and kNumQStages == 2 and kCandidateSlots == 2),
                     "short-context fused TopK supports parallel control or compact double-slot layout");
    DG_STATIC_ASSERT(kNumTMAThreads == 128,
                     "Invalid short-context fused TopK specialized threads");
    DG_STATIC_ASSERT(
        kNumMathWarps == 8 and kWarpSegments == 2,
        "short-context fused TopK requires 16 logical segments across 8 math warps");
    DG_STATIC_ASSERT(CANDIDATE_CAPACITY % kNumCandidateSegments == 0,
                     "candidate segments must divide evenly");
    if (threadIdx.x / 32 == kNumMathThreads / 32 and cute::elect_one_sync()) {
        cute::prefetch_tma_descriptor(&tensor_map_q);
        cute::prefetch_tma_descriptor(&tensor_map_kv);
        cute::prefetch_tma_descriptor(&tensor_map_kv_scales);
        cute::prefetch_tma_descriptor(&tensor_map_weights);
    }
    __syncwarp();

    // Shared memory configs
    // NOTES: weight may be unaligned
    static constexpr uint32_t kSwizzleAlignment = kHeadDim * 8;
    static constexpr uint32_t SMEM_Q_SIZE_PER_STAGE = BLOCK_Q * kNumHeads * kHeadDim * sizeof(__nv_fp8_e4m3);
    static constexpr uint32_t SMEM_WEIGHT_SIZE_PER_STAGE = BLOCK_Q * kNumHeads * sizeof(float);
    static constexpr uint32_t SMEM_KV_SIZE_PER_STAGE = BLOCK_KV * kHeadDim * sizeof(__nv_fp8_e4m3);
    static constexpr uint32_t SMEM_KV_SCALE_SIZE_PER_STAGE = BLOCK_KV * sizeof(float);
    static constexpr uint32_t kGemmDataBytes =
        kNumQStages * (SMEM_Q_SIZE_PER_STAGE + SMEM_WEIGHT_SIZE_PER_STAGE) +
        kNumKVStages * (SMEM_KV_SIZE_PER_STAGE + SMEM_KV_SCALE_SIZE_PER_STAGE);
    static constexpr uint32_t kAllBarrierBytes =
        (2 * kNumQStages + 2 * kNumKVStages + 2 * kCandidateSlots) * sizeof(Barrier);
    // Preserve the candidate storage offset for the parallel-only control.
    static constexpr uint32_t kCandidateOffset = kNumQStages == 3 ? 101632 : 84608;
    DG_STATIC_ASSERT(kGemmDataBytes + kAllBarrierBytes <= kCandidateOffset,
                     "candidate buffers overlap GEMM/barrier storage");

    // Align to swizzling alignment bytes
    extern __shared__ __align__(kSwizzleAlignment) uint8_t smem_buffer[];
    DG_STATIC_ASSERT(SMEM_Q_SIZE_PER_STAGE % kSwizzleAlignment == 0, "Unaligned TMA swizzling");
    DG_STATIC_ASSERT(SMEM_KV_SIZE_PER_STAGE % kSwizzleAlignment == 0, "Unaligned TMA swizzling");

    // Data on shared memory
    auto smem_q = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer +
            SMEM_Q_SIZE_PER_STAGE * i);
    });
    auto smem_kv = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer + (
            SMEM_Q_SIZE_PER_STAGE * kNumQStages + SMEM_KV_SIZE_PER_STAGE * i));
    });
    auto smem_weights = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<float*>(smem_buffer +
            SMEM_Q_SIZE_PER_STAGE * kNumQStages + SMEM_KV_SIZE_PER_STAGE * kNumKVStages + SMEM_WEIGHT_SIZE_PER_STAGE * i);
    });
    auto smem_kv_scales = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<float*>(smem_buffer +
            SMEM_Q_SIZE_PER_STAGE * kNumQStages + SMEM_KV_SIZE_PER_STAGE * kNumKVStages +
            SMEM_WEIGHT_SIZE_PER_STAGE * kNumQStages + SMEM_KV_SCALE_SIZE_PER_STAGE * i);
    });

    auto* selections = reinterpret_cast<SelectionScratch<CANDIDATE_CAPACITY>*>(
        smem_buffer + kCandidateOffset);

    // TMA barriers
    auto barrier_ptr = reinterpret_cast<Barrier*>(smem_kv_scales[kNumKVStages]);
    auto full_q_barriers   = PatternVisitor([&](const uint32_t& i) { return barrier_ptr + i; });
    auto empty_q_barriers  = PatternVisitor([&](const uint32_t& i) { return barrier_ptr + (kNumQStages + i); });
    auto full_kv_barriers  = PatternVisitor([&](const uint32_t& i) { return barrier_ptr + (kNumQStages * 2 + i); });
    auto empty_kv_barriers = PatternVisitor([&](const uint32_t& i) { return barrier_ptr + (kNumQStages * 2 + kNumKVStages + i); });
    auto* candidate_ready = barrier_ptr + 2 * kNumQStages + 2 * kNumKVStages;
    auto* candidate_free = candidate_ready + kCandidateSlots;

    // Initialize barriers
    const bool& is_tma_load_warp = kNumMathThreads <= threadIdx.x and
                                   threadIdx.x < kNumMathThreads + kNumTMAWarpThreads;
    if (is_tma_load_warp and cute::elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumQStages; ++ i) {
            full_q_barriers[i]->init(1);
            empty_q_barriers[i]->init(kNumMathThreads);
        }
        #pragma unroll
        for (uint32_t i = 0; i < kNumKVStages; ++ i) {
            full_kv_barriers[i]->init(1);
            empty_kv_barriers[i]->init(kNumMathThreads);
        }
        for (uint32_t i = 0; i < kCandidateSlots; ++i) {
            candidate_ready[i].init(kNumMathThreads);
            candidate_free[i].init(96);
        }

        // Make initialized barrier visible in async proxy
        cutlass::arch::fence_barrier_init();
    }
    __syncthreads();

    // Register reconfigurations
    constexpr uint32_t kNumTMARegisters = 64;
    // Initial cubin allocation is 168 registers/thread for this 384-thread CTA.
    // setmaxnreg borrows from that CTA pool, not all unallocated SM registers.
    constexpr uint32_t kNumMathRegisters = ITK_SHORT_MATH_REGISTERS;
    DG_STATIC_ASSERT(ITK_SHORT_MATH_SCHEDULE == 1 or ITK_SHORT_MATH_SCHEDULE == 2,
                     "short-context fused TopK only uses the verified paired/steady WGMMA control flows");
    DG_STATIC_ASSERT(kNumMathRegisters >= 168 and kNumMathRegisters <= 216 and
                     kNumMathRegisters % 8 == 0, "Invalid short-context fused TopK Math register budget");
    DG_STATIC_ASSERT(kNumMathThreads * kNumMathRegisters +
                     kNumTMAThreads * kNumTMARegisters <= 384 * 168,
                     "Per-role register requests exceed the initial CTA pool");

    // Block scheduler
    uint32_t block_q_idx = blockIdx.x, q_iter_idx = 0;
    const auto& get_next_block_q_idx = [&]() -> cute::tuple<uint32_t, uint32_t> {
        return {block_q_idx + gridDim.x, q_iter_idx + 1};
    };
    const auto& load_schedule = [&](const uint32_t& q_iter_offset = 0) -> cute::tuple<uint32_t, uint32_t, uint32_t, uint32_t> {
        uint32_t end = cute::numeric_limits<uint32_t>::min();

        #pragma unroll
        for (uint32_t i = 0; i < BLOCK_Q; ++ i) {
            const auto& q_idx = block_q_idx * BLOCK_Q + i;
            end = max(end, min(__ldg(cu_seq_len_k_end + q_idx), seq_len_kv));
        }
        return {(q_iter_idx + q_iter_offset) % kNumQStages,       // Q pipeline stage
                ((q_iter_idx + q_iter_offset) / kNumQStages) & 1, // Q pipeline phase
                0u, ceil_div(end, BLOCK_KV)};                     // Task info
    };

    // KV pipeline
    uint32_t num_total_kv_blocks = 0;
    const auto& get_kv_pipeline = [&](const uint32_t& kv_block_idx) -> cute::tuple<uint32_t, uint32_t> {
        return {
            (num_total_kv_blocks + kv_block_idx) % kNumKVStages,         // KV pipeline stage
            ((num_total_kv_blocks + kv_block_idx) / kNumKVStages) & 1    // KV pipeline phase
        };
    };

    if (threadIdx.x >= kNumMathThreads) {
        // One TMA warp and three TopK warps share this specialized warpgroup.
        // All 128 threads first execute the register redistribution instruction.
        cutlass::arch::warpgroup_reg_dealloc<kNumTMARegisters>();
        if (not is_tma_load_warp) {
            const int tid = threadIdx.x - kNumMathThreads - 32;
            while (block_q_idx < num_q_blocks) {
                const uint32_t slot = q_iter_idx % kCandidateSlots;
                const uint32_t phase = (q_iter_idx / kCandidateSlots) & 1;
                int64_t* trace = nullptr;
                if constexpr (ITK_SHORT_DIAGNOSTIC) trace = phase_trace + block_q_idx * 64;
                phase_stamp(trace, 4, tid == 0);
                candidate_ready[slot].wait(phase);
                phase_stamp(trace, 5, tid == 0);
                auto* selection = selections + slot;
                for (uint32_t i = 0; i < BLOCK_Q; ++i) {
                    const uint32_t row = block_q_idx * BLOCK_Q + i;
                    const int start = min(__ldg(cu_seq_len_k_start + row), seq_len_kv);
                    const int end = min(__ldg(cu_seq_len_k_end + row), seq_len_kv);
                    int64_t* row_trace = nullptr;
                    if constexpr (ITK_SHORT_DIAGNOSTIC) row_trace = trace + 16 + i * 24;
                    select_row(selection, i, tid, start, end,
                        output_indices + static_cast<uint64_t>(row) * TOP_K,
                        failure_flags + row, row_trace);
                }
                phase_stamp(trace, 6, tid == 0);
                // All 96 consumers release after their final SMEM reads.
                candidate_free[slot].arrive();
                CUTE_TIE(get_next_block_q_idx(), block_q_idx, q_iter_idx);
            }
            return;
        }

        // Prefetch
        const auto& issue_tma_q = [&](const uint32_t& stage_idx, const auto& block_idx) {
            tma_copy(&tensor_map_q, reinterpret_cast<uint64_t*>(full_q_barriers[stage_idx]), smem_q[stage_idx], 0, block_idx * BLOCK_Q * kNumHeads);
            tma_copy(&tensor_map_weights, reinterpret_cast<uint64_t*>(full_q_barriers[stage_idx]), smem_weights[stage_idx], 0, block_idx * BLOCK_Q);
            full_q_barriers[stage_idx]->arrive_and_expect_tx(SMEM_Q_SIZE_PER_STAGE + SMEM_WEIGHT_SIZE_PER_STAGE);
        };
        if (cute::elect_one_sync() and block_q_idx < num_q_blocks)
            issue_tma_q(0, block_q_idx);

        // Only the first lane persistently schedules over blocks
        if (cute::elect_one_sync()) {
            while (block_q_idx < num_q_blocks) {
                CUTE_TIE_DECL(load_schedule(1), q_stage_idx, q_phase, kv_start, num_kv_blocks);

                // Wait Q consumer release
                empty_q_barriers[q_stage_idx]->wait(q_phase ^ 1);

                // Issue TMA Q
                if (const auto& next_block_q_idx = cute::get<0>(get_next_block_q_idx()); next_block_q_idx < num_q_blocks)
                    issue_tma_q(q_stage_idx, next_block_q_idx);

                // Issue TMA KV
                #pragma unroll
                for (uint32_t kv_block_idx = 0; kv_block_idx < num_kv_blocks; ++ kv_block_idx) {
                    // Wait consumer release
                    CUTE_TIE_DECL(get_kv_pipeline(kv_block_idx), kv_stage_idx, kv_phase);
                    empty_kv_barriers[kv_stage_idx]->wait(kv_phase ^ 1);

                    // Issue TMA KV
                    tma_copy(&tensor_map_kv, reinterpret_cast<uint64_t*>(full_kv_barriers[kv_stage_idx]),
                             smem_kv[kv_stage_idx], 0, kv_start + kv_block_idx * BLOCK_KV);
                    tma_copy(&tensor_map_kv_scales, reinterpret_cast<uint64_t*>(full_kv_barriers[kv_stage_idx]),
                             smem_kv_scales[kv_stage_idx], kv_start + kv_block_idx * BLOCK_KV, 0);
                    full_kv_barriers[kv_stage_idx]->arrive_and_expect_tx(SMEM_KV_SIZE_PER_STAGE + SMEM_KV_SCALE_SIZE_PER_STAGE);
                }
                num_total_kv_blocks += num_kv_blocks;

                // Jump to the next block
                CUTE_TIE(get_next_block_q_idx(), block_q_idx, q_iter_idx);
            }
        }
    } else {
        // Math warp-groups for WGMMA
        cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();

        // NOTES: use `__shfl_sync` to encourage NVCC to use unified registers
        const auto& thread_idx = threadIdx.x % kNumMathThreads;
        const auto& warp_idx = __shfl_sync(0xffffffff, thread_idx / 32, 0);
        const auto& warpgroup_idx = warp_idx / 4;
        const auto& lane_idx = get_lane_idx();
        float accum[WGMMA::kNumAccum], weights[BLOCK_Q][kNumHeads / 4];

        const auto& warp_offset = warp_idx * 16;
        const auto& v_0_offset = lane_idx / 4 + 0;
        const auto& v_1_offset = lane_idx / 4 + 8;

        while (block_q_idx < num_q_blocks) {
            CUTE_TIE_DECL(load_schedule(), q_stage_idx, q_phase, kv_start, num_kv_blocks);
            const uint32_t candidate_slot = q_iter_idx % kCandidateSlots;
            const uint32_t candidate_phase = (q_iter_idx / kCandidateSlots) & 1;
            int64_t* trace = nullptr;
            if constexpr (ITK_SHORT_DIAGNOSTIC) trace = phase_trace + block_q_idx * 64;
            phase_stamp(trace, 0, thread_idx == 0);
            // Initial free phase is zero, so waiting on phase one admits the
            // first use; subsequent uses wait for the preceding consumer.
            candidate_free[candidate_slot].wait(candidate_phase ^ 1);
            phase_stamp(trace, 1, thread_idx == 0);
            auto* selection = selections + candidate_slot;
            auto* segment_counts = &selection->counts[0][0];

            uint32_t row_start[BLOCK_Q], row_end[BLOCK_Q];
            float thresholds[BLOCK_Q];
            int warp_candidate_counts[BLOCK_Q][kWarpSegments] = {{0}};
            #pragma unroll
            for (uint32_t i = 0; i < BLOCK_Q; ++ i) {
                const uint32_t q_idx = block_q_idx * BLOCK_Q + i;
                row_end[i] = min(__ldg(cu_seq_len_k_end + q_idx), seq_len_kv);
                row_start[i] = min(__ldg(cu_seq_len_k_start + q_idx), seq_len_kv);
                thresholds[i] = __ldg(sample_thresholds + q_idx);
            }

            // Wait TMA Q arrival
            full_q_barriers[q_stage_idx]->wait(q_phase);
            phase_stamp(trace, 2, thread_idx == 0);

            uint64_t kv_wait_cycles = 0, issue_cycles = 0;
            uint64_t wgmma_wait_cycles = 0, consume_cycles = 0;

            // Cache once per Q pair; this adds 32 long-lived FP32 registers.
            // CACHE_WEIGHTS=0 is the unchanged non-cached SMEM reload path.
            if constexpr (ITK_SHORT_CACHE_WEIGHTS) {
            // Read weights
            #pragma unroll
            for (uint32_t i = 0; i < BLOCK_Q; ++ i) {
                #pragma unroll
                for (uint32_t j = 0; j < kNumHeads / 4; ++ j)
                    weights[i][j] = ld_shared(smem_weights[q_stage_idx] + i * kNumHeads + (j / 2) * 8 + (j & 1) + (lane_idx % 4) * 2);
            }

            }

            // The GEMM math and score reduction are inherited from DeepGEMM.
            const auto& issue_wgmma_tile = [&](const uint32_t& kv_block_idx,
                                               auto& tile_accum,
                                               float& scale_kv_0,
                                               float& scale_kv_1) {
                CUTE_TIE_DECL(get_kv_pipeline(kv_block_idx), kv_stage_idx, kv_phase);
                const uint64_t before_ready = phase_clock();
                full_kv_barriers[kv_stage_idx]->wait(kv_phase);
                const uint64_t after_ready = phase_clock();
                if constexpr (ITK_SHORT_DIAGNOSTIC)
                    kv_wait_cycles += after_ready - before_ready;

                scale_kv_0 = ld_shared(
                    smem_kv_scales[kv_stage_idx] + warp_offset + v_0_offset);
                scale_kv_1 = ld_shared(
                    smem_kv_scales[kv_stage_idx] + warp_offset + v_1_offset);

                DG_STATIC_ASSERT(BLOCK_KV == kNumMathThreads / 2, "Invalid block size");
                DG_STATIC_ASSERT(kHeadDim % WGMMA::K == 0, "Invalid head dim");
                #pragma unroll
                for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i)
                    warpgroup_fence_operand(tile_accum[i]);
                warpgroup_arrive();
                #pragma unroll
                for (uint32_t k = 0; k < kHeadDim / WGMMA::K; ++ k) {
                    auto desc_a = make_smem_desc(smem_kv[kv_stage_idx] + (warpgroup_idx * WGMMA::M) * kHeadDim + k * WGMMA::K,
                                                 to_swizzle_cute_type<kHeadDim>(), 0, kHeadDim * 8);
                    auto desc_b = make_smem_desc(smem_q[q_stage_idx] + k * WGMMA::K,
                                                 to_swizzle_cute_type<kHeadDim>(), 0, kHeadDim * 8);
                    WGMMA::wgmma(desc_a, desc_b, tile_accum, k);
                }
                warpgroup_commit_batch();
                #pragma unroll
                for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i)
                    warpgroup_fence_operand(tile_accum[i]);
                if constexpr (ITK_SHORT_DIAGNOSTIC)
                    issue_cycles += phase_clock() - after_ready;
            };

            // Preserve DeepGEMM score arithmetic; the compact output now goes to
            // CTA SMEM instead of global memory. Also honor the row start.
            const auto& consume_wgmma_tile = [&](const uint32_t& kv_block_idx,
                                                 auto& tile_accum,
                                                 const float& scale_kv_0,
                                                 const float& scale_kv_1) {
                const uint64_t before_consume = phase_clock();
                CUTE_TIE_DECL(get_kv_pipeline(kv_block_idx), kv_stage_idx, kv_phase);
                empty_kv_barriers[kv_stage_idx]->arrive();

                const auto& kv_offset = kv_start + kv_block_idx * BLOCK_KV + warp_offset;
                const uint32_t segment_slot =
                    ((kv_start / BLOCK_KV) + kv_block_idx) &
                    (kWarpSegments - 1u);
                const uint32_t segment_idx =
                    warp_idx + segment_slot * kNumMathWarps;
                static constexpr uint32_t kNumAccumPerReduce = kNumHeads / 2;
                DG_STATIC_ASSERT(WGMMA::kNumAccum % kNumAccumPerReduce == 0, "Invalid accumulation");
                DG_STATIC_ASSERT(WGMMA::kNumAccum / kNumAccumPerReduce == BLOCK_Q, "Invalid accumulation");
                DG_STATIC_ASSERT(kNumHeads % 8 == 0, "Invalid head");
                #pragma unroll
                for (uint32_t i = 0; i < BLOCK_Q; ++ i) {
                    auto shifted_accum = tile_accum + i * kNumAccumPerReduce;
                    const auto& transform = [&](const uint32_t& j) {
                        const uint32_t weight_idx = (j / 4) * 2 + (j & 1);
                        float weight;
                        if constexpr (not ITK_SHORT_CACHE_WEIGHTS) {
                            weight = ld_shared(smem_weights[q_stage_idx] +
                                i * kNumHeads + (weight_idx / 2) * 8 +
                                (weight_idx & 1) + (lane_idx % 4) * 2);
                        } else {
                            weight = weights[i][weight_idx];
                        }
                        return fmaxf(shifted_accum[j], 0) * weight;
                    };

                    // Intra-thread reduction
                    float sum[4] = {transform(0), transform(1), transform(2), transform(3)};
                    #pragma unroll
                    for (uint32_t j = 1; j < kNumHeads / 8; ++ j) {
                        #pragma unroll
                        for (uint32_t k = 0; k < 4; k ++)
                            sum[k] += transform(j * 4 + k);
                    }
                    float v_0 = (sum[0] + sum[1]) * scale_kv_0;
                    float v_1 = (sum[2] + sum[3]) * scale_kv_1;

                    // Inter-thread reduction
                    #pragma unroll
                    for (uint32_t j = 0; j < 2; ++ j) {
                        const auto& offset = static_cast<int>(1u << j);
                        v_0 += __shfl_xor_sync(0xffffffffu, v_0, offset);
                        v_1 += __shfl_xor_sync(0xffffffffu, v_1, offset);
                    }

                    // Lanes 0 and 1 in each four-lane reduction group represent
                    // its two final scores. Keep the values in their original
                    // accumulators so selecting one does not extend register
                    // liveness; one ballot still compacts all 16 scores.
                    const uint32_t role = lane_idx & 3u;
                    const uint32_t id_0 = kv_offset + v_0_offset;
                    const uint32_t id_1 = kv_offset + v_1_offset;
                    const bool active_0 = role == 0u and
                        id_0 >= row_start[i] and id_0 < row_end[i] and v_0 >= thresholds[i];
                    const bool active_1 = role == 1u and
                        id_1 >= row_start[i] and id_1 < row_end[i] and v_1 >= thresholds[i];
                    const bool active = active_0 or active_1;
                    const uint32_t active_mask =
                        __ballot_sync(0xffffffffu, active);
                    const uint32_t lower_lanes = lane_idx == 0
                        ? 0u : ((1u << lane_idx) - 1u);
                    int base;
                    if constexpr (ITK_SHORT_TUNING & 4) {
                        // A variable array subscript lowers this tiny counter
                        // array to local memory. Explicit alternatives let the
                        // compiler keep both counters in registers.
                        static_assert(kWarpSegments == 2);
                        base = segment_slot == 0 ? warp_candidate_counts[i][0]
                                                 : warp_candidate_counts[i][1];
                    } else {
                        base = warp_candidate_counts[i][segment_slot];
                    }
                    const int position =
                        base + __popc(active_mask & lower_lanes);
                    const int segment_offset = segment_idx * kSegmentCandidateCapacity;
                    if (active_0 and position < kSegmentCandidateCapacity)
                        selection->store_pair(i, segment_offset + position,
                            pack_score_id(v_0, id_0));
                    if (active_1 and position < kSegmentCandidateCapacity)
                        selection->store_pair(i, segment_offset + position,
                            pack_score_id(v_1, id_1));
                    const int next_count = base + __popc(active_mask);
                    if constexpr (ITK_SHORT_TUNING & 4) {
                        if (segment_slot == 0) warp_candidate_counts[i][0] = next_count;
                        else warp_candidate_counts[i][1] = next_count;
                    } else {
                        warp_candidate_counts[i][segment_slot] = next_count;
                    }
                }
                if constexpr (ITK_SHORT_DIAGNOSTIC)
                    consume_cycles += phase_clock() - before_consume;
            };

            const auto& wait_wgmma = [&](auto groups) {
                const uint64_t before_wait = phase_clock();
                warpgroup_wait<decltype(groups)::value>();
                if constexpr (ITK_SHORT_DIAGNOSTIC)
                    wgmma_wait_cycles += phase_clock() - before_wait;
            };

            // Each issue commits exactly one tile. wait<1> retires the older
            // accumulator before it is read; wait<0> drains the odd/even tail.
            // consume releases only the completed tile's KV stage. Persistent
            // Q-stage reuse occurs after the final wait<0>, never while live.
            if constexpr (ITK_SHORT_MATH_SCHEDULE == 1) {
                // Bounded two-tile batch: no asynchronous accumulator crosses
                // a loop backedge. Preserve one completed tile for consume
                // while the second group is allowed to remain outstanding.
                float second[WGMMA::kNumAccum];
                uint32_t tile = 0;
                for (; tile + 1 < num_kv_blocks; tile += 2) {
                    float a0, a1, b0, b1;
                    issue_wgmma_tile(tile, accum, a0, a1);
                    issue_wgmma_tile(tile + 1, second, b0, b1);
                    wait_wgmma(std::integral_constant<int, 1>{});
                    consume_wgmma_tile(tile, accum, a0, a1);
                    wait_wgmma(std::integral_constant<int, 0>{});
                    consume_wgmma_tile(tile + 1, second, b0, b1);
                }
                if (tile < num_kv_blocks) {
                    float a0, a1;
                    issue_wgmma_tile(tile, accum, a0, a1);
                    wait_wgmma(std::integral_constant<int, 0>{});
                    consume_wgmma_tile(tile, accum, a0, a1);
                }
            } else if constexpr (ITK_SHORT_MATH_SCHEDULE == 2) {
                // Branch-free steady-state pipeline. Conditional tails are
                // outside the loop, unlike the paired schedule.
                float second[WGMMA::kNumAccum];
                float a0, a1, b0, b1;
                if (num_kv_blocks > 0) {
                    issue_wgmma_tile(0, accum, a0, a1);
                    uint32_t tile = 0;
                    for (; tile + 2 < num_kv_blocks; tile += 2) {
                        issue_wgmma_tile(tile + 1, second, b0, b1);
                        wait_wgmma(std::integral_constant<int, 1>{});
                        consume_wgmma_tile(tile, accum, a0, a1);
                        issue_wgmma_tile(tile + 2, accum, a0, a1);
                        wait_wgmma(std::integral_constant<int, 1>{});
                        consume_wgmma_tile(tile + 1, second, b0, b1);
                    }
                    if (tile + 1 < num_kv_blocks) {
                        issue_wgmma_tile(tile + 1, second, b0, b1);
                        wait_wgmma(std::integral_constant<int, 1>{});
                        consume_wgmma_tile(tile, accum, a0, a1);
                        wait_wgmma(std::integral_constant<int, 0>{});
                        consume_wgmma_tile(tile + 1, second, b0, b1);
                    } else {
                        wait_wgmma(std::integral_constant<int, 0>{});
                        consume_wgmma_tile(tile, accum, a0, a1);
                    }
                }
            } else if constexpr (ITK_SHORT_MATH_SCHEDULE == 3) {
                float next_accum[WGMMA::kNumAccum];
                float scale0, scale1, next_scale0, next_scale1;
                if (num_kv_blocks > 0)
                    issue_wgmma_tile(0, accum, scale0, scale1);
                #pragma unroll
                for (uint32_t tile = 0; tile < num_kv_blocks; tile += 2) {
                    if (tile + 1 < num_kv_blocks) {
                        issue_wgmma_tile(tile + 1, next_accum, next_scale0, next_scale1);
                        wait_wgmma(std::integral_constant<int, 1>{});
                    } else {
                        wait_wgmma(std::integral_constant<int, 0>{});
                    }
                    consume_wgmma_tile(tile, accum, scale0, scale1);
                    if (tile + 1 < num_kv_blocks) {
                        if (tile + 2 < num_kv_blocks) {
                            issue_wgmma_tile(tile + 2, accum, scale0, scale1);
                            wait_wgmma(std::integral_constant<int, 1>{});
                        } else {
                            wait_wgmma(std::integral_constant<int, 0>{});
                        }
                        consume_wgmma_tile(tile + 1, next_accum, next_scale0, next_scale1);
                    }
                }
            } else {
                #pragma unroll
                for (uint32_t tile = 0; tile < num_kv_blocks; ++tile) {
                    float scale0, scale1;
                    issue_wgmma_tile(tile, accum, scale0, scale1);
                    wait_wgmma(std::integral_constant<int, 0>{});
                    consume_wgmma_tile(tile, accum, scale0, scale1);
                }
            }
            if constexpr (ITK_SHORT_DIAGNOSTIC) {
                if (thread_idx == 0) {
                    trace[8] = kv_wait_cycles;
                    trace[9] = issue_cycles;
                    trace[10] = wgmma_wait_cycles;
                    trace[11] = consume_cycles;
                    trace[12] = num_kv_blocks;
                }
            }

            // Publish counts before handing the two row buffers to TopK.
            if (lane_idx == 0) {
                #pragma unroll
                for (uint32_t i = 0; i < BLOCK_Q; ++ i) {
                    #pragma unroll
                    for (uint32_t slot = 0; slot < kWarpSegments; ++slot) {
                        const uint32_t segment_idx =
                            warp_idx + slot * kNumMathWarps;
                        segment_counts[i * kNumCandidateSegments + segment_idx] =
                            warp_candidate_counts[i][slot];
                    }
                }
            }
            // Each math thread publishes its own stores with an arrival.
            phase_stamp(trace, 3, thread_idx == 0);
            candidate_ready[candidate_slot].arrive();
            if constexpr (not kOverlap)
                candidate_free[candidate_slot].wait(candidate_phase);
            num_total_kv_blocks += num_kv_blocks;

            // Release Q empty
            empty_q_barriers[q_stage_idx]->arrive();

            // Jump to the next block
            CUTE_TIE(get_next_block_q_idx(), block_q_idx, q_iter_idx);
        }
    }
}

} // namespace fused_index_topk::kernel::short_context
