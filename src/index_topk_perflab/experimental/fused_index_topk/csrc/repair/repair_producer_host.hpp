#pragma once

#include <torch/extension.h>

#include <string>
#include <utility>

#include <jit/compiler.hpp>
#include <jit/device_runtime.hpp>
#include <jit/kernel_runtime.hpp>
#include <jit_kernels/heuristics/sm90.hpp>
#include <jit_kernels/impls/runtime_utils.hpp>

namespace index_topk_perflab::fused_r5i {

using deep_gemm::Compiler;
using deep_gemm::DGException;
using deep_gemm::KernelHandle;
using deep_gemm::KernelRuntime;
using deep_gemm::LaunchArgs;
using deep_gemm::LaunchConfigHandle;
using deep_gemm::LaunchRuntime;
using deep_gemm::SM90ArchSpec;
using deep_gemm::compiler;
using deep_gemm::device_runtime;
using deep_gemm::launch_kernel;
using deep_gemm::make_tma_2d_desc;

constexpr int kTopK = 2048;
constexpr int kCandidateCapacity = 14080;
constexpr int kRepairCandidateCapacity = 16384;
constexpr int kBlockQ = 2;
constexpr int kBlockKV = 256;
constexpr int kNumQStages = 3;
constexpr int kNumKVStages = 3;
constexpr int kNumSpecializedThreads = 128;
constexpr int kNumMathThreads = 512;
constexpr int kNumCandidateSegments = 16;
constexpr int kExpectedProducerSmemBytes = 152228;

template <bool kMaskedRepair>
class SM90FP8MQACandidateR5iRuntime final
    : public LaunchRuntime<SM90FP8MQACandidateR5iRuntime<kMaskedRepair>> {
public:
    struct Args {
        int seq_len;
        int seq_len_kv;
        int num_heads;
        int head_dim;
        int* cu_seq_len_k_start;
        int* cu_seq_len_k_end;
        float* sample_thresholds;
        uint64_t* candidate_pairs;
        int* segment_counts;
        uint8_t* repair_flags;
        CUtensorMap tensor_map_q;
        CUtensorMap tensor_map_kv;
        CUtensorMap tensor_map_kv_scales;
        CUtensorMap tensor_map_weights;
        LaunchArgs launch_args;
    };

    static inline std::string custom_kernel_header;
    static inline std::string custom_source_sha;

    static void configure(std::string header, std::string source_sha) {
        DG_HOST_ASSERT(not header.empty());
        DG_HOST_ASSERT(not source_sha.empty());
        if (not custom_kernel_header.empty()) {
            DG_HOST_ASSERT(custom_kernel_header == header);
            DG_HOST_ASSERT(custom_source_sha == source_sha);
            return;
        }
        custom_kernel_header = std::move(header);
        custom_source_sha = std::move(source_sha);
    }

    static std::string generate_impl(const Args& args) {
        DG_HOST_ASSERT(args.num_heads == 64);
        DG_HOST_ASSERT(args.head_dim == 128);
        DG_HOST_ASSERT(custom_kernel_header.find('"') == std::string::npos);

        constexpr int candidate_capacity =
            kMaskedRepair ? kRepairCandidateCapacity : kCandidateCapacity;
        return fmt::format(R"(
#include "{}"

using namespace index_topk_perflab::fused_r5i;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&itk_sm90_fp8_mqa_candidate_producer_r5i<
        {}, 64, 128, 2, 256, 3, 3, 128, 512, 2048, {}
    >);
}}
)",
            custom_kernel_header,
            kMaskedRepair ? "true" : "false",
            candidate_capacity);
    }

    static void launch_impl(
        const KernelHandle& kernel,
        const LaunchConfigHandle& config,
        Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(
            kernel,
            config,
            args.seq_len,
            args.seq_len_kv,
            args.cu_seq_len_k_start,
            args.cu_seq_len_k_end,
            args.sample_thresholds,
            args.candidate_pairs,
            args.segment_counts,
            args.repair_flags,
            args.tensor_map_q,
            args.tensor_map_kv,
            args.tensor_map_kv_scales,
            args.tensor_map_weights));
    }
};

inline void init_jit(
    const std::string& deep_gemm_package_root,
    const std::string& cuda_home,
    const std::string& custom_kernel_header,
    const std::string& custom_source_sha) {
    Compiler::prepare_init(deep_gemm_package_root, cuda_home);
    KernelRuntime::prepare_init(cuda_home);
    SM90FP8MQACandidateR5iRuntime<false>::configure(
        custom_kernel_header, custom_source_sha);
    SM90FP8MQACandidateR5iRuntime<true>::configure(
        custom_kernel_header, custom_source_sha);
}

template <bool kMaskedRepair>
inline void fp8_mqa_candidate_impl(
    const torch::Tensor& q,
    const torch::Tensor& kv,
    const torch::Tensor& kv_scales,
    const torch::Tensor& weights,
    const torch::Tensor& cu_seq_len_k_start,
    const torch::Tensor& cu_seq_len_k_end,
    const torch::Tensor* sample_thresholds,
    const torch::Tensor& candidate_pairs,
    const torch::Tensor& segment_counts,
    const torch::Tensor* repair_flags) {
    const auto& [seq_len, num_heads, head_dim] = deep_gemm::get_shape<3>(q);
    const auto& [seq_len_kv, kv_head_dim] = deep_gemm::get_shape<2>(kv);
    const auto& [weights_seq_len, weights_num_heads] =
        deep_gemm::get_shape<2>(weights);
    const auto& [kv_scales_len] = deep_gemm::get_shape<1>(kv_scales);

    DG_HOST_ASSERT(seq_len > 0 and seq_len % kBlockQ == 0);
    DG_HOST_ASSERT(seq_len_kv > 0);
    DG_HOST_ASSERT(num_heads == 64 and head_dim == 128);
    DG_HOST_ASSERT(kv_head_dim == head_dim and kv_scales_len == seq_len_kv);
    DG_HOST_ASSERT(weights_seq_len == seq_len and weights_num_heads == num_heads);
    DG_HOST_ASSERT(cu_seq_len_k_start.numel() == seq_len);
    DG_HOST_ASSERT(cu_seq_len_k_end.numel() == seq_len);
    if constexpr (kMaskedRepair) {
        DG_HOST_ASSERT(seq_len_kv == kRepairCandidateCapacity);
        DG_HOST_ASSERT(repair_flags != nullptr);
        DG_HOST_ASSERT(repair_flags->dim() == 1);
        DG_HOST_ASSERT(repair_flags->numel() == seq_len);
    } else {
        DG_HOST_ASSERT(sample_thresholds != nullptr);
        DG_HOST_ASSERT(sample_thresholds->dim() == 1);
        DG_HOST_ASSERT(sample_thresholds->numel() == seq_len);
    }
    DG_HOST_ASSERT(candidate_pairs.dim() == 2);
    DG_HOST_ASSERT(candidate_pairs.size(0) == seq_len);
    constexpr int candidate_capacity =
        kMaskedRepair ? kRepairCandidateCapacity : kCandidateCapacity;
    DG_HOST_ASSERT(candidate_pairs.size(1) == candidate_capacity);
    DG_HOST_ASSERT(segment_counts.dim() == 2);
    DG_HOST_ASSERT(segment_counts.size(0) == seq_len);
    DG_HOST_ASSERT(segment_counts.size(1) == kNumCandidateSegments);

    DG_HOST_ASSERT(q.is_cuda() and kv.is_cuda() and kv_scales.is_cuda());
    DG_HOST_ASSERT(weights.is_cuda() and cu_seq_len_k_start.is_cuda());
    DG_HOST_ASSERT(cu_seq_len_k_end.is_cuda());
    DG_HOST_ASSERT(candidate_pairs.is_cuda() and segment_counts.is_cuda());
    if constexpr (kMaskedRepair)
        DG_HOST_ASSERT(repair_flags->is_cuda());
    else
        DG_HOST_ASSERT(sample_thresholds->is_cuda());
    DG_HOST_ASSERT(q.is_contiguous() and kv.is_contiguous());
    DG_HOST_ASSERT(kv_scales.is_contiguous() and weights.is_contiguous());
    DG_HOST_ASSERT(cu_seq_len_k_start.is_contiguous());
    DG_HOST_ASSERT(cu_seq_len_k_end.is_contiguous());
    if constexpr (kMaskedRepair)
        DG_HOST_ASSERT(repair_flags->is_contiguous());
    else
        DG_HOST_ASSERT(sample_thresholds->is_contiguous());
    DG_HOST_ASSERT(candidate_pairs.is_contiguous());
    DG_HOST_ASSERT(segment_counts.is_contiguous());
    DG_HOST_ASSERT(q.get_device() == kv.get_device());
    DG_HOST_ASSERT(q.get_device() == kv_scales.get_device());
    DG_HOST_ASSERT(q.get_device() == weights.get_device());
    DG_HOST_ASSERT(q.get_device() == cu_seq_len_k_start.get_device());
    DG_HOST_ASSERT(q.get_device() == cu_seq_len_k_end.get_device());
    DG_HOST_ASSERT(q.get_device() == candidate_pairs.get_device());
    DG_HOST_ASSERT(q.get_device() == segment_counts.get_device());
    if constexpr (kMaskedRepair)
        DG_HOST_ASSERT(q.get_device() == repair_flags->get_device());
    else
        DG_HOST_ASSERT(q.get_device() == sample_thresholds->get_device());
    DG_HOST_ASSERT(q.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(kv.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(kv_scales.scalar_type() == torch::kFloat);
    DG_HOST_ASSERT(weights.scalar_type() == torch::kFloat);
    DG_HOST_ASSERT(cu_seq_len_k_start.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(cu_seq_len_k_end.scalar_type() == torch::kInt);
    if constexpr (kMaskedRepair)
        DG_HOST_ASSERT(repair_flags->scalar_type() == torch::kUInt8);
    else
        DG_HOST_ASSERT(sample_thresholds->scalar_type() == torch::kFloat);
    DG_HOST_ASSERT(candidate_pairs.scalar_type() == torch::kInt64);
    DG_HOST_ASSERT(segment_counts.scalar_type() == torch::kInt);

    const auto tensor_map_q = make_tma_2d_desc(
        q,
        head_dim,
        seq_len * num_heads,
        head_dim,
        kBlockQ * num_heads,
        head_dim,
        head_dim);
    const auto tensor_map_kv = make_tma_2d_desc(
        kv,
        head_dim,
        seq_len_kv,
        head_dim,
        kBlockKV,
        head_dim,
        head_dim);
    const auto tensor_map_kv_scales = make_tma_2d_desc(
        kv_scales,
        deep_gemm::get_tma_aligned_size(
            seq_len_kv,
            static_cast<int>(kv_scales.element_size())),
        1,
        kBlockKV,
        1,
        0,
        0);
    const auto tensor_map_weights = make_tma_2d_desc(
        weights,
        num_heads,
        seq_len,
        num_heads,
        kBlockQ,
        num_heads,
        0);

    // The producer preserves DeepGEMM's exact dynamic shared-memory layout.
    int smem_size = 0;
    smem_size += kNumQStages * kBlockQ * num_heads * head_dim * q.element_size();
    smem_size += kNumKVStages * kBlockKV * head_dim * kv.element_size();
    smem_size +=
        kNumQStages * kBlockQ * num_heads * weights.element_size();
    smem_size += kNumKVStages * kBlockKV * kv_scales.element_size();
    smem_size +=
        (kNumQStages * 2 + kNumKVStages * 2 + (kNumMathThreads / 128) * 2) * 8;
    smem_size += 4;
    DG_HOST_ASSERT(smem_size == kExpectedProducerSmemBytes);
    DG_HOST_ASSERT(smem_size <= SM90ArchSpec::smem_capacity);

    using Runtime = SM90FP8MQACandidateR5iRuntime<kMaskedRepair>;
    const typename Runtime::Args args = {
        .seq_len = seq_len,
        .seq_len_kv = seq_len_kv,
        .num_heads = num_heads,
        .head_dim = head_dim,
        .cu_seq_len_k_start = cu_seq_len_k_start.data_ptr<int>(),
        .cu_seq_len_k_end = cu_seq_len_k_end.data_ptr<int>(),
        .sample_thresholds = kMaskedRepair
            ? nullptr : sample_thresholds->data_ptr<float>(),
        .candidate_pairs = reinterpret_cast<uint64_t*>(
            candidate_pairs.data_ptr<int64_t>()),
        .segment_counts = segment_counts.data_ptr<int>(),
        .repair_flags = kMaskedRepair
            ? repair_flags->data_ptr<uint8_t>() : nullptr,
        .tensor_map_q = tensor_map_q,
        .tensor_map_kv = tensor_map_kv,
        .tensor_map_kv_scales = tensor_map_kv_scales,
        .tensor_map_weights = tensor_map_weights,
        .launch_args = LaunchArgs(
            kMaskedRepair
                ? (seq_len + kBlockQ - 1) / kBlockQ
                : device_runtime->get_num_sms(),
            kNumSpecializedThreads + kNumMathThreads,
            smem_size),
    };

    const auto code = Runtime::generate(args);
    const std::string build_name =
        std::string(kMaskedRepair
            ? "itk_sm90_fp8_mqa_masked_repair_producer_r5i_"
            : "itk_sm90_fp8_mqa_candidate_producer_r5i_") +
        Runtime::custom_source_sha.substr(0, 12);
    const auto runtime = compiler->build(build_name, code);
    Runtime::launch(runtime, args);
}

inline void fp8_mqa_candidate_r5i_out(
    const torch::Tensor& q,
    const torch::Tensor& kv,
    const torch::Tensor& kv_scales,
    const torch::Tensor& weights,
    const torch::Tensor& cu_seq_len_k_start,
    const torch::Tensor& cu_seq_len_k_end,
    const torch::Tensor& sample_thresholds,
    const torch::Tensor& candidate_pairs,
    const torch::Tensor& segment_counts) {
    fp8_mqa_candidate_impl<false>(
        q,
        kv,
        kv_scales,
        weights,
        cu_seq_len_k_start,
        cu_seq_len_k_end,
        &sample_thresholds,
        candidate_pairs,
        segment_counts,
        nullptr);
}

inline void fp8_mqa_candidate_repair_r5i_out(
    const torch::Tensor& q,
    const torch::Tensor& kv,
    const torch::Tensor& kv_scales,
    const torch::Tensor& weights,
    const torch::Tensor& cu_seq_len_k_start,
    const torch::Tensor& cu_seq_len_k_end,
    const torch::Tensor& repair_flags,
    const torch::Tensor& candidate_pairs,
    const torch::Tensor& segment_counts) {
    fp8_mqa_candidate_impl<true>(
        q,
        kv,
        kv_scales,
        weights,
        cu_seq_len_k_start,
        cu_seq_len_k_end,
        nullptr,
        candidate_pairs,
        segment_counts,
        &repair_flags);
}

}  // namespace index_topk_perflab::fused_r5i
