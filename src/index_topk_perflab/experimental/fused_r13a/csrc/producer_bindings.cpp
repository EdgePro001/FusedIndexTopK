#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "smxx_fp8_mqa_candidate_r13a.hpp"

#ifndef TORCH_EXTENSION_NAME
#define TORCH_EXTENSION_NAME itk_fused_r13a_producer_host
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.doc() = "Host/JIT wrapper for the fused R13a compact-candidate producer";
    module.def(
        "init_jit",
        &index_topk_perflab::fused_r13a::init_jit,
        pybind11::arg("deep_gemm_package_root"),
        pybind11::arg("cuda_home"),
        pybind11::arg("custom_kernel_header"),
        pybind11::arg("custom_source_sha"));
    module.def(
        "fp8_mqa_candidate_r13a_out",
        &index_topk_perflab::fused_r13a::fp8_mqa_candidate_r13a_out,
        pybind11::arg("q"),
        pybind11::arg("kv"),
        pybind11::arg("kv_scales"),
        pybind11::arg("weights"),
        pybind11::arg("cu_seq_len_k_start"),
        pybind11::arg("cu_seq_len_k_end"),
        pybind11::arg("sample_thresholds"),
        pybind11::arg("candidate_pairs"),
        pybind11::arg("segment_counts"));
    module.def(
        "fp8_mqa_candidate_r13b_out",
        &index_topk_perflab::fused_r13a::fp8_mqa_candidate_r13b_out,
        pybind11::arg("q"),
        pybind11::arg("kv"),
        pybind11::arg("kv_scales"),
        pybind11::arg("weights"),
        pybind11::arg("cu_seq_len_k_start"),
        pybind11::arg("cu_seq_len_k_end"),
        pybind11::arg("sample_thresholds"),
        pybind11::arg("candidate_pairs"),
        pybind11::arg("segment_counts"));
    module.def(
        "fp8_mqa_candidate_r13d_out",
        &index_topk_perflab::fused_r13a::fp8_mqa_candidate_r13d_out,
        pybind11::arg("q"),
        pybind11::arg("kv"),
        pybind11::arg("kv_scales"),
        pybind11::arg("weights"),
        pybind11::arg("cu_seq_len_k_start"),
        pybind11::arg("cu_seq_len_k_end"),
        pybind11::arg("sample_thresholds"),
        pybind11::arg("candidate_pairs"),
        pybind11::arg("segment_counts"));
    module.def(
        "fp8_mqa_candidate_r15a_out",
        &index_topk_perflab::fused_r13a::fp8_mqa_candidate_r15a_out,
        pybind11::arg("q"),
        pybind11::arg("kv"),
        pybind11::arg("kv_scales"),
        pybind11::arg("weights"),
        pybind11::arg("cu_seq_len_k_start"),
        pybind11::arg("cu_seq_len_k_end"),
        pybind11::arg("sample_thresholds"),
        pybind11::arg("candidate_pairs"),
        pybind11::arg("segment_counts"));
    module.def(
        "fp8_mqa_candidate_repair_r13a_out",
        &index_topk_perflab::fused_r13a::fp8_mqa_candidate_repair_r13a_out,
        pybind11::arg("q"),
        pybind11::arg("kv"),
        pybind11::arg("kv_scales"),
        pybind11::arg("weights"),
        pybind11::arg("cu_seq_len_k_start"),
        pybind11::arg("cu_seq_len_k_end"),
        pybind11::arg("repair_flags"),
        pybind11::arg("candidate_pairs"),
        pybind11::arg("segment_counts"));
}
