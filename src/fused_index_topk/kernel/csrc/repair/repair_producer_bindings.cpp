#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "repair_producer_host.hpp"

#ifndef TORCH_EXTENSION_NAME
#define TORCH_EXTENSION_NAME fused_index_topk_repair_producer_host
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.doc() = "Host/JIT wrapper for FusedIndexTopK exact repair";
    module.def(
        "init_jit",
        &fused_index_topk::kernel::repair::init_jit,
        pybind11::arg("deep_gemm_package_root"),
        pybind11::arg("cuda_home"),
        pybind11::arg("custom_kernel_header"),
        pybind11::arg("custom_source_sha"));
    module.def(
        "produce_repair_candidates_out",
        &fused_index_topk::kernel::repair::fp8_mqa_candidate_repair_out,
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
