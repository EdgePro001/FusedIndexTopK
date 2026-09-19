#include <torch/extension.h>
#include <unordered_map>
#include <jit/compiler.hpp>
#include <jit/device_runtime.hpp>
#include <jit/kernel_runtime.hpp>
#include <jit_kernels/heuristics/sm90.hpp>
#include <jit_kernels/impls/runtime_utils.hpp>

namespace fused_index_topk::kernel::short_context {
using namespace deep_gemm;

// Must match cta_topk.cuh; checked independently by the device instantiation.
constexpr int kParallelSmemBytes = 225696;
constexpr int kCompactSmemBytes = 218048;
constexpr int kPackedSmemBytes = 228288;

class Runtime final : public LaunchRuntime<Runtime> {
public:
    struct Args {
        int seq_len, seq_len_kv, mode, tuning, math_schedule, math_registers;
        bool diagnostic, cache_weights;
        int64_t* trace;
        int *start, *end;
        float* thresholds;
        int* output;
        uint8_t* failures;
        CUtensorMap q_map, kv_map, scales_map, weights_map;
        LaunchArgs launch_args;
    };
    static inline std::string header;
    static inline std::string source_sha;

    static std::string generate_impl(const Args& args) {
        DG_HOST_ASSERT(header.find('"') == std::string::npos);
        return fmt::format(R"(
#define ITK_SHORT_TUNING {}
#define ITK_SHORT_DIAGNOSTIC {}
#define ITK_SHORT_MATH_SCHEDULE {}
#define ITK_SHORT_MATH_REGISTERS {}
#define ITK_SHORT_CACHE_WEIGHTS {}
#include "{}"
using namespace fused_index_topk::kernel::short_context;
static_assert(101632 + sizeof(SelectionScratch<7680>) == 225696);
static_assert(84608 + 2 * sizeof(SelectionScratch<4096>) == 218048);
static_assert(84608 + 2 * sizeof(SelectionScratch<5888>) == 228288);
static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&itk_fused_index_topk_short<
        64, 128, 2, 128, {}, 3, 128, 256, 2048, {}, {}, {}>);
}}
)", args.tuning, args.diagnostic ? 1 : 0, args.math_schedule, args.math_registers,
        args.cache_weights ? 1 : 0, header, args.mode == 0 ? 3 : 2,
        args.mode == 0 ? 7680 : (args.mode >= 3 ? 5888 : 4096),
        args.mode == 0 ? 1 : 2, (args.mode == 2 or args.mode == 4) ? "true" : "false");
    }
    static void launch_impl(const KernelHandle& kernel,
                            const LaunchConfigHandle& config, Args args) {
        // Validate the actual cubin, not only the assumed 168-register entry budget.
        // Cache attributes outside hot launches after the first use of each kernel.
        static thread_local std::unordered_map<KernelHandle, int> register_cache;
        auto found = register_cache.find(kernel);
        if (found == register_cache.end()) {
            int initial_registers = 0;
#if CUDART_VERSION >= 12080 and defined(DG_JIT_USE_RUNTIME_API)
            cudaFuncAttributes attributes{};
            DG_CUDA_RUNTIME_CHECK(cudaFuncGetAttributes(&attributes, kernel));
            initial_registers = attributes.numRegs;
#else
            DG_CUDA_DRIVER_CHECK(cuFuncGetAttribute(
                &initial_registers, CU_FUNC_ATTRIBUTE_NUM_REGS, kernel));
#endif
            found = register_cache.emplace(kernel, initial_registers).first;
        }
        const int initial_registers = found->second;
        TORCH_CHECK(initial_registers >= 64 and initial_registers <= args.math_registers,
                    "short-context fused TopK invalid initial register count for setmaxnreg inc/dec: ", initial_registers);
        TORCH_CHECK(256 * args.math_registers + 128 * 64 <= 384 * initial_registers,
                    "short-context fused TopK requested role registers exceed actual initial CTA pool: ",
                    256 * args.math_registers + 128 * 64, " > ", 384 * initial_registers);
        DG_CUDA_UNIFIED_CHECK(launch_kernel(
            kernel, config, args.seq_len, args.seq_len_kv,
            args.start, args.end, args.thresholds, args.output, args.failures, args.trace,
            args.q_map, args.kv_map, args.scales_map, args.weights_map));
    }
};

void init_jit(const std::string& package, const std::string& cuda_home,
              const std::string& header, const std::string& source_sha) {
    DG_HOST_ASSERT(not header.empty() and not source_sha.empty());
    if (not Runtime::header.empty()) {
        DG_HOST_ASSERT(Runtime::header == header and Runtime::source_sha == source_sha);
        return;
    }
    Compiler::prepare_init(package, cuda_home);
    KernelRuntime::prepare_init(cuda_home);
    Runtime::header = header;
    Runtime::source_sha = source_sha;
}

void fp8_mqa_topk_out(
        const torch::Tensor& q, const torch::Tensor& kv,
        const torch::Tensor& kv_scales, const torch::Tensor& weights,
        const torch::Tensor& start, const torch::Tensor& end,
        const torch::Tensor& thresholds, const torch::Tensor& output,
        const torch::Tensor& failures, int mode, int debug_ctas,
        int tuning, int math_schedule, int math_registers, bool cache_weights,
        bool diagnostic, const torch::Tensor& trace) {
    DG_HOST_ASSERT(mode >= 0 and mode <= 4);
    DG_HOST_ASSERT(tuning >= 0 and tuning <= 7);
    DG_HOST_ASSERT(math_schedule == 1 or math_schedule == 2);
    DG_HOST_ASSERT(math_registers >= 168 and math_registers <= 216 and math_registers % 8 == 0);
    DG_HOST_ASSERT(256 * math_registers + 128 * 64 <= 384 * 168);
    const int default_ctas = device_runtime->get_num_sms() * 2;
    DG_HOST_ASSERT(debug_ctas >= 0 and debug_ctas <= default_ctas);
    const int num_ctas = debug_ctas == 0 ? default_ctas : debug_ctas;
    const int smem_bytes = mode == 0 ? kParallelSmemBytes :
        (mode >= 3 ? kPackedSmemBytes : kCompactSmemBytes);
    const auto& [rows, heads, dim] = get_shape<3>(q);
    const auto& [columns, kv_dim] = get_shape<2>(kv);
    DG_HOST_ASSERT(rows > 0 and rows % 2 == 0);
    DG_HOST_ASSERT(trace.is_cuda() and trace.is_contiguous());
    DG_HOST_ASSERT(trace.get_device() == q.get_device());
    DG_HOST_ASSERT(trace.scalar_type() == torch::kLong);
    DG_HOST_ASSERT(trace.numel() == (diagnostic ? rows / 2 * 64 : 0));
    // This baseline deliberately retains the frozen N16k exact repair contract.
    DG_HOST_ASSERT(columns == 16384 and heads == 64 and dim == 128 and kv_dim == dim);
    DG_HOST_ASSERT(q.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(kv.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(kv_scales.dim() == 1 and kv_scales.numel() == columns);
    DG_HOST_ASSERT(kv_scales.scalar_type() == torch::kFloat);
    DG_HOST_ASSERT(weights.dim() == 2 and weights.size(0) == rows and weights.size(1) == heads);
    DG_HOST_ASSERT(weights.scalar_type() == torch::kFloat);
    for (const auto* tensor : {&start, &end}) {
        DG_HOST_ASSERT(tensor->dim() == 1 and tensor->numel() == rows);
        DG_HOST_ASSERT(tensor->scalar_type() == torch::kInt);
    }
    DG_HOST_ASSERT(thresholds.dim() == 1 and thresholds.numel() == rows);
    DG_HOST_ASSERT(thresholds.scalar_type() == torch::kFloat);
    DG_HOST_ASSERT(output.dim() == 2 and output.size(0) == rows and output.size(1) == 2048);
    DG_HOST_ASSERT(output.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(failures.dim() == 1 and failures.numel() == rows);
    DG_HOST_ASSERT(failures.scalar_type() == torch::kUInt8);
    for (const auto* tensor : {&q, &kv, &kv_scales, &weights, &start, &end,
                               &thresholds, &output, &failures}) {
        DG_HOST_ASSERT(tensor->is_cuda() and tensor->is_contiguous());
        DG_HOST_ASSERT(tensor->get_device() == q.get_device());
    }
    DG_HOST_ASSERT(smem_bytes <= SM90ArchSpec::smem_capacity);
    const Runtime::Args args = {
        .seq_len = rows,
        .seq_len_kv = columns,
        .mode = mode,
        .tuning = tuning,
        .math_schedule = math_schedule,
        .math_registers = math_registers,
        .diagnostic = diagnostic,
        .cache_weights = cache_weights,
        .trace = diagnostic ? trace.data_ptr<int64_t>() : nullptr,
        .start = start.data_ptr<int>(),
        .end = end.data_ptr<int>(),
        .thresholds = thresholds.data_ptr<float>(),
        .output = output.data_ptr<int>(),
        .failures = failures.data_ptr<uint8_t>(),
        .q_map = make_tma_2d_desc(q, dim, rows * heads, dim, 2 * heads, dim, dim),
        .kv_map = make_tma_2d_desc(kv, dim, columns, dim, 128, dim, dim),
        .scales_map = make_tma_2d_desc(
            kv_scales, get_tma_aligned_size(columns, 4), 1, 128, 1, 0, 0),
        .weights_map = make_tma_2d_desc(weights, heads, rows, heads, 2, heads, 0),
        .launch_args = LaunchArgs(num_ctas, 384, smem_bytes),
    };
    const auto code = Runtime::generate(args);
    const auto runtime = compiler->build(
        "itk_fused_index_topk_short_" + std::to_string(mode) + "_" +
            Runtime::source_sha.substr(0, 12), code);
    Runtime::launch(runtime, args);
}
} // namespace fused_index_topk::kernel::short_context

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.doc() = "Short-context GEMM + SMEM-resident exact TopK GEMM + SMEM-resident exact TopK";
    module.def("init_jit", &fused_index_topk::kernel::short_context::init_jit);
    module.def("fp8_mqa_topk_out", &fused_index_topk::kernel::short_context::fp8_mqa_topk_out);
}
