#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

constexpr int kThresholdThreads = 256;
constexpr int kWarpSize = 32;
constexpr int kHeadDim = 128;
constexpr int kVectorBytes = 16;
constexpr int kVectorsPerKV = kHeadDim / kVectorBytes;
constexpr unsigned kFullMask = 0xffffffffu;

__device__ __forceinline__ float negative_infinity() {
  return -__int_as_float(0x7f800000);
}

__device__ __forceinline__ uint32_t ordered_float(float value) {
  uint32_t bits = __float_as_uint(value);
  if ((bits & 0x7fffffffu) == 0) bits = 0;
  return (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
}

__device__ __forceinline__ float ordered_to_float(uint32_t ordered) {
  const uint32_t bits =
      (ordered & 0x80000000u) ? (ordered ^ 0x80000000u) : ~ordered;
  return __uint_as_float(bits);
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

__global__ void gather_sampled_kv(
    const uint8_t* __restrict__ kv,
    const float* __restrict__ kv_scales,
    const int32_t* __restrict__ sample_ids,
    uint8_t* __restrict__ sampled_kv,
    float* __restrict__ sampled_scales,
    int samples) {
  const int work = blockIdx.x * blockDim.x + threadIdx.x;
  const int total_work = samples * kVectorsPerKV;
  if (work >= total_work) return;
  const int sample = work / kVectorsPerKV;
  const int vector = work - sample * kVectorsPerKV;
  const int source = sample_ids[sample];
  reinterpret_cast<uint4*>(sampled_kv)[work] =
      reinterpret_cast<const uint4*>(kv +
          static_cast<int64_t>(source) * kHeadDim)[vector];
  if (vector == 0) sampled_scales[sample] = kv_scales[source];
}

__global__ void sampled_threshold_radix(
    const float* __restrict__ sampled_scores,
    int64_t row_stride,
    const int32_t* __restrict__ sample_ids,
    const int32_t* __restrict__ k_start,
    const int32_t* __restrict__ k_end,
    float* __restrict__ thresholds,
    int rows,
    int samples,
    int sample_rank,
    int target_candidates) {
  const int row = blockIdx.x;
  if (row >= rows) return;
  const int tx = threadIdx.x;
  const int start = k_start[row];
  const int end = k_end[row];
  if (end - start <= target_candidates) {
    if (tx == 0) thresholds[row] = negative_infinity();
    return;
  }

  __shared__ int histogram[256];
  __shared__ int selected_byte;
  __shared__ int selected_greater;
  __shared__ int valid_samples;
  uint32_t threshold = 0;
  uint32_t prefix_mask = 0;
  int rank = sample_rank;

  #pragma unroll
  for (int round = 0; round < 4; ++round) {
    const int shift = 24 - round * 8;
    if (tx < 256) histogram[tx] = 0;
    __syncthreads();

    for (int sample = tx; sample < samples; sample += blockDim.x) {
      const int logical_id = sample_ids[sample];
      const float score = sampled_scores[
          static_cast<int64_t>(row) * row_stride + sample];
      const bool valid = start <= logical_id && logical_id < end &&
          isfinite(score);
      const uint32_t ordered = valid ? ordered_float(score) : 0u;
      const bool active = valid && (ordered & prefix_mask) == threshold;
      warp_histogram_add(
          histogram, active, static_cast<int>((ordered >> shift) & 0xffu));
    }
    __syncthreads();

    if (round == 0) {
      if (tx == 0) {
        int total = 0;
        #pragma unroll
        for (int bin = 0; bin < 256; ++bin) total += histogram[bin];
        valid_samples = total;
      }
      __syncthreads();
      if (valid_samples < sample_rank) {
        if (tx == 0) thresholds[row] = negative_infinity();
        return;
      }
    }

    if (tx < kWarpSize) {
      int byte = 0;
      int greater = 0;
      if (select_histogram_byte(histogram, rank, &byte, &greater)) {
        selected_byte = byte;
        selected_greater = greater;
      }
    }
    __syncthreads();
    rank -= selected_greater;
    threshold |= static_cast<uint32_t>(selected_byte) << shift;
    prefix_mask |= 0xffu << shift;
  }
  if (tx == 0) thresholds[row] = ordered_to_float(threshold);
}

void check_cuda_contiguous(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda() && tensor.is_contiguous(),
              name, " must be contiguous CUDA storage");
}

void gather_out(torch::Tensor kv,
                torch::Tensor kv_scales,
                torch::Tensor sample_ids,
                torch::Tensor sampled_kv,
                torch::Tensor sampled_scales) {
  check_cuda_contiguous(kv, "kv");
  check_cuda_contiguous(kv_scales, "kv_scales");
  check_cuda_contiguous(sample_ids, "sample_ids");
  check_cuda_contiguous(sampled_kv, "sampled_kv");
  check_cuda_contiguous(sampled_scales, "sampled_scales");
  TORCH_CHECK(kv.scalar_type() == at::kFloat8_e4m3fn &&
                  sampled_kv.scalar_type() == at::kFloat8_e4m3fn,
              "kv tensors must be FP8 E4M3FN");
  TORCH_CHECK(kv.dim() == 2 && kv.size(1) == kHeadDim,
              "kv must have shape [N, 128]");
  TORCH_CHECK(kv_scales.scalar_type() == at::kFloat &&
                  kv_scales.dim() == 1 && kv_scales.numel() == kv.size(0),
              "kv_scales must be float32 [N]");
  TORCH_CHECK(sample_ids.scalar_type() == at::kInt && sample_ids.dim() == 1,
              "sample_ids must be int32 [S]");
  const int64_t samples = sample_ids.numel();
  TORCH_CHECK(samples > 0 && samples <= INT32_MAX, "invalid sample count");
  TORCH_CHECK(sampled_kv.dim() == 2 && sampled_kv.size(0) == samples &&
                  sampled_kv.size(1) == kHeadDim,
              "sampled_kv must have shape [S, 128]");
  TORCH_CHECK(sampled_scales.scalar_type() == at::kFloat &&
                  sampled_scales.dim() == 1 &&
                  sampled_scales.numel() == samples,
              "sampled_scales must be float32 [S]");
  const int device = kv.get_device();
  TORCH_CHECK(kv_scales.get_device() == device &&
                  sample_ids.get_device() == device &&
                  sampled_kv.get_device() == device &&
                  sampled_scales.get_device() == device,
              "gather tensors must share one CUDA device");

  c10::cuda::CUDAGuard guard(kv.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(device).stream();
  const int work = static_cast<int>(samples) * kVectorsPerKV;
  gather_sampled_kv<<<(work + 255) / 256, 256, 0, stream>>>(
      reinterpret_cast<const uint8_t*>(kv.data_ptr()),
      kv_scales.data_ptr<float>(),
      sample_ids.data_ptr<int32_t>(),
      reinterpret_cast<uint8_t*>(sampled_kv.data_ptr()),
      sampled_scales.data_ptr<float>(),
      static_cast<int>(samples));
  const cudaError_t status = cudaGetLastError();
  TORCH_CHECK(status == cudaSuccess,
              "sampled KV gather launch failed: ", cudaGetErrorString(status));
}

void threshold_out(torch::Tensor sampled_scores,
                   torch::Tensor sample_ids,
                   torch::Tensor k_start,
                   torch::Tensor k_end,
                   torch::Tensor thresholds,
                   int64_t sample_rank,
                   int64_t target_candidates) {
  check_cuda_contiguous(sample_ids, "sample_ids");
  check_cuda_contiguous(k_start, "k_start");
  check_cuda_contiguous(k_end, "k_end");
  check_cuda_contiguous(thresholds, "thresholds");
  TORCH_CHECK(sampled_scores.is_cuda() && sampled_scores.dim() == 2 &&
                  sampled_scores.stride(1) == 1 &&
                  sampled_scores.scalar_type() == at::kFloat,
              "sampled_scores must be CUDA float32 [Q, S] with unit inner stride");
  const int64_t rows = sampled_scores.size(0);
  const int64_t samples = sampled_scores.size(1);
  TORCH_CHECK(rows > 0 && rows <= INT32_MAX && samples > 0 &&
                  samples <= INT32_MAX,
              "invalid sampled score shape");
  TORCH_CHECK(sample_ids.scalar_type() == at::kInt &&
                  sample_ids.dim() == 1 && sample_ids.numel() == samples,
              "sample_ids must be int32 [S]");
  TORCH_CHECK(k_start.scalar_type() == at::kInt && k_end.scalar_type() == at::kInt &&
                  k_start.dim() == 1 && k_end.dim() == 1 &&
                  k_start.numel() == rows && k_end.numel() == rows,
              "causal ranges must be int32 [Q]");
  TORCH_CHECK(thresholds.scalar_type() == at::kFloat && thresholds.dim() == 1 &&
                  thresholds.numel() == rows,
              "thresholds must be float32 [Q]");
  TORCH_CHECK(sample_rank > 0 && sample_rank <= samples,
              "sample_rank must lie in [1, S]");
  TORCH_CHECK(target_candidates > 0 && target_candidates <= INT32_MAX,
              "invalid target candidate count");
  const int device = sampled_scores.get_device();
  TORCH_CHECK(sample_ids.get_device() == device && k_start.get_device() == device &&
                  k_end.get_device() == device && thresholds.get_device() == device,
              "threshold tensors must share one CUDA device");

  c10::cuda::CUDAGuard guard(sampled_scores.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(device).stream();
  sampled_threshold_radix<<<
      static_cast<unsigned>(rows), kThresholdThreads, 0, stream>>>(
          sampled_scores.data_ptr<float>(),
          sampled_scores.stride(0),
          sample_ids.data_ptr<int32_t>(),
          k_start.data_ptr<int32_t>(),
          k_end.data_ptr<int32_t>(),
          thresholds.data_ptr<float>(),
          static_cast<int>(rows),
          static_cast<int>(samples),
          static_cast<int>(sample_rank),
          static_cast<int>(target_candidates));
  const cudaError_t status = cudaGetLastError();
  TORCH_CHECK(status == cudaSuccess,
              "sample threshold launch failed: ", cudaGetErrorString(status));
}

pybind11::dict resource_report() {
  cudaFuncAttributes threshold_attributes{};
  cudaFuncAttributes gather_attributes{};
  TORCH_CHECK(cudaFuncGetAttributes(
                  &threshold_attributes, sampled_threshold_radix) == cudaSuccess,
              "cannot query threshold resources");
  TORCH_CHECK(cudaFuncGetAttributes(
                  &gather_attributes, gather_sampled_kv) == cudaSuccess,
              "cannot query gather resources");
  pybind11::dict result;
  result["threshold_threads"] = kThresholdThreads;
  result["threshold_registers_per_thread"] = threshold_attributes.numRegs;
  result["threshold_static_shared_bytes"] = threshold_attributes.sharedSizeBytes;
  result["gather_registers_per_thread"] = gather_attributes.numRegs;
  return result;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("gather_out", &gather_out, "Gather one common random KV sample");
  module.def("threshold_out", &threshold_out, "Estimate per-row sample thresholds");
  module.def("resource_report", &resource_report, "Report sampling kernel resources");
}
