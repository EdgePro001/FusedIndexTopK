# Measurement and correctness methodology

## Primary and guardrail timing

Every published run records two clocks:

- `formal_kernel_sum`: the sum of CUDA kernel activity durations reported by
  Kineto/CUPTI inside the operator range. This is the primary optimization
  metric and excludes CPU launch API time and inter-kernel gaps.
- `cuda_event_total`: CUDA Events around the full device pipeline. This includes
  same-stream gaps and acts as an independent guardrail.

Compilation, allocation, replay loading, correctness checks, and input hashing
run outside the timed range. An 8,000,000,000-byte device buffer is written
before every trial to evict L2-resident working data; the flush itself is outside
both timing ranges.

The Kineto topology gate requires every active trial for a shape to expose the
same ordered activity sequence and requires every declared stage to be fully
covered. This prevents accidental inclusion of the flush or omission of a
candidate kernel.

## FusedIndexTopK release campaign

- GPU: NVIDIA H20-3e, SM90, 78 SMs;
- Q=4096, K=2048, H=64, D=128;
- N in {6144, 8192, 12288, 16384};
- layers {0, 15, 30, 45, 60};
- held-out `test_normal` and `test_hard` replay splits;
- A/B fixture bytes matched within each comparison cell;
- 10 warmups, 20 Event trials, and 30 CUPTI trials per process;
- three independent balanced-order blocks: FlashInfer/FusedIndexTopK/Torch,
  FusedIndexTopK/Torch/FlashInfer, Torch/FlashInfer/FusedIndexTopK.

The aggregate uses the median of block medians for each of 40 cells. Confidence
intervals are a deterministic nonparametric bootstrap across cells. They do not
treat repeated trials in one process as independent experiments.

## Correctness

`torch.topk` over DeepGEMM scores is the independent semantic reference.
FlashInfer is only the performance baseline and cannot certify itself.

Before and after benchmark trials, the harness checks:

- exact score multiset at the Kth threshold;
- uniqueness, causal validity, dtype, shape, contiguity, and padding;
- input SHA-256 and tensor version counters;
- source, runtime, protocol, plan, and input-content fingerprints.

Fault-injection cases force all rows through underflow repair, working-capacity
overflow repair, and all-equal-score repair. The long-context path additionally tests partial
tail chunks and CUDA memcheck/racecheck.

## Replay data policy

The reported replay tensors were generated from DeepSeek-V3.2-style indexer
workloads and frozen before evaluation. Tuning fixtures are excluded from the
release campaign. Model weights and tensor payloads are not redistributed by
this repository; only aggregate evidence and hashes are public.

This means another user can reproduce the code and protocol, but not the exact
input bytes without independently generating a compatible replay corpus.

The public packaging was flattened after qualification so only one operator is
exposed. The measured sampling, producer, reducer, and repair CUDA sources were
verified byte-for-byte against the qualified sources; only module layout,
Python dispatch, and public binding names changed.
