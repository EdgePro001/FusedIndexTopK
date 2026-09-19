# Methodology

## Qualified workload

- GPU: NVIDIA H20-3e, SM90, 78 SMs
- query tokens: 4096
- context tokens: 8192, 16384, 32768, 65536, 131072, 163840
- Top-K: 2048, exact and unordered
- indexer heads: 64
- head dimension: 128
- Q/KV: FP8 E4M3FN
- causal range: `[0, N - Q + q + 1)`

The software revisions are frozen in the configuration and
[THIRD_PARTY.md](../THIRD_PARTY.md).

## Real-corpus replay

The release campaign used native-length contextual model captures from held-out
real text, not synthetic random tensors and not truncated copies of one long
sample. The source set contained 1,032 documents and 24 prompts. Measurements
cover layers 0, 15, 30, 45, and 60; normal and hard splits; and two fixtures per
split.

Captured tensors are not redistributed. The public result includes SHA-256
identities for the token and capture manifests so authorized holders can verify
the replay set without exposing source data.

## Correctness

Each candidate output is compared with an independent
DeepGEMM-plus-`torch.topk` reference. The check validates exact Top-K
membership under the documented tie policy and also requires zero unresolved
repair rows.

DeepSelect is the performance baseline, not the correctness oracle.

## Timing

Each cell uses the same input on the same GPU for the candidate and baseline:

- 12 hot trials per implementation
- 4 operator calls per trial
- 3 CUPTI observations per case
- sampling and device repair included
- JIT compilation, input loading, and post-run correctness checks excluded

The primary table reports the mean of paired hot-latency changes. Pairing avoids
turning independent clock or load drift into a claimed operator improvement.

## Reproducing the public matrix

```bash
scripts/setup_h20.sh
make check CONFIG=configs/fused_index_topk_h20.json
make bench CONFIG=configs/fused_index_topk_h20.json
```

Set `CUDA_VISIBLE_DEVICES` to the intended H20 and `ITK_RUNTIME_ROOT` if the
default cache location is unsuitable. A different GPU, CUDA/PyTorch build,
upstream revision, shape, or corpus requires a new qualification result.
