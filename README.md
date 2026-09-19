# FusedIndexTopK

FusedIndexTopK is an exact, unordered Top-K operator for the DeepSeek V3.2
indexer workload on NVIDIA Hopper GPUs. It keeps score production and the
normal Top-K path in one kernel, so a dense `Q × N` score matrix is never
materialized.

The current release targets:

- FP8 MQA indexer inputs compatible with the pinned DeepGEMM revision
- `Q = 4096`, `K = 2048`, and `8192 <= N <= 163840`
- SM90/H20
- exact Top-K membership, including ties at the selection threshold

## What is fused

The main kernel combines tensor-core score production, threshold filtering,
candidate collection, radix selection, and final result emission. Candidates
remain in CTA shared memory. Long-context execution has a small bounded spill
workspace; the consumer merges that spill while later math work is still in
flight. A separate sampled-GEMM prepass estimates the threshold, and a
device-masked exact repair path handles the uncommon rows that fail the fast
path.

The normal path therefore has zero dense-score traffic and no full candidate
list in global memory. Long-context overflow may use only the bounded spill
described in [Architecture](docs/ARCHITECTURE.md) and
[Repair analysis](docs/REPAIR_ANALYSIS.md).

![Execution-path comparison for DeepGEMM plus FlashInfer, DeepGEMM plus DeepSelect, and FusedIndexTopK](docs/assets/execution-paths.svg)

| Property | DeepGEMM + FlashInfer | DeepGEMM + DeepSelect | FusedIndexTopK |
|---|---|---|---|
| Score production | DeepGEMM kernel | DeepGEMM kernel | fused main kernel |
| Dense `Q × N` scores | written to GMEM | written to GMEM | not materialized |
| Exact Top-K | separate kernel | separate kernel | same CTA consumer |
| Normal candidate list in GMEM | n/a | n/a | none; bounded overflow spill only |
| Exceptional rows | handled by Top-K kernel | handled by Top-K kernel | device-masked exact repair |

## H20 results

The published campaign used 120 native-length, held-out real-corpus cases:
five model layers, two difficulty splits, two fixtures per split, and six
context lengths. Every case matched an independent exact reference.

| N | DeepGEMM + DeepSelect | FusedIndexTopK | Paired change | Wins |
|---:|---:|---:|---:|---:|
| 8K | 1.870 ms | 1.949 ms | +4.26% | 0/20 |
| 16K | 4.255 ms | 3.799 ms | -10.72% | 20/20 |
| 32K | 8.689 ms | 7.881 ms | -9.30% | 20/20 |
| 64K | 17.174 ms | 16.084 ms | -6.34% | 20/20 |
| 128K | 33.848 ms | 32.470 ms | -4.07% | 20/20 |
| 160K | 41.999 ms | 40.531 ms | -3.49% | 20/20 |

![H20 operator latency comparison against DeepSelect, with separately qualified historical FlashInfer campaigns](docs/assets/h20-baseline-comparison.svg)

Lower is better. The 16K–160K range won all 100 measured cases. The 8K path is
supported and exact, but is not faster than the baseline; a production
dispatcher should retain DeepSelect at 8K unless that path is retuned.

Two independently qualified baseline tracks are available:

| Baseline track | Workload and implementation generation | Evidence |
|---|---|---|
| DeepGEMM + DeepSelect | current 2.0.0; real corpus, 8K–160K | 100/100 wins at 16K–160K; 8K is 4.26% slower |
| DeepGEMM + FlashInfer | previous short-context release; real corpus, 6K–16K | 40/40 wins; median CUPTI reduction 14.71% |
| DeepGEMM + FlashInfer | previous long-context development run; layer 0, 32K–128K | six of six cells faster by 4.87%–7.77% |

The FlashInfer rows are historical qualification results from an earlier fused
implementation, not a cross-run estimate for 2.0.0. They are shown to document
both baseline families without pretending that measurements from different
campaigns are directly interchangeable. A fresh three-way campaign is required
for a current head-to-head ranking. The compact provenance record is
[fused-index-topk-baseline-comparison-h20.json](results/fused-index-topk-baseline-comparison-h20.json).

These are operator measurements, not end-to-end TTFT, TPOT, or TPS claims.
Full measurements and provenance are in
[the release artifact](results/fused-index-topk-real-corpus-h20-v2.json) and
[Results](docs/RESULTS.md).

## Repository layout

```text
src/fused_index_topk/
  kernel/                     fused operator and CUDA sources
  variants/deepgemm_deepselect/
                              frozen baseline adapter
  api.py, config.py, ...      benchmark and validation framework
configs/
  fused_index_topk_h20.json   qualified H20 matrix
scripts/
  setup_h20.sh                reproducible environment setup
  evaluate_h20.sh             correctness and timing campaign
  benchmark_real_replay.py    real-corpus replay entry point
docs/                         design, method, and results
results/                      small, reviewable release summaries
```

The repository contains release code and compact evidence only. It does not
include model weights, captured tensors, build caches, profiler reports, or
private experiment history.

## Installation

The qualified environment uses CUDA 13.0, PyTorch 2.10.0+cu130, H20, and the
frozen DeepGEMM and DeepSelect revisions listed in
[THIRD_PARTY.md](THIRD_PARTY.md).

```bash
git clone https://github.com/EdgePro001/FusedIndexTopK.git
cd FusedIndexTopK
scripts/setup_h20.sh
```

By default, dependencies and JIT artifacts are stored under
`${XDG_CACHE_HOME:-$HOME/.cache}/fused-index-topk`. Set
`ITK_RUNTIME_ROOT` to choose another location.

Run the public qualification matrix:

```bash
make check
make bench
```

List registered implementations:

```bash
fused-index-topk list
```

Programmatic loading:

```python
from fused_index_topk.registry import load_variant

operator = load_variant("fused_index_topk")
```

## Reproducibility

The harness records source hashes, upstream commit IDs, device properties,
variant options, input seeds, timing protocol, and exactness checks. Real-corpus
tensors are not redistributed; their manifests are identified by SHA-256 so an
authorized holder can verify that the same capture set was replayed.

Before interpreting performance, read [Methodology](docs/METHODOLOGY.md).
Results outside the qualified hardware, software, and workload envelope require
new validation.

## License

FusedIndexTopK is released under the MIT License. Upstream dependencies retain
their own licenses; see [THIRD_PARTY.md](THIRD_PARTY.md) and [NOTICE](NOTICE).
