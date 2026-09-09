# Results

## R13a gold qualification

R13a won all 40 held-out cells against DeepGEMM + FlashInfer under both CUPTI
kernel-sum timing and whole-pipeline CUDA Events.

| Aggregate metric | CUPTI | CUDA Event |
|---|---:|---:|
| Winning cells | 40/40 | 40/40 |
| Median latency reduction vs FlashInfer | 14.713% | 14.867% |
| Bootstrap 95% interval | 13.185%–15.943% | 13.292%–16.092% |
| Median reduction vs PyTorch | 30.133% | 30.586% |

Across individual cells, CUPTI reductions versus FlashInfer ranged from 11.592%
to 19.366%. The largest observed range among the three independent block
medians was 1.230%, below the predeclared 5% stability gate.

The full per-cell table and source fingerprints are in
[`results/fused-r13a-release-qualification-h20-v1.json`](../results/fused-r13a-release-qualification-h20-v1.json)
and its compact [CSV](../results/fused-r13a-release-qualification-h20-v1.csv).

## R16a long-context development qualification

| N | Split | FlashInfer CUPTI ms | R16a CUPTI ms | Reduction |
|---:|---|---:|---:|---:|
| 32,768 | normal | 8.630 | 7.978 | 7.55% |
| 32,768 | hard | 8.620 | 7.950 | 7.77% |
| 65,536 | normal | 17.153 | 16.176 | 5.70% |
| 65,536 | hard | 17.125 | 16.173 | 5.56% |
| 131,072 | normal | 34.023 | 32.368 | 4.87% |
| 131,072 | hard | 34.054 | 32.125 | 5.66% |

All listed measurements include the empty device-masked repair launches. Forced
all-row repair was exact at 32K, 64K, 128K, 160K, and a 40K partial-tail case;
memcheck reported zero errors and racecheck reported zero hazards after the
recorded synchronization fix.

This is not yet equivalent to R13a's five-layer, three-block gold campaign. The
long-context numbers cover layer 0 and two held-out splits, so the status remains
development qualification.

Machine-readable evidence is in
[`results/fused-r16a-long-context-exact-h20-v1.json`](../results/fused-r16a-long-context-exact-h20-v1.json).

The independently assembled clean public snapshot was also exercised on H20:
all 130 tests passed, an N=16K short-path exactness smoke passed, and an N=32K
forced one-row underflow was repaired across two chunks with zero mismatches or
unresolved flags. The compact record is
[`results/public-release-smoke-h20-v1.json`](../results/public-release-smoke-h20-v1.json).

## Claim boundary

The qualified claim is specific to the published shape, software stack, timing
protocol, and replay distribution. It does not imply that R13a/R16a wins on all
Top-K shapes, GPUs, score distributions, or worst-case repair patterns.
