# Results

## Release result

FusedIndexTopK 2.0.0 passed all 120 exactness checks in the real-corpus H20
campaign. Against the pinned DeepGEMM + DeepSelect baseline, it won every
measured case from 16K through 160K context.

| Context | Cases | Baseline mean | Fused mean | Paired mean change | Wins |
|---:|---:|---:|---:|---:|---:|
| 8,192 | 20 | 1.870 ms | 1.949 ms | +4.26% | 0 |
| 16,384 | 20 | 4.255 ms | 3.799 ms | -10.72% | 20 |
| 32,768 | 20 | 8.689 ms | 7.881 ms | -9.30% | 20 |
| 65,536 | 20 | 17.174 ms | 16.084 ms | -6.34% | 20 |
| 131,072 | 20 | 33.848 ms | 32.470 ms | -4.07% | 20 |
| 163,840 | 20 | 41.999 ms | 40.531 ms | -3.49% | 20 |

A negative change means FusedIndexTopK is faster. The campaign includes
sampling and device repair in both timing and correctness.

## Stability across layers

At every context length from 16K to 160K, all four inputs at each sampled layer
won. The weakest winning layer/context cell was still faster than the paired
baseline. At 8K, all layer cells were slower, which establishes a clear
dispatch boundary rather than a universal speed claim.

## Repair observations

The fast path flagged 73 rows at 16K and 3 rows at 128K across the complete
campaign. All other context lengths had zero flagged rows. Every flagged row
was repaired exactly and the final unresolved-repair count was zero.

A fast-path flag is not an incorrect output. It means the conservative device
guard selected the exact repair path for that row.

## Claim boundary

The reported values are paired operator latencies on NVIDIA H20-3e. They are
not whole-model measurements and do not establish TTFT, TPOT, TPS, energy, or
multi-GPU scaling. Those claims require an end-to-end serving experiment.

Machine-readable data and hashes are in
[fused-index-topk-real-corpus-h20-v2.json](../results/fused-index-topk-real-corpus-h20-v2.json).
