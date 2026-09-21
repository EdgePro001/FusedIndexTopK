# Results

## Release result

FusedIndexTopK 2.1.0 passed all 120 exactness checks in the real-corpus H20
campaign. Against the pinned DeepGEMM + DeepSelect baseline, it won 116 of 120
paired cases, including every case from 16K through 128K context.

| Context | Cases | Baseline mean | Fused mean | Paired mean change | Wins |
|---:|---:|---:|---:|---:|---:|
| 8,192 | 20 | 1.887 ms | 1.881 ms | -0.33% | 17 |
| 16,384 | 20 | 4.280 ms | 3.883 ms | -9.27% | 20 |
| 32,768 | 20 | 8.721 ms | 7.893 ms | -9.49% | 20 |
| 65,536 | 20 | 17.170 ms | 16.087 ms | -6.31% | 20 |
| 131,072 | 20 | 33.831 ms | 32.540 ms | -3.82% | 20 |
| 163,840 | 20 | 41.979 ms | 40.591 ms | -3.31% | 19 |

A negative change means FusedIndexTopK is faster. The campaign includes
sampling and device repair in both timing and correctness.

## Stability across layers

All 80 cases from 16K through 128K won. At 160K, 19 of 20 cases won; the single
loss was +0.56%. At 8K, 17 of 20 cases won, but the mean improvement was only
0.33% and the per-case range was -0.89% to +0.89%. This establishes strong
mid- and long-context gains while treating 8K honestly as near parity.

## Exactness and repair

Sampling, bounded overflow handling, and device-side exact repair are included
in the timed operator path. Every final output matched the independent exact
reference. The compact public summary intentionally reports final correctness
rather than internal fast-path flag counts.

## Claim boundary

The reported values are paired operator latencies on NVIDIA H20-3e. They are
not whole-model measurements and do not establish TTFT, TPOT, TPS, energy, or
multi-GPU scaling. Those claims require an end-to-end serving experiment.

Machine-readable data and hashes are in
[fused-index-topk-real-corpus-h20-v2.1.json](../results/fused-index-topk-real-corpus-h20-v2.1.json).
