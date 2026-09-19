# Exact repair

Sampling supplies a threshold estimate; exactness never depends on that estimate
being perfect. The main kernel records a device flag whenever a row cannot be
proved complete within its bounded on-chip resources.

## Fast overflow handling

Long-context kernels first keep candidates in shared memory. Small overflow is
preserved in an `8 × 256` packed-pair workspace per row and merged by the
Top-K consumer. This avoids rescanning the key sequence and repeating the GEMM
for ordinary threshold variance.

The workspace is bounded. If it is insufficient, or if the candidate set is
otherwise unsafe, the row is flagged instead of returning an approximation.

## Full repair

The exact repair graph processes only flagged rows:

1. device code derives the per-chunk causal ranges;
2. a DeepGEMM-compatible producer recomputes 16K-key chunks;
3. each chunk emits exact local Top-K score/index pairs;
4. pairs are merged incrementally on device; and
5. final indices overwrite the flagged output rows.

The host does not copy failure flags or choose a repair branch. Unflagged rows
remain masked, so the graph is capturable and does no score work for them.

## Release evidence

Across 120 real-corpus cases, the fast path flagged:

- 73 rows at 16K;
- 3 rows at 128K;
- zero rows at 8K, 32K, 64K, and 160K.

All flagged rows were repaired, and all 120 final outputs were exact.

The default sampling guard remains two standard deviations. More aggressive
threshold calibration can reduce candidate pressure, but development sweeps did
not show a stable latency gain and increased underflow risk. The conservative
guard plus bounded spill is therefore the release setting.

Repair cost is input-dependent. The release data demonstrates low observed
incidence on the measured corpus; it is not a worst-case latency guarantee.
