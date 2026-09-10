# Repair trigger, tail, and break-even analysis

## Natural trigger evidence

For FusedIndexTopK at 16K, a 64-seed probe across five layers and A/B hard
fixtures evaluated 640 invocations and 2,621,440 rows:

- 5 invocations triggered repair: 0.78125%, Wilson 95% interval 0.334%–1.816%;
- 9 rows failed the fast path: 0.000343% of rows;
- the largest failure cluster was 3 rows;
- every observed failure was candidate underflow.

Existing 32-seed long-context probes found one triggered invocation out of 128
at each of 32K, 64K, and 128K. Maximum clusters were 1, 1, and 9 rows.

## Forced repair break-even

The experiment injects a constant-work device mask between the fast reducer and
repair stage. The same injection kernel executes for every forced-row level,
including zero, so increments are measured against an instrumented zero-row
control without modifying operator kernel sources.

| N | Fast P50 ms | First-repair increment ms | Last clear winning forced rows | First clear losing forced rows |
|---:|---:|---:|---:|---:|
| 16K | 3.903 | 0.126 | 256 | 512 |
| 32K | 7.953 | 0.296 | 128 | 256 |
| 64K | 16.026 | 0.641 | 128 | 256 |
| 128K | 32.122 | 1.333 | 64 | 256 |

At 128K, 128 forced rows were on the P50 measurement boundary: 1.611 microseconds
above the lower bracketing FlashInfer median and 5.741 microseconds below the
upper median. It is therefore not classified as a clear win or loss.

Repair cost is stepwise rather than linear per row. One failed row activates a
persistent repair grid; additional cost appears when the number of failed rows
requires more execution waves. The observed natural clusters remain far below
the measured crossovers.

## What this proves

- Device repair preserves exactness in the tested underflow/overflow paths.
- On the held-out replay distribution, observed repair frequency and cluster
  size are too small to consume the measured fast-path advantage.
- It does not prove worst-case performance superiority. All-row repair was
  approximately 1.97–2.35× slower than the bracketed FlashInfer baseline across
  16K–128K.

The mixed-seed natural P95 is inferred by joining trigger counts with the forced
cost curve; it was not measured as one unified 640-invocation CUPTI campaign.
The forced sweep is diagnostic, uses one layer per context, and brackets temporal
drift with FlashInfer runs before and after the sweep.

Full compact data are in
[`results/repair-trigger-tail-break-even-h20-v1.json`](../results/repair-trigger-tail-break-even-h20-v1.json).
