# v0.3.0: Q1 release

Package version: 0.3.0. Kernel implementation version: 2.3.0.
Base: `6485fbb1a682e42dafffc65a5d0e3c838e216d59`.
Q1 implementation: `c7bdb902bbf2a404a069188547fae159c4b7af09`.

## Frozen scope

- Unified long kernel for N=8192..163840, N divisible by 128.
- Default `block_q=1`; explicit `block_q=2` remains an ablation.
- Single-request causal FP8 inputs, 64 indexer heads, head dimension 128,
  exact unordered Top-2048, positive even Q <= N.
- Existing sampling schedule and +2-sigma guard are unchanged.
- Existing on-chip candidate layout, bounded overflow spill, and device-masked
  exact repair are unchanged. The repair launch chain is not graph-conditional.
- No native batching, expanded N range, sample elision, sample-size reduction,
  or sampler/main-stream overlap is included.
- Diagnostic SM/timestamp recording is compiled out in the default build.

Direct extension callers must pass `block_q` immediately after `mode` in
`fp8_mqa_topk_out`. The public Python variant supplies it automatically.
Serving adapters must use the unified long loader and the new argument list;
loading this package alone does not update an external adapter.

## Existing H20 qualification

420 paired real-buffer replay points cover layers 0/15/30/45/60, normal/hard
inputs, N=8192/16384/32768/65536/131072/163840, and sampled Q values from
8 through 4096. Shorter Q is a contiguous suffix of a frozen Q4096 capture,
not an independent model execution. Full-path timings include sampling and
device-masked repair. Q1/Q2 alternate on identical inputs; the broad sweep
uses CUDA Event medians of three paired groups. Transition-region follow-up
uses fifteen groups.

| Q | Points | Q2/Q1 full-path geometric mean | Range |
|---:|---:|---:|---:|
| 8 | 60 | 1.770x | 1.521–1.865x |
| 64 | 60 | 1.769x | 1.547–1.859x |
| 512 | 60 | 1.152x | 1.131–1.187x |
| 1024 | 60 | 1.024x | 1.004–1.062x |
| 2048 | 60 | 1.035x | 1.027–1.047x |
| 4096 | 120 | 1.021x | 1.014–1.034x |

All 420 points had equal final index sets and per-case failure counts;
each variant accumulated 16 fast-failed rows, handled by exact repair.
A separate synthetic boundary audit identified an allowed equal-cutoff tie
at Q512/N8192. Exact Top-K does not prescribe which equal-score IDs to choose.
Synthetic Q8192 is near parity; independent real Q8192 qualification is pending.

The measured Q512/N131584 scheduling intervention supports improved CTA tail
packing as the primary explanation, not faster per-query matrix arithmetic.
At the default grid, the slowest SM processes seven query rows with Q1 versus
eight with Q2; per-query Math duration is approximately unchanged.

These are prior operator qualification results, not a new GPU campaign at
release time. No TTFT, concurrent serving, or whole-model speedup is claimed.
The historical README charts retain their original implementation labels.

## Reproducibility boundary

Pin the v0.3.0 Git tag (and record its full commit), package version, kernel
implementation version, CUDA/driver, GPU, DeepGEMM revision, and adapter source
hash before end-to-end measurement. Record actual fused dispatch and fallback
reasons on every rank. Timing must include sampling, spill handling, repair,
and required adapter work. Keep profiling separate from formal timing.
