# Architecture

FusedIndexTopK computes exact Top-K indices for the DeepSeek V3.2 indexer
without materializing the dense score matrix.

## Execution graph

```text
sample gather -> sampled GEMM -> guarded threshold
                                      |
                                      v
                     fused score production + exact Top-K
                                      |
                         fast rows ---+--- flagged rows
                                           |
                                           v
                                device-masked exact repair
```

The sampled prepass estimates a conservative score threshold for each query
row. The main kernel then performs the full GEMM and Top-K together. Sampling
changes only the amount of fast-path work: it never changes the exact result.

## Main kernel

Math warp groups produce score tiles with the pinned DeepGEMM-compatible SM90
pipeline. A CTA-local consumer performs four-byte radix selection on ordered
FP32 score bits and writes the final `K = 2048` indices directly.

For `N = 16384`, the short-context layout uses:

- 16 candidate segments
- 5,888 total candidate slots
- 228,288 bytes of configured shared memory
- no spill workspace

For `N = 8192` and `N > 16384`, the long-context layout uses:

- 8 candidate segments
- 16-bit segment-relative candidate indices
- 5,888 shared-memory candidate slots
- a bounded `8 × 256` packed-pair spill per row

The segment-relative representation remains lossless through the qualified
`N = 163840` limit. The spill is consumed by the same CTA and can overlap
with subsequent math tiles. Dense scores and the normal candidate list are
never written to global memory.

## Exactness

The threshold is an optimization hint. A row is accepted only if the main
kernel proves that it has enough candidates and that every bounded buffer was
handled. Otherwise, it sets a device failure flag.

The repair path:

1. partitions the row into 16K-key chunks;
2. recomputes only flagged rows;
3. selects exact local Top-K pairs per chunk;
4. merges chunk results on device; and
5. emits the exact global indices.

No host synchronization is required to decide whether repair runs. A row that
does not need repair is masked out of every repair stage.

## Memory boundary

“Fused” here means that the full score production and normal exact Top-K are in
one kernel. It does not mean that every byte in the complete graph lives in
shared memory:

- the sampled prepass is a separate launch;
- long-context overflow may use the bounded spill workspace;
- flagged rows use a separate exact repair graph;
- final indices are written to global memory.

This boundary is deliberate: it keeps the common path on chip while preserving
exactness for arbitrary qualified inputs.
