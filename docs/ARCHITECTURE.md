# Operator architecture

## Short-context path (6K–16K qualified range)

The public operator is a multi-kernel device pipeline. `prepare()` allocates all
workspace before timing; a timed invocation performs the following stages:

```text
random token gather
        ↓
sampled DeepGEMM score pass → per-row conservative threshold
        ↓
persistent DeepGEMM score + compact candidate producer
        ↓
exact compact-candidate radix reducer (9 + 7 + 8 + 8 bits)
        ↓
device failure flags ── no failure ──→ exact unordered indices
        │
        └── underflow/overflow → masked exact repair producer + reducer
```

The producer uses a 128-token KV tile, 256 math threads, and a two-CTA-per-SM
launch target. It emits packed score/index pairs into 16 logical segments with
a capacity of 14,080 candidates per row. The reducer uses a 6,656-entry on-chip
working set and records any condition that could invalidate the fast result.

The threshold is only a performance optimization. It is not part of the
correctness proof: every detected capacity failure is routed to exact repair.

## Long-context path

FusedIndexTopK keeps the same fast producer and reducer. If a row is flagged,
the row is rescored in chunks of at most 16,384 keys. Every chunk is reduced to
an exact local Top-2048 and merged with a Top-2048 packed-pair accumulator.

An item outside a chunk's local Top-K cannot enter the global Top-K, so the
hierarchical merge is exact. Workspace is bounded by one 16K candidate tile and
`O(QK)` local/accumulator buffers rather than `O(QN)` dense scores.

At `N<=16384`, the operator uses the complete-row repair path directly.

## Output contract

- indices: contiguous INT32 `[Q, 1, 2048]`;
- order: unordered;
- causal valid range for query `q`: `[0, N-Q+q+1)`;
- padding index: `-1` when fewer than K causal keys exist;
- tie policy: exact score-threshold semantics, not deterministic equal-score ID order.
