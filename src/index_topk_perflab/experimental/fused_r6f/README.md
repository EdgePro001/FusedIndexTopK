# Fused R6f candidate

R6f preserves R6e's exact partition/third-byte-histogram fusion. Its single
change is the histogram update mechanism. R6e called `warp_histogram_add` for
every candidate, so even non-selected candidates participated in a ballot.
R6f executes one direct shared-memory `atomicAdd` only when `is_selected` is
true. Real replay probes observed at most 167 such values per row, and prior
NCU showed that the shared-atomic pipeline was far from saturated.

The eight-warp tail still performs only the fourth radix byte. The 256-value
guard and the complete R5i masked repair path are unchanged.
