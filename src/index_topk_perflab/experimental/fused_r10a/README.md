# Fused R10a candidate

R10a keeps R9b's producer, 256-sample threshold, 2816 target, and complete-row
repair. It changes only the fast reducer's execution resources.

The producer input remains `[Q, 14080]` with 16 segments of 880 entries, so no
producer ABI changes. Before compaction, the reducer checks that the total is
at most 6656. Accepted rows use a 6656-entry on-chip score/index working set and
512 threads; overflow rows set the existing failure flag and take the timed
exact repair path. The intended H20 resource point is 53,248 dynamic SMEM bytes
and four CTAs per SM. A 7040-entry draft reached only three CTAs/SM after CUDA
allocation granularity was applied, so it was rejected before timing.
