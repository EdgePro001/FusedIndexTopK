# Fused R8a candidate

R8a preserves R6f's target, candidate producer, reducer, exactness guard, and
timed repair. Its only algorithmic control change is the launch shape of the
sample-threshold kernel.

At the supported `N=16384`, the sampling schedule produces 128 scores per row.
R6f launches 256 threads, leaving four warps without sample work. R8a launches
128 threads and has every thread clear two of the 256 shared histogram bins.
The same four exact radix bytes are selected, so the computed threshold and all
downstream behavior are unchanged bit for bit.
