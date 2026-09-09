# Fused R9a candidate

R9a changes only R8a's independent sample count from 128 to 256. The candidate
target remains 3072 and all exact guard/repair code is unchanged.

DeepGEMM's sampled scorer has a fixed `BLOCK_KV=256`. With 128 samples it still
issues the complete 256-wide WGMMA tile. R9a fills the previously padded half
with real random-token samples, seeking a lower-variance threshold estimate
without adding another Tensor Core tile.
