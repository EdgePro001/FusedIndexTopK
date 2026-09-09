# Fused R5i exact candidate

This directory contains the experimental DeepGEMM-producer + sampled TopK
fusion. It does not modify the frozen DeepGEMM or FlashInfer source trees.

- `plugin.py`: graph wiring, buffers, exact support domain, and stage metadata.
- `sampling.py` / `csrc/sampling_threshold_r5i.cu`: random-token KV gather and
  sampled radix threshold.
- `producer.py`, `csrc/smxx_fp8_mqa_candidate_r5i.hpp`, and
  `csrc/include/itk_fused_r5i/sm90_fp8_mqa_candidate_r5i.cuh`: JIT wrapper and
  project-local DeepGEMM derivative with the role-split one-ballot epilogue.
- `candidate_reducer.py` / `csrc/segmented_candidate_topk_r5i.cu`: compact
  candidate radix plus persistent masked complete-row repair reducer.
- `csrc/candidate_topk_r5i.cu`: earlier unsegmented reducer retained only as an
  experimental control.

The v3 plugin is `exact_topk=True` only for its explicit compressed-workload
specialization `N=16384`, `K=2048`, and even `Q`. Device guards detect fast-path
underflow or overflow; two timed masked kernels recompute and exactly select
only failed rows without a host synchronization. See
`docs/FUSED_DEEPGEMM_R5I_EXACT_V3.md` for the proof boundary, fault injection,
formal five-layer H20 results, memory cost, and NCU evidence. The v2 document
retains the optimization history and rejected alternatives.
