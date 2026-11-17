# Algorithm and data movement audit

The hybrid implements the blockwise online-softmax recurrence:

```
m_new = max(m_old, rowmax(Q @ K_block.T * scale))
alpha = exp(m_old - m_new)
P_block = exp(scores - m_new)       # causal future keys masked
l_new = alpha * l_old + rowsum(P_block)
O_new = alpha * O_old + P_block @ V_block
output = O / l                      # once, after all visible blocks
```

It never allocates the full S by S score/probability tensor. However, only the
KV axis is tiled: each iteration covers the remaining query suffix (up to S
rows), not a fixed small query tile held in UB. The matmul score tile is a GM
tensor, the custom softmax kernel reads it from GM and writes a probability
tensor to GM, and the second matmul reads that GM tensor. Running output/max/sum
are also GM tensors updated between launches. Thus online softmax is present,
but fusion/on-chip intermediate residency is absent. Online softmax alone does
not prove the IO behavior of the original optimized FlashAttention kernel.

For B=1, H=2, S=16384, D=128, KV tile=128:

* 128 custom-stage calls and 256 framework matmul calls are expected.
* The largest single score or probability tile is 8 MiB; a full FP16 score
  tensor would be 1 GiB. These are tensor sizes, not total peak allocation.
* Score plus probability tensors incur about 2.016 GiB of logical writes/reads
  over the call. This is source-derived tensor traffic, **not measured HBM**:
  caches, packing and backend workspace affect physical transfers.
* The original 384 state concatenations wrote about 2.031 GiB of newly allocated
  state outputs, in addition to reading their inputs and computing updates.
  In-place suffix updates remove these concatenations; state still lives in GM.

The new default updates O, m and l in place and preserves finished earlier
queries. `--state-update concat` retains the old path for comparison. This is a
memory-allocation/copy optimization, not on-chip kernel fusion. No NPU speedup
has yet been established for this change.

Run `python test_torch_cpu.py` to validate the actual Python orchestration
against independent full FP32 attention, compare both update paths and check
causality. This requires PyTorch but not an NPU. CPU tests do not validate the
compiled Ascend C stage or asynchronous NPU ordering.

On the target, run `python audit_execution.py --seq 16384 --profile --output
audit_inplace` to capture one warmed call and summarize exported kernel names.
It first compares the entire result to NPU SDPA. A profile can establish multiple
kernel launches; it cannot by itself prove a fused kernel's internal storage
residency or exact memory traffic. Those require source/ISA and hardware counters.

The next fusion change must introduce a bounded query tile and keep per-query
O/m/l on chip while iterating KV blocks, connect Cube QK and PV with Vector
softmax and explicit cross-core synchronization, and transfer final output to
GM. The current code does not implement that architecture.
