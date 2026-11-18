# Mixed Cube/Vector causal attention on Ascend 910B

This implementation moves **both QK and PV to AIC Cube/MAC**. A single
`__mix__(1,2)` kernel pairs each AIC with two AIVs. It is a new scheduling and
online-softmax implementation using CATLASS's low-level layout, copy and Mmad
primitives, analogous to using a matrix primitive library. It does not call
PyTorch matmul/SDPA, flash-attention-npu's attention operator, or the old AIV
vector matrix products.

**Target validation is pending.** The development host has no CANN compiler or
NPU. Source/API review and CPU semantic tests cannot establish device correctness,
physical synchronization, performance or utilization. The target tests below
must pass before using results. There is no fallback to another attention
implementation when a check fails.

## Computation and memory path

1. AIC loads Q into L1 once per query tile, streams K, and computes `Q @ K.T`
   with FP16 inputs and FP32 Cube accumulation. K is viewed column-major, so
   it needs no explicit GM transpose.
2. Fixpipe writes one score tile to a per-group relay. Both AIVs wait for it,
   each DMA-load half the rows, apply the causal mask and online softmax.
3. AIVs retain FP32 maxima, denominators and output accumulators in UB. The
   row max and sum use vector reduction APIs. Only the diagonal tile needs
   per-element mask edits. Exponentiated weights are cast to FP16 and sent to
   the relay; FP32 denominators use the weights before this cast.
4. AIC waits for both halves and computes `P @ V` using Cube, with FP32 output
   through Fixpipe. Both AIVs DMA-load their result rows and update accumulated
   output with the online rescale factor. A reverse handshake protects relay
   reuse. Final normalized output is written once per query tile.

This design intentionally uses **bounded GM relay transfers**. It does not
claim all scores/probabilities stay on chip. There is no full S-by-S tensor.
Default BM64/BN128 uses 80 KiB of relay per group, or **1.875 MiB for 24 groups**,
independent of sequence length. BM32/BN64 uses 28 KiB per group. Max configured
L1/L0A/L0B/L0C storage is 64/16/32/32 KiB, respectively; source UB allocation
per AIV for BM64/BN128 is 66,432 bytes, excluding compiler overhead/spills.

Both AIVs execute every handshake; neither exits early. The four phases are
QK_READY (FIX), P_READY (MTE3), PV_READY (FIX), TILE_DONE (V). This serial
schedule prioritizes validating Cube integration. It does not yet overlap
different KV iterations or double-buffer the relay, so high MAC utilization
and speedup are not guaranteed.

Supported: contiguous equal-shaped FP16 BNSD Q/K/V, D128, causal self-attention,
S divisible by BN and up to 16384; BM32/64, BN64/128, 1..24 AIC groups. No GQA,
dropout, variable-length tails or backward.

## Build and run on the target

The pinned, unmodified CATLASS header bundle is included with its license and
checksum. `build.py` unpacks it locally, so no dependency download or pip change
is needed. `--catlass-root` or `CATLASS_ROOT` can override it for experiments.

```bash
python build.py
python -u run.py --seq 16384 --heads 2 --seeds 1234 --verify-only \
  --profile ./profile_cube_16k --output fused_cube_verify_16k.json
```

Before mixed attention, the runner automatically tests actual Cube QK and PV
products for all tile configurations, using both random and coordinate-coded
matrices against CPU FP32 reference. A failure stops before cross-core attention.
It then checks ordinary attention execution and a profiled invocation against
64 sampled query positions from independent CPU FP32 reference, and checks all
output elements for finiteness. The profile needs a new/empty output directory.

Inspect `profile.cube_activity_verified` and `profile.kernel_metrics`: the
verification requires positive **MAC time, MAC ratio or Cube utilization**;
positive generic AIC cycles alone do not prove matrix activity. Some profiler
versions export multiple rows for a mixed launch, so `one_task_verified=false`
needs inspection rather than proving the host launched multiple kernels.
The host code makes exactly one mixed attention launch per forward call.

Repeated checks and synchronized complete-call timing:

```bash
python -u run.py --seq 16384 --heads 2 --warmup 1 --repeat 3 \
  --output fused_cube_benchmark_16k.json
```

Each timed output is checked after stopping the clock. Compare wall time to
wall time and task duration to task duration. Profile collection itself can
change runtime. The prior AIV-only 16K trace measured one 533.21 ms vector task
with zero Cube MAC activity; no speedup over it is claimed without target data.

## Source and validation boundaries

`cube_tiles.hpp` wraps the CATLASS copy/layout/Mmad primitives; `mixed_kernel.hpp`
implements scheduling, the two-AIV online recurrence and flags; `geometry.hpp`
defines aligned relay regions; `fused_attention.cpp` owns the stream/device,
bounded allocation and launch. The standalone `product` binding can isolate
Cube layout checks using `python verify_products.py`.

`sim/` tests run the actual mixed scheduling/recurrence header using a CPU
semantic mock. Its abstract GEMM does not simulate physical fractal conversion,
and software flags do not validate hardware event ordering. Actual CANN/NPU
compilation, products, repeated mixed runs and MAC counters remain necessary.

Local semantic validation passed 64 cases: all four tile configurations,
S128/256, H1/2, groups1/3 and two input repetitions. The maximum absolute error
against independent FP64 causal reference was 0.00108919, within the established
atol=.002, rtol=.01 check. Tests checked two matrix products per KV iteration,
both-AIV flag participation, repeated relay reuse, bounds/alignment, and UB
allocation accounting. They do not validate full 16K or real-device behavior.

```bash
g++ -std=c++17 -O2 -pthread -DCUBE_FA_CPU_SIM sim/test_mixed_kernel.cpp -o /tmp/test_mixed_kernel
/tmp/test_mixed_kernel
```

The CUDA algorithm reference remains Dao-AILab/flash-attention
`flash_fwd_kernel.h` at `9f9742c80e69b3ca28a231c26ef931e98c2c3826`.
Hardware primitives are from pinned CATLASS revision
`769cd40a8716b28650b6bebb08db4834eea4462f`; see `third_party/NOTICE.md` and license.

### Experimental BM128 traffic reduction

`--block-m 128 --block-n 128` reuses each K/V tile across twice as many query
rows as the original 64x128 configuration. The online softmax and FP32 relay
are unchanged. `traffic_model.py` predicts source-level logical GM bytes, not
physical HBM traffic. For S=16384, batch*heads=2 it predicts 7240 MiB total for
64x128 and 6208 MiB for 128x128 (14.25% less). These are not measured results.

The 128x128 shape requires CANN compilation and independent device correctness
and performance validation. The default remains 64x128. Each AIV uses about
132864 bytes of UB and AIC uses 64 KiB of L0C. CPU simulation cannot validate
device lowering, hardware buffer limits, event synchronization or performance.

```
python build.py
python run.py --seq 16384 --heads 2 --block-m 64 --block-n 128 --seeds 1234 4321 --rows 64 --warmup 3 --repeat 5 --profile ./profile_bm64_memory --profile-metrics memory-access --output result_bm64.json
python run.py --seq 16384 --heads 2 --block-m 128 --block-n 128 --seeds 1234 4321 --rows 64 --warmup 3 --repeat 5 --profile ./profile_bm128_memory --profile-metrics memory-access --output result_bm128.json
```

`kernel_metrics` in each result JSON contains the profiler's original fields;
compare `*_GM_to_L1_datas`, `*_L0C_to_GM_datas`, `*_GM_to_UB_datas` and
`*_UB_to_GM_datas` alongside the time. A run must pass all verification checks.
MemoryAccess availability and collected fields depend on torch_npu and CANN.
