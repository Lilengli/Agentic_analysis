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
2. Fixpipe writes each score tile to one of four rotating GM relay slots.
   Both AIVs wait for the tile's slot, then each DMA-loads half the rows and
   performs causal masking and online softmax. AIC can issue three further QKs
   before waiting for the current tile's probabilities. This bounds how many
   QK writes can queue ahead of the first PV.
3. AIVs retain FP32 maxima, denominators and output accumulators in UB. The
   row maxima use two levels of BlockReduceMax per 64-column segment. Row
   sums use two levels of BlockReduceSum per 64-column segment. The host caches
   packed causal masks in GM by device and tile shape. Each active AIV loads its mask
   table into UB once before waiting for the first QK tile. The
   diagonal tile applies its mask with one Tensor–Scalar Select call. Each
   AIV updates the running row maxima and alpha exponents with vector
   Max/Sub, then broadcasts the updated row maxima with Brcb and subtracts them
   across all rows with one Sub call per 64-column segment.
   Exponentiated weights are cast to FP16 and sent to
   the relay; FP32 denominators use the weights before this cast.
4. AIC waits for both halves and computes `P @ V` using Cube, with FP32 output
   through Fixpipe. Each AIV broadcasts alpha and scales the old output
   with strided vector multiplies while PV is pending. When PV is ready,
   the AIV DMA-loads its result rows and adds them to the scaled output. A
   reverse handshake protects reuse of each PV slot. Final normalized output
   is written once per query tile.

This design intentionally uses **bounded GM relay transfers**. It does not
claim all scores/probabilities stay on chip. There is no full S-by-S tensor.
Default BM64/BN128 uses 272 KiB of relay per group, or **6.375 MiB for 24 groups**,
independent of sequence length. BM32/BN64 uses 92 KiB per group and BM128/BN128
uses 544 KiB, or **12.75 MiB for 24 groups**. The extra slots increase workspace
capacity without adding a transfer per tile. For BM64/BN128, configured
L1/L0A/L0B/L0C storage is 64/16/32/64 KiB, respectively. Source UB allocation
per AIV is 93,312 bytes.
The UB budget also leaves 8 KiB for Select's internal workspace.

Ordered QK notifications alternate two FIX ready flags across the four score
slots. Each of the three P/PV relay pairs
has its own P_READY (MTE3), PV_READY (FIX), and TILE_DONE (V) flags. P_READY
also releases the score slot. The AIC can issue the first three PV products
without a TILE_DONE wait, then waits for both AIVs to consume PV from three tiles
earlier before writing to that same slot. At the end of a query tile it drains
only the outstanding releases before resetting the relay indices.

The AIC alternates two L0C slots across QK and PV products independently of
the GM slots. Each slot's M_FIX event gates its copy to GM, and its FIX_M
event gates reuse after that copy.

Each AIV alternates two score buffers and rotates three P views and three
alpha vectors in UB. It scales the old output with the current tile's alpha,
then prepares and publishes up to two later P tiles before waiting for the
current PV. The AIV has consumed the previous PV from a slot before reusing
its P buffer. Output accumulation stays in KV order. On a device trace, compare when tile C's
P_READY and PV MMAD occur relative to tile A's TILE_DONE, and measure the
steady interval between PV_READY events and the whole call time.

Supported: contiguous equal-shaped FP16 BNSD Q/K/V, D128, causal self-attention,
S divisible by BN and up to 16384; BM32/64/128 with BM<=BN and BN64/128, plus
the experimental BM128/BN512 path, 1..24 AIC groups. The experimental path
requires S divisible by 512. No GQA,
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

The semantic suite covers all five tile configurations,
S128/256, H1/2, groups1/3 and two input repetitions. It also checks S512 with
BM128/BN128 and S640 with BM32/BN64, including a full score ring and wraparound.
BM128/BN512 runs at S512 and S2560 to cover all four causal alignments and
reuse of both relay rings.
It checks the score ring depth, PV relay reuse waits, and compares results
against an independent FP64 causal reference with atol=.002 and rtol=.01.
Other checks cover the number of QK and PV MMADs and the logical GM writes,
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
185600 bytes of explicit UB plus the 8 KiB Select workspace, and each AIC uses
two 64 KiB L0C slots. CPU simulation cannot validate
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

### Experimental BM128 logical 512-token stack

The `--block-m 128 --block-n 512` experiment increases the logical KV stack
processed by each softmax/PV handoff to 512 tokens while keeping each Cube QK
slice at 128 columns. For one logical stack, Cube produces four 128x128 QK
products. For PV, it accumulates four 128x128 MMADs into the same L0C result and
does one Fixpipe only after all four products. This reduces PV relay handoffs
from four per 512 KV tokens to one.

Each AIV handles 64 of the 128 query rows in four 16x512 score chunks. The
score and P buffers ping-pong between those chunks, and row reductions use
three stages over the 512 columns. Explicit UB allocation is 184,576 bytes
per AIV plus the 8 KiB Select workspace. The implementation
retains four QK score slots in GM and the triple P/PV relay. Every diagonal
logical stack is processed in full and then causally masked, so sequence length
must be a multiple of 512. This path is experimental and needs target correctness,
profiling and timing validation. The default remains BM64/BN128.

For BM128/BN512, the relay formula accounts for four FP32 score slots and three
FP16 P plus FP32 PV slots:
`4 * 128 * 512 * 4 + 3 * (128 * 512 * 2 + 128 * 128 * 4)` bytes =
**1600 KiB per group**, or **37.5 MiB for 24 groups**. The ordinary standalone
product probes stay at their physical tile sizes; the mixed forward path checks
every output dimension on the selected query rows against the CPU reference.

```bash
python build.py
python -u run.py --seq 16384 --heads 2 --block-m 128 --block-n 512 \
  --seeds 1234 4321 --rows 64 --warmup 3 --repeat 5 \
  --output generated_16k_bm128_bn512.json
```

### PV V prefetch before P_READY

For BM128/BN512, `--prefetch-pv-v` submits the four 128×128 FP16 V
loads to a separate 128 KiB L1 buffer before waiting for P_READY.
The subsequent PV GEMMs consume those staged slices instead of loading
V again. The P loads, L0A/L0B reuse, QK order and FP32 PV accumulation
remain the same. There are four PV MMADs and one PV Fixpipe per stack.
The GM relay and V transfer volume are unchanged; the extra L1 staging
brings the explicit Cube L1 buffers to 224 KiB.

Run the same BM128/BN512 benchmark with and without `--prefetch-pv-v`.
A 1536 / core9 simulator comparison can show the V loads on MTE2 before
P_READY and how much of their time overlaps the wait. Only the 16K board
timings can determine whether the new reads help the full kernel.

### Cross-query QK overlap (BM128/BN512)

V prefetch is the default for BM128/BN512. In the 16K two-head board
comparison, it changed the event median from 1155.00 to 1145.52 us.
Use `--no-prefetch-pv-v` to select the earlier baseline.

`--cross-query` builds on V prefetch. Once Cube has issued all PVs for
a query block, it can submit the first QK stacks of the next query while
Vector is still consuming the preceding PV results or writing its final
output. Cube still waits for both AIVs' TILE_DONE before reusing each PV
relay slot; it drains the outstanding slots when the group's last query
ends. Vector processes one query at a time, so no additional UB state,
GM relay slots, or cross-core flag IDs are needed.

Compare BM128/BN512 with and without `--cross-query` in the same build.
For an instruction comparison, use a group with at least two query
blocks, such as S=1536, H=1, 10 groups and core0 (tasks 0 and 10).
A group with only one query block cannot show the new overlap.

### Trim the causal diagonal stack (BM128/BN512)

`--cross-query --trim-diagonal` runs on the measured fastest cross-query
schedule. The first KV stack of each query contains the causal boundary.
Its active width is `query_start % 512 + 128`: 128, 256, 384 or 512 columns.
Cube only issues QK and PV MMADs for that prefix and prefetches only the
matching V slices. On the diagonal stack, Vector uses strided DataCopy
for the same score and P prefix. It retains the 512-column UB layout and
softmax math, initializing unfilled UB lanes before applying the usual
row-level causal mask. All older stacks, GM relay strides, and flags
are unchanged.

For S=16384, each head processes 8448 physical 128-token slices without
trimming and 8256 with trimming, a 2.27% reduction in QK/PV slice count.
Benchmark both modes with `--cross-query` enabled to isolate trimming.

### Batch the final output normalization (BM128/BN512)

Add `--vector-normalize` to `--cross-query --trim-diagonal` to compare
the final AIV output normalization. Each AIV uses its existing UB
scratch to broadcast the 64 FP32 row denominators, then issues two
strided vector divisions covering 64 output columns each. Each
division processes all 64 rows. The result is cast to FP16 and
written to GM after the divisions. The baseline computes a reciprocal
and scales the output for each row.

### Balance query work across AIC groups

Add `--balance-queries` to `--cross-query --trim-diagonal`. Even waves
of query tiles go to groups in ascending order, and odd waves go in
descending order. Cube and both Vector subcores use the same mapping,
including for the partially filled last wave. Each group keeps its
own GM relay slots and writes each query to its original output rows.

For S=16384, two heads and 24 groups, the predicted physical KV slice
counts go from 610–768 per group to 678–693. Compare with the flag off
using the same tile, diagonal trimming and normalization settings.

### Keep the query tile in L0A (BM128/BN512)

Add `--keep-q-l0a` to `--cross-query --trim-diagonal`. The first QK
slice of each query loads its 128-by-128 FP16 Q tile into a dedicated
32 KiB L0A slot. All subsequent QK slices of that query read the same
slot. PV loads P into the other 32 KiB slot. Each AIC uses 64 KiB of
L0A in this mode.

With S=16384 and two heads, Q loads from L1 to L0A drop from 16,512 to
256, a reduction of 508 MiB of logical on-chip transfers. Compare
with the flag off using the same query assignment and normalization.
