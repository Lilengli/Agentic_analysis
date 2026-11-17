# Experimental tiled causal attention for Ascend 910B

This is an **unfinished hybrid implementation**, not a complete Ascend C port
of the optimized CUDA kernel. It is an original implementation of causal
blockwise online softmax, guided by the forward algorithm in
Dao-AILab/flash-attention at `9f9742c80e69b3ca28a231c26ef931e98c2c3826`,
`csrc/flash_attn/src/flash_fwd_kernel.h`. No CUDA source is copied here.

`online_stage.cpp` is a new Ascend C AIV kernel: it applies the causal mask,
updates online maximum and sum, computes normalized block exponentials, and
returns the previous-output correction. `tiled_attention.py` orchestrates
128-key blocks, does QK and PV using PyTorch matmul (on the NPU backend),
rescales/accumulates output in FP32, and normalizes it at the end. The matmul
backend is expected to use the NPU matrix engine where supported; profile it
to establish actual execution. This is **multiple kernels**, not a single
fused Cube/Vector kernel, and has Python dispatch, intermediate writes and
allocations. It does not reproduce CUDA's pipelining or achieve its performance.

Inputs: FP16 contiguous BNSD; equal Q/K/V shapes; head dimension 128; sequence
length positive and divisible by 128; causal self-attention; no dropout or
backward. QK scores and per-block PV results are rounded to FP16; the online
state and cross-block accumulated output are FP32. This may produce errors
different from a fully FP32-accumulating fused attention kernel.

On the target machine, after loading the proper CANN environment:

```bash
python build.py
python verify_benchmark.py --seq 16384 --heads 2 --rows 64 --warmup 1 --repeat 3
```

These commands are **not yet verified on 910B**: the development host has no
CANN or NPU. Compilation, runtime correctness, and performance must all be
validated on the target. This implementation can be substantially slower and
use more memory than the built-in SDPA; it produces an O(B*H*S*128) score
temporary and launches multiple operators per tile. The output timing is a
synchronized complete-call wall time, not a single-kernel time. If compilation
or correctness fails, do not compare performance. `test_numpy.py` only checks
the recurrence and tile/mask indexing on CPU; it does not test Ascend C.

For first-divergence reporting, use `python verify_benchmark.py --seq 16384
--diagnose`. It checks each tile against a framework reference and skips timing.
The updated device stage transfers both inputs and padded state outputs through
UB/DMA rather than scalar GM cache accesses. Explicit hardware events order
MTE2, Scalar, Vector, and MTE3, including completion before UB buffer reuse.
These changes address memory visibility/synchronization defects; target NPU
testing is still required before claiming correctness.

The direct host launch now uses `stream(true)` to drain TorchNPU's host task
queue before submitting the raw kernel. `stream(false)` with an unqueued raw
launch could overtake queued QK/state producers; per-block diagnostics hid
that ordering bug by synchronizing before the stage. Queue draining adds host
overhead; future asynchronous integration should use OpCommand::RunOpApiV2
instead of removing the drain from a direct launch.
