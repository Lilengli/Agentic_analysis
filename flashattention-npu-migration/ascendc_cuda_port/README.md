# Ascend C causal attention: first correctness prototype

This is a new, limited implementation informed by the causal forward algorithm
in the official Dao-AILab CUDA FlashAttention source:
`csrc/flash_attn/src/flash_fwd_kernel.h` at commit
`9f9742c80e69b3ca28a231c26ef931e98c2c3826` (upstream LICENSE included). It is
an algorithm-level translation, **not** a line-by-line CUDA translation and
not the full official FlashAttention implementation.

It maps one query row to a task, iterates causal K/V in chunks of 32, maintains
online softmax max/sum/output, and uses AIV scalar operations plus vector Exp.
It does not use Cube/MAC, asynchronous buffering, CUDA GPU scheduling, graph
execution, GQA or KV cache. It will be dramatically slower than optimized
FlashAttention and is intended solely to validate the Ascend C launch and the
causal online-softmax recurrence. Supported inputs: contiguous FP16 BNSD,
batch 1, 1 or 2 heads, sequence lengths 1 through 128, head dimension 128.
Do not use this prototype for long contexts or model inference.

## Verify

The CPU-only recurrence sanity test works without an NPU:

```bash
python test_online_math.py
```

On an Ascend 910B environment with CANN, bisheng, torch and torch_npu already
configured:

```bash
python build.py
python verify.py --seq 8 --heads 1
python verify.py --seq 32 --heads 2
python verify.py --seq 65 --heads 2
python verify.py --seq 128 --heads 2
```

Run in order; stop if compilation or any test fails. Scalar global-memory
loads make the larger tests potentially slow. Test execution requires a free
NPU and can hit device watchdog/resource limits; outputs are not established
safe or correct until all rows have been checked on the target. `build.py`
compiles in this directory without pip installation and assumes dav-2201,
CANN 9.x, and a compatible torch_npu ABI. Compilation and NPU execution have
not been performed in the development workspace; compatibility is unverified.

The current process must have NPU device 0 selected. Tests compare every output
against an independent CPU FP32 reference for two random seeds. They are not
performance measurements. Use a dedicated process and check any compilation
or runtime failure before progressing to longer sequences.
