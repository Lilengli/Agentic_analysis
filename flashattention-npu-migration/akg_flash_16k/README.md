# AKG-generated causal attention experiment

This is an unverified candidate produced by AKG KernelDesigner and KernelGen
with a Codex CLI model adapter. The manifest records the AKG revision, model,
and SHA-256 of the unmodified output. No Ascend device was available during
generation: the generated code may fail to compile, produce incorrect output,
or perform poorly. The target identifier `ascend910b4` was an assumption.

Workload: contiguous FP16 Q/K/V in BNSD layout, B=1, H=2, S=16384,
D=128, causal masking, scale=1/sqrt(D), dropout=0. Output is FP16.
This is dense prefill rather than single-token decode.

On an Ascend NPU with the existing torch, torch_npu and triton-ascend
environment, run:

```bash
python benchmark.py --candidate ./generated_kernel.py \
  --seq 16384 --rows 64 --warmup 20 --repeat 100 \
  --output akg_compare_16k.json
```

The script compares the candidate and SDPA on the same CPU-generated inputs.
It checks 64 sampled rows against an explicit FP32 CPU reference before timing;
passing is not a proof for every row or shape. It alternates the order of
the two implementations and records full-call NPU-event and synchronized wall
times; these are not individual kernel times. Compilation is outside timing.
If compilation or correctness fails, save the error and JSON and stop: no speed
comparison is valid until corrected. Profiling is required to isolate kernels.
