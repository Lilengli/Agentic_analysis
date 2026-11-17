# Compare causal attention implementations

`official_fused_attention.py` is an unmodified copy of
`third_party/ascend/tutorials/06-fused-attention.py` from
https://github.com/triton-lang/triton-ascend at commit
`877602d68629a6e9471800b3fb32a30459af37db`. The source's MIT notice and the
upstream project license are retained (`LICENSE.triton-ascend`). This is the
upstream main branch, not necessarily the version matching Triton Ascend 3.2.2.

On the NPU, from this directory:

```bash
python compare_attention.py --output attention_compare_results.json
```

It compares migrated Triton, PyTorch SDPA and the official example using the
same CPU-seeded FP16 inputs `(1,2,128,128)`, causal mask, scale `1/sqrt(128)`,
and dropout zero. Each output must pass an FP32 CPU reference with atol and rtol
0.02 before timings are accepted. For sequence lengths at most 512, all query
rows are checked; longer sequences check 32 deterministic, evenly spaced query
rows (reported in JSON). This is not full long-sequence correctness validation.
Calls are warmed and interleaved; the NPU
event range measures the complete function call, including device operations
for setup, and the synchronized host wall time is reported separately. Results
contain per-sample times and environment versions. This is eager execution,
not graph replay, and profiler and benchmark times must not be mixed.

The example uses fixed grid 20 and allocates temporary buffers, including
zero-initialization, on each call. This remains unchanged. Source-level
compatibility with the installed Triton Ascend, including the extension API,
and causal correctness for the chosen tile must be checked on the device.
An implementation that fails import, compilation, or correctness is reported in
JSON and is never given a successful timing. These are short-kernel events;
dispatch gaps and instrumentation can affect event measurements. Compare
medians, repeat runs, and use profiling for device-kernel-only comparisons.
