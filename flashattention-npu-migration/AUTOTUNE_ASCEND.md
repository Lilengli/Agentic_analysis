# Experimental Ascend attention autotuning

`flash_attention_ascend_autotune.py` wraps the same migrated Triton kernel in
`triton.autotune` with default candidates: 32x32, 64x32, 64x64, 128x64.
The benchmark can select experimental larger candidates with `--tiles`, for
example `--tiles 128x64 128x128 256x32 256x64 256x128`. Run large configurations
one at a time initially because compilation may fail, hang or fault the device.
It first compiles/runs and checks each candidate against a CPU FP32 causal
reference for the calibration input; only candidates passing the check are
sent to Triton's autotuner. For length >512 the check covers 64 query rows,
not the entire output. Failed configurations are reported. A device-side fault
may invalidate the NPU process; failures are not guaranteed recoverable.

The in-process cache key includes device, dtype, shape, all input strides,
candidate tiles and validation settings. A new input value with the same key reuses the selected
configuration; calibration therefore cannot prove correctness for all inputs.
This prototype is not thread-safe, does not support graph capture on the first
call, and has not been validated on the target NPU. The first call includes CPU
reference computation, candidate checks, compilation and autotuning. Exclude it
from steady-state timings. It does not automatically choose new candidates or
persist tuning results across processes.

On the NPU, run from this directory with the existing torch_npu environment:

```bash
TRITON_PRINT_AUTOTUNING=1 python benchmark_attention_autotune.py \
  --seq 128 --output autotune_128.json

TRITON_PRINT_AUTOTUNING=1 python benchmark_attention_autotune.py \
  --seq 16384 --output autotune_16384.json
```

The benchmark checks a second random input after calibration, then measures
steady-state complete-call NPU events and synchronized host wall time for the
autotuned version, SDPA and passing 32x32/128x64 manual candidates. First-call
time, selected config when exposed by the installed Triton, versions, per-run
samples and candidate errors are recorded in JSON. Results include dispatcher
and allocation overhead and are not profiler-measured standalone kernel time.
Triton Ascend versions have different autotune/backend APIs; failures on the
installed 3.2.2 stack require diagnosis rather than being silently hidden.
