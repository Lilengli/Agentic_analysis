# Sample repeated attention workloads on Ascend

Run these commands in this directory with the same Python environment used for
`verify_ascend.py`. Check correctness separately before performance sampling.
Each workload runs causal FP16 attention with shape `(1, 2, 128, 128)`.

```bash
msprof --output=./sample_triton --ai-core=on --aic-mode=sample-based \
  --aic-freq=100 --aic-metrics=PipeUtilization --sys-hardware-mem=on \
  --sys-hardware-mem-freq=100 \
  python run_attention_sampling.py --impl triton --seconds 10

msprof --output=./sample_sdpa --ai-core=on --aic-mode=sample-based \
  --aic-freq=100 --aic-metrics=PipeUtilization --sys-hardware-mem=on \
  --sys-hardware-mem-freq=100 \
  python run_attention_sampling.py --impl sdpa --seconds 10

find ./sample_triton ./sample_sdpa -type f \
  \( -name '*.csv' -o -name '*.json' -o -name '*.db' \) \
  -printf '%p  %s bytes\n'
```

Use new output directories for each run. These system samples cover the entire
process, including startup and warmup; the steady workload is repeated kernel
calls interspersed with host dispatch and synchronization. At 100 Hz the data
cannot resolve the internal activity of an individual 20–25 microsecond kernel.
The available metrics and permissions depend on the device and CANN release.
The run count and elapsed time include profiling overhead and are not latency
or throughput benchmarks. Retain the earlier per-kernel profiles for that
separate analysis.
