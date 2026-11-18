# Mixed cube CPU semantic simulator

From `ascendc_fused_cube`, compile and run the short semantic suite with:

```sh
g++ -std=c++17 -O2 -pthread -DCUBE_FA_CPU_SIM sim/test_mixed_kernel.cpp -o /tmp/test_mixed_kernel
/tmp/test_mixed_kernel
```

The suite runs `RunCore<AIC,...>` alongside two `RunCore<AIV,...>` workers per
group. It checks BM 32/64, BN 64/128, sequence lengths 128/256, heads 1/2,
groups 1/3, and two independent inputs for each combination. Results are
compared with a full causal FP64 reference using absolute tolerance 0.002 plus
relative tolerance 0.01. It also checks Mmad counts, vector operation counts,
UB allocations, GM copy alignment and bounds, relay group boundaries, and
outer canaries.

`--smoke` runs only BM=32, BN=64, sequence=128, one head, and one group.

This is a CPU semantic check. Its float32 GEMM and host synchronization do not
validate CANN compilation, CATLASS lowering, device event/flag semantics, or
Ascend hardware behavior. The optional 16K tail case is omitted from this short
suite.
