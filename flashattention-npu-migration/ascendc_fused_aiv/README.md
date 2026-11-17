# Single-kernel tiled online attention, Ascend C AIV

An independent FP16 causal self-attention forward implementation, guided by
`compute_attn_1rowblock` in Dao-AILab/flash-attention at commit
`9f9742c80e69b3ca28a231c26ef931e98c2c3826`. No upstream implementation is copied.
The CUDA reference supplies the algorithmic structure; GPU MMA/thread layouts
and the optimized CUDA pipeline have not been ported.

## Implemented data path

* Exactly one direct device-kernel launch per forward call; no Python tile loop,
  framework matmul, softmax, or pre-existing NPU attention kernel inside it.
* Fixed Q tiles BM=8/16 and KV tiles BN=32/64, D=128. Q and FP32 running O/m/l
  remain in UB through the reverse-order causal KV loop.
* QK uses FP32 vector multiply and row reductions. Softmax uses online maximum
  and denominator rescaling. PV uses vector broadcast/multiply and key-axis
  tree reduction. Scores, weights, products, maxima and sums remain in UB.
* K/V are streamed once per KV tile per Q tile. Q is read once per Q tile.
  The kernel's only GM write copies the final normalized output Q tile.
* There is no score/probability/state GM workspace. Source-level UB allocation
  is fixed at 79,296 / 91,648 / 146,240 / 158,592 bytes for 8x32 / 16x32 / 8x64 /
  16x64. These do not include compiler overhead or potential spills; inspect
  target compiler output before claiming a measured total.

This is **AIV only**. It does not use AIC Cube/MAC and is not expected to match
a tuned Cube-based SDPA merely because it is fused. No NPU speedup is claimed.
On A2, the documented UB-to-L1 ND2NZ Matmul interface can use a GM relay; a
local-tensor API signature alone is not evidence of a physical on-chip path.
An efficient mixed Cube/Vector implementation with bounded GM exchange is a
different design from the strict no-GM-intermediate experiment here.

Inputs are contiguous matching BNSD FP16 Q/K/V with D=128 and S divisible by BN,
up to 16384. Batch and heads are flattened for scheduling. No dropout, GQA,
variable lengths, backward or arbitrary tail padding is implemented.

## Validation status

**Not compiled with CANN or executed on an NPU yet.** The development host has
no CANN/NPU. The same `fused_kernel.hpp` was executed in a bounds-checked CPU
semantic mock of the documented vector/DMA APIs. It passed 24 complete cases
at S=64/128/256, H=1/2, all four tile configurations and different simulated
core counts, against independent FP64 causal attention (max error <0.000914).
Four additional S=16384 cases checked the last Q tile of the second head and
all preceding KV tiles (max error <0.000015). These are not full 16K validation.
Tests also checked input immutability, aligned DMA lengths and that GM writes
contain exactly the final output element count.

The mock is **not a CANN simulator**. It cannot prove device API compatibility,
event correctness, hardware rounding, actual on-chip residency or performance.
No sanitizer result is claimed (the development host lacks its link library).

To repeat the CPU semantic tests:

```bash
g++ -std=c++17 -O2 -D_GLIBCXX_ASSERTIONS -I sim sim/test_kernel.cpp -o /tmp/test_fused_aiv
/tmp/test_fused_aiv
```

## On the 910B target

Load the same CANN environment used by the hybrid, then:

```bash
python build.py
python -u run.py --seq 16384 --heads 2 --seeds 1234 --verify-only \
  --profile ./profile_fused_aiv_16k --output fused_aiv_verify_16k.json
```

The first command compiles the extension and stores a source/binary manifest.
The second performs ordinary, unsynchronized-before-call execution, checks
64 sampled query rows against CPU FP32 reference plus all-output finiteness,
then profiles one further invocation. It does not use intermediate `.item()`
checks that could hide task-queue ordering defects. No fallback is used on failure.
The profiler summary requires exactly one matching kernel task and no other
kernel tasks to claim `one_task_verified=true`. Missing/mangled symbols or
missing CSV produce a warning, not a false confirmation. Source inspection
and compiler output are still needed to assess on-chip data residency.

For ordinary repeated verification and timing:

```bash
python -u run.py --seq 16384 --heads 2 --warmup 1 --repeat 3 \
  --output fused_aiv_benchmark_16k.json
```

Every timed output is verified after the clock stops. Reported synchronized
wall time includes wrapper and queue-drain overhead; NPU event time is a stream
interval. Use `kernel_details.csv` for the device task's own duration. This
serial AIV design can be slow at 16K; hardware testing must establish feasibility.

## References for hardware/API assumptions

* [A2 UB-to-L1 ND2NZ transfer interface](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMD-API/basic_api/cube_compute_ISASI/cube_compute_load/DataCopy_UBToL1_ND2NZ.md)
* [WholeReduceSum](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0081.html)
* [Brcb](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910beta2/API/ascendcopapi/atlasascendc_api_07_0199.html)
* [TorchNPU direct-launch and task-queue ordering](https://gitcode.com/Ascend/op-plugin/tree/master/examples/cpp_extension)
