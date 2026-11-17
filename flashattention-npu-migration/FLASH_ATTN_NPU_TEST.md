# flash-attention-npu v2 experiment (Ascend 910B)

Upstream https://github.com/MinghuasLab/flash-attention-npu, inspected commit
`65e3f0adbab14075738c85cf1dda83db9dc317f5` and CATLASS submodule commit
`769cd40a8716b28650b6bebb08db4834eea4462f`. Upstream source is not vendored
here. The tested API is `flash_attn_npu.flash_attn_func`, not the v3 or v4 API.

Build only v2 / Ascend 910 in a separate checkout and leave the existing Python
environment packages alone:

```bash
git clone https://github.com/MinghuasLab/flash-attention-npu.git "$HOME/flash-attention-npu"
git -C "$HOME/flash-attention-npu" checkout 65e3f0adbab14075738c85cf1dda83db9dc317f5
git -C "$HOME/flash-attention-npu" submodule update --init --depth 1 csrc/catlass
cd "$HOME/flash-attention-npu"
FLASH_ATTN_BUILD_VERSION=v2 FLASH_ATTN_BUILD_NPU=910 FLASH_ATTN_BUILD_JOBS=2 \
  python setup.py build_ext --inplace
```

These commands need network access and the CANN compiler, torch/torch_npu and
build prerequisites already installed. Compilation and version compatibility
have not been tested on the target NPU. If the source directory already exists,
inspect it and use a different path rather than overwriting it.

From this repository's `flashattention-npu-migration` directory:

```bash
python benchmark_flash_attn_npu.py --source "$HOME/flash-attention-npu" \
  --seq 128 --output flash_attn_npu_128.json
python benchmark_flash_attn_npu.py --source "$HOME/flash-attention-npu" \
  --seq 16384 --output flash_attn_npu_16384.json
```

All methods use identical random numerical Q/K/V, causal mask, FP16 and
scale=1/sqrt(128). The independent CPU FP32 reference checks every query row
up to 512 and 64 evenly spaced rows above 512; this is not a full long-context
correctness check. Results include two FlashAttention paths: pre-converted
contiguous BSND tensors, and a path that includes BNSD-to-BSND copies and a
contiguous BNSD output copy. SDPA uses contiguous BNSD input. The preconverted
case isolates a useful native-layout path but is not a same-layout comparison;
the conversion-inclusive case includes three input copies and one output copy.
Warmups and 30 interleaved NPU-event timings plus synchronized wall-clock times
are reported. These are eager complete-call timings, not standalone kernel
times. If a call fails or is inaccurate it is not timed.
