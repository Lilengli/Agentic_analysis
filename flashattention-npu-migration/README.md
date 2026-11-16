# FlashAttention GPU-to-Ascend first candidate

This is an unverified, preliminary port of the local GPU Triton implementation
`/home/gyli/kernelagent-local/local_results/attention_search/expanded_20261014/best_kernel.py`.
The GPU source computes causal self-attention for same-shaped FP16 Q/K/V tensors
in `[batch, heads, sequence, head_dim]` layout. This candidate preserves the
online-softmax algorithm and changes the launch to one dimension, replaces CUDA
device selection, removes GPU autotuning, and uses fixed 32-by-32 tiles.

This is a prefill MHA case, not the Qwen3 decode GQA/paged KV-cache path. The
candidate has not been compiled or run on an NPU. Triton Ascend versions and
`tl.dot` resource/shape constraints may require further adaptation.

On an Ascend environment with compatible torch, torch_npu and Triton Ascend:

```bash
python verify_ascend.py
```

This exercises small and non-aligned sequence lengths against an FP32 CPU
reference; it is not a performance benchmark or full correctness proof.
No GPU or NPU performance result is claimed. The AscendOpGenAgent migration
workflow was not run: it requires its configured agent runtime and test inputs.
