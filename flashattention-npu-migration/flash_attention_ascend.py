import torch
import torch_npu
import triton
import triton.language as tl


@triton.jit
def _causal_attention_kernel(
    Q,
    K,
    V,
    O,
    H: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    SQB: tl.constexpr,
    SQH: tl.constexpr,
    SQN: tl.constexpr,
    SQD: tl.constexpr,
    SKB: tl.constexpr,
    SKH: tl.constexpr,
    SKN: tl.constexpr,
    SKD: tl.constexpr,
    SVB: tl.constexpr,
    SVH: tl.constexpr,
    SVN: tl.constexpr,
    SVD: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    program = tl.program_id(0)
    query_blocks = tl.cdiv(N, BLOCK_M)
    query_block = program % query_blocks
    batch_head = program // query_blocks
    batch = batch_head // H
    head = batch_head % H

    rows = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
    columns = tl.arange(0, BLOCK_N)
    features = tl.arange(0, BLOCK_D)

    q_base = Q + batch * SQB + head * SQH
    k_base = K + batch * SKB + head * SKH
    v_base = V + batch * SVB + head * SVH

    q = tl.load(
        q_base + rows[:, None] * SQN + features[None, :] * SQD,
        mask=(rows[:, None] < N) & (features[None, :] < D),
        other=0.0,
    )

    scale = tl.rsqrt(tl.full((), D, tl.float32))
    log2_e = 1.4426950408889634

    running_max = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    running_sum = tl.zeros((BLOCK_M,), tl.float32)
    accumulator = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

    # Fuse QK^T, scaling, causal masking, online softmax, and multiplication
    # by V. Scores and probabilities remain local to this program.
    key_end = tl.minimum((query_block + 1) * BLOCK_M, N)
    history_end = (query_block * BLOCK_M // BLOCK_N) * BLOCK_N
    for key_start in tl.range(0, history_end, BLOCK_N):
        keys = key_start + columns

        k = tl.load(
            k_base + features[:, None] * SKD + keys[None, :] * SKN,
            mask=(features[:, None] < D) & (keys[None, :] < N),
            other=0.0,
        )

        scores = tl.dot(q, k).to(tl.float32) * scale

        next_max = tl.maximum(running_max, tl.max(scores, axis=1))
        correction = tl.exp2((running_max - next_max) * log2_e)
        probabilities = tl.exp2((scores - next_max[:, None]) * log2_e)

        next_sum = (
            running_sum * correction + tl.sum(probabilities, axis=1)
        )
        accumulator = accumulator * correction[:, None]

        v = tl.load(
            v_base + keys[:, None] * SVN + features[None, :] * SVD,
            mask=(keys[:, None] < N) & (features[None, :] < D),
            other=0.0,
        )
        accumulator = tl.dot(
            probabilities.to(v.dtype), v, accumulator
        )

        running_max = next_max
        running_sum = next_sum

    for key_start in tl.range(history_end, key_end, BLOCK_N):
        keys = key_start + columns

        k = tl.load(
            k_base + features[:, None] * SKD + keys[None, :] * SKN,
            mask=(features[:, None] < D) & (keys[None, :] < N),
            other=0.0,
        )

        scores = tl.dot(q, k).to(tl.float32) * scale
        allowed = (keys[None, :] < N) & (keys[None, :] <= rows[:, None])
        scores = tl.where(allowed, scores, -float("inf"))

        next_max = tl.maximum(running_max, tl.max(scores, axis=1))
        correction = tl.exp2((running_max - next_max) * log2_e)
        probabilities = tl.exp2((scores - next_max[:, None]) * log2_e)

        next_sum = (
            running_sum * correction + tl.sum(probabilities, axis=1)
        )
        accumulator = accumulator * correction[:, None]

        v = tl.load(
            v_base + keys[:, None] * SVN + features[None, :] * SVD,
            mask=(keys[:, None] < N) & (features[None, :] < D),
            other=0.0,
        )
        accumulator = tl.dot(
            probabilities.to(v.dtype), v, accumulator
        )

        running_max = next_max
        running_sum = next_sum

    result = accumulator / running_sum[:, None]
    output_offsets = (
        batch_head * N * D + rows[:, None] * D + features[None, :]
    )
    tl.store(
        O + output_offsets,
        result.to(O.dtype.element_ty),
        mask=(rows[:, None] < N) & (features[None, :] < D),
    )


def kernel_function(q, k, v, *, block_m=32, block_n=32):
    """Return causal FP16 attention for tensors shaped [batch, heads, N, D].

    block_m/block_n set the query/key tile sizes. The defaults preserve the
    original 32x32 configuration. Supported sizes still depend on the NPU
    compiler and resource limits, and must be checked on the target device.
    One fused Triton kernel computes scaled QK^T, causal masking, stable
    online softmax, and multiplication by V without intermediate global
    tensors. The wrapper only validates inputs, allocates, and launches.
    """
    if (block_m not in (16, 32, 64, 128, 256) or block_n not in (16, 32, 64, 128, 256)
            or block_m < block_n):
        raise ValueError('block sizes must be 16, 32, 64, 128 or 256 and block_m >= block_n')
    for tensor in (q, k, v):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("q, k, and v must be torch.Tensor instances.")
        if tensor.ndim != 4:
            raise ValueError("Inputs must have shape [batch, heads, N, D].")
        if tensor.device.type != "npu":
            raise ValueError("Inputs must be NPU tensors.")
        if tensor.dtype != torch.float16:
            raise TypeError("Inputs must have dtype torch.float16.")

    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q, k, and v must have identical shapes.")
    if q.device != k.device or q.device != v.device:
        raise ValueError("q, k, and v must be on the same device.")

    batch, heads, sequence_length, head_dimension = q.shape
    if head_dimension < 1 or head_dimension > 256:
        raise ValueError("The head dimension must be between 1 and 256.")

    output = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    if batch == 0 or heads == 0 or sequence_length == 0:
        return output

    # One program per query tile and head. Tile choice is manual, not autotuned.
    grid = (triton.cdiv(sequence_length, block_m) * batch * heads,)
    _causal_attention_kernel[grid](
        q, k, v, output, heads, sequence_length, head_dimension,
        *q.stride(), *k.stride(), *v.stride(),
        BLOCK_D=max(16, triton.next_power_of_2(head_dimension)),
        BLOCK_M=block_m, BLOCK_N=block_n,
    )
    return output
