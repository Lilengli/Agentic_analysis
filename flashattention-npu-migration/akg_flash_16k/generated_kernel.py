import math

import torch
import triton
import triton.language as tl

try:
    import torch_npu
except Exception:
    torch_npu = None


@triton.jit
def causal_attention_16k_kernel(
    Q,
    K,
    V,
    O,
    total_tasks,
    num_q_tiles,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CORE_NUM: tl.constexpr,
):
    pid = tl.program_id(0)

    # 每个任务处理一个 head 的 query tile，在线 softmax 避免生成完整注意力矩阵。
    for task_idx in range(pid, total_tasks, CORE_NUM):
        head_idx = task_idx // num_q_tiles
        q_tile_idx = task_idx % num_q_tiles
        q_start = q_tile_idx * BLOCK_M

        q_offsets = q_start + tl.arange(0, BLOCK_M)
        d_offsets = tl.arange(0, HEAD_DIM)
        q_mask = q_offsets < SEQ_LEN

        q_ptrs = Q + (
            (head_idx * SEQ_LEN + q_offsets[:, None]) * HEAD_DIM
            + d_offsets[None, :]
        )
        q = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0)

        row_max = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
        row_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)
        out_acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)

        # 使用全编译期循环边界；运行时条件跳过 query tile 之后的 key tile。
        for k_start in range(0, SEQ_LEN, BLOCK_N):
            if k_start <= q_start:
                k_offsets = k_start + tl.arange(0, BLOCK_N)
                k_mask = k_offsets < SEQ_LEN

                k_ptrs = K + (
                    (head_idx * SEQ_LEN + k_offsets[:, None]) * HEAD_DIM
                    + d_offsets[None, :]
                )
                v_ptrs = V + (
                    (head_idx * SEQ_LEN + k_offsets[:, None]) * HEAD_DIM
                    + d_offsets[None, :]
                )
                k = tl.load(k_ptrs, mask=k_mask[:, None], other=0.0)
                v = tl.load(v_ptrs, mask=k_mask[:, None], other=0.0)

                scores = tl.dot(q, tl.trans(k)) * SCALE
                scores = tl.where(
                    k_mask[None, :], scores, -float("inf")
                )

                causal_mask = k_offsets[None, :] <= q_offsets[:, None]
                scores = tl.where(
                    causal_mask, scores, -float("inf")
                )

                tile_max = tl.max(scores, axis=1)
                next_max = tl.maximum(row_max, tile_max)

                alpha = tl.math.exp2((row_max - next_max) * 1.4426950408889634)
                probs = tl.math.exp2(
                    (scores - next_max[:, None]) * 1.4426950408889634
                )
                tile_sum = tl.sum(probs, axis=1)

                row_sum = row_sum * alpha + tile_sum
                out_acc_scaled = out_acc * alpha[:, None]
                probs_f16 = probs.to(tl.float16)
                out_acc = tl.dot(probs_f16, v, acc=out_acc_scaled)
                row_max = next_max

        result = out_acc / row_sum[:, None]
        out_ptrs = O + (
            (head_idx * SEQ_LEN + q_offsets[:, None]) * HEAD_DIM
            + d_offsets[None, :]
        )
        tl.store(
            out_ptrs,
            result.to(tl.float16),
            mask=q_mask[:, None],
        )


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        try:
            if torch_npu is None:
                raise RuntimeError("torch_npu is unavailable")
            device_limits = torch_npu.npu.npu_config.get_device_limit(0)
            self.CUBE_CORE_NUM = int(device_limits.get("cube_core_num", 20))
        except Exception:
            self.CUBE_CORE_NUM = 20

    def forward(self, q, k, v):
        if not q.is_contiguous():
            q = q.contiguous()
        if not k.is_contiguous():
            k = k.contiguous()
        if not v.is_contiguous():
            v = v.contiguous()

        batch_size, num_heads, seq_len, head_dim = q.shape
        output = torch.empty_like(q)

        block_m = 32
        block_n = 128
        num_q_tiles = triton.cdiv(seq_len, block_m)
        total_tasks = batch_size * num_heads * num_q_tiles
        scale = 1.0 / math.sqrt(head_dim)

        causal_attention_16k_kernel[(self.CUBE_CORE_NUM,)](
            q,
            k,
            v,
            output,
            total_tasks,
            num_q_tiles,
            SEQ_LEN=seq_len,
            HEAD_DIM=head_dim,
            SCALE=scale,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            CORE_NUM=self.CUBE_CORE_NUM,
        )
        return output