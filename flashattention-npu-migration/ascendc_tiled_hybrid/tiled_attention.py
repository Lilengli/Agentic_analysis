"""Experimental causal attention: Ascend C online stage + framework Cube matmuls.

Not a fused FlashAttention kernel. Sequential tile loop and fp16 intermediates
are deliberate limitations of this first hybrid implementation.
"""
import math
import torch

TILE = 128


def _reference_stage(scores, old_max, old_sum, rows_per_head, start, scale):
    """CPU-only implementation used to validate orchestration."""
    device = scores.device
    pos = torch.arange(rows_per_head, device=device).repeat(scores.shape[0] // rows_per_head)
    cols = torch.arange(TILE, device=device)
    x = (scores.float() * scale).masked_fill(cols[None, :] > pos[:, None], float('-inf'))
    m = torch.maximum(old_max, x.max(dim=1).values)
    p = torch.exp(x - m[:, None])
    alpha = torch.exp(old_max - m)
    return p.to(scores.dtype), alpha, m, old_sum * alpha + p.sum(dim=1)


def causal_attention(q, k, v, *, cpu_reference_stage=False, diagnose=False, state_update='inplace'):
    if state_update not in ('inplace', 'concat'):
        raise ValueError('state_update must be inplace or concat')
    if torch.is_grad_enabled() and any(x.requires_grad for x in (q, k, v)):
        raise ValueError('this inference implementation does not implement backward')
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError('q/k/v must have matching B,H,S,D shapes')
    if q.dtype != torch.float16 or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError('q/k/v must be FP16')
    if q.device != k.device or q.device != v.device or not all(x.is_contiguous() for x in (q, k, v)):
        raise ValueError('q/k/v must be contiguous and on the same device')
    b, h, seq, d = q.shape
    if d != 128 or seq == 0 or seq % TILE:
        raise ValueError('requires D=128 and positive S divisible by 128')
    if cpu_reference_stage:
        if q.device.type != 'cpu':
            raise ValueError('reference stage is only for CPU tests')
        stage = _reference_stage
    else:
        if q.device.type not in ('npu', 'privateuseone'):
            raise ValueError('NPU required')
        import online_stage_ext
        stage = online_stage_ext.online_stage
    q, k, v = [x.view(b * h, seq, d) for x in (q, k, v)]
    m = torch.full((b * h, seq), -torch.finfo(torch.float32).max, dtype=torch.float32, device=q.device)
    l = torch.zeros_like(m)
    o = torch.zeros((b * h, seq, d), dtype=torch.float32, device=q.device)
    for start in range(0, seq, TILE):
        rows = seq - start
        # Score and probability tiles are O(B*H*S*TILE), never O(S*S).
        scores = torch.matmul(q[:, start:, :], k[:, start:start + TILE, :].transpose(-1, -2)).contiguous()
        old_m = m[:, start:].contiguous().view(-1)
        old_l = l[:, start:].contiguous().view(-1)
        def check(name, x):
            if not torch.isfinite(x).all().item():
                raise RuntimeError(f'first nonfinite: key_start={start} tensor={name}')
        if diagnose:
            check('scores', scores)
            check('old_max', old_m)
            check('old_sum', old_l)
        p, alpha, new_m, new_l = stage(scores.view(-1, TILE), old_m, old_l,
                                       rows, start, 1.0 / math.sqrt(d))
        if diagnose:
            ref = _reference_stage(scores.view(-1, TILE), old_m, old_l, rows, start, 1.0 / math.sqrt(d))
            for name, got, expected in zip(('p', 'alpha', 'new_max', 'new_sum'), (p, alpha, new_m, new_l), ref):
                check(name, got)
                if not torch.allclose(got, expected, atol=0.002, rtol=0.01):
                    err = (got.float() - expected.float()).abs().max().item()
                    raise RuntimeError(f'first divergence: key_start={start} tensor={name} max_abs={err}')
            if (new_l <= 0).any().item():
                raise RuntimeError(f'nonpositive denominator: key_start={start}')
        p = p.view(b * h, rows, TILE)
        alpha = alpha.view(b * h, rows, 1)
        # Cube path receives FP16 P and V; accumulation across tiles remains FP32.
        delta = torch.matmul(p, v[:, start:start + TILE, :]).float()
        if state_update == 'inplace':
            # Preserve finalized earlier queries; update only the active suffix.
            # The producer/consumer ordering is maintained by TorchNPU's queue
            # and the stream(true) drain at the next custom-stage launch.
            o[:, start:, :].mul_(alpha).add_(delta)
            m[:, start:].copy_(new_m.view(b * h, rows))
            l[:, start:].copy_(new_l.view(b * h, rows))
        else:
            # Original implementation retained for an A/B comparison.
            o = torch.cat((o[:, :start, :], o[:, start:, :] * alpha + delta), dim=1)
            m = torch.cat((m[:, :start], new_m.view(b * h, rows)), dim=1)
            l = torch.cat((l[:, :start], new_l.view(b * h, rows)), dim=1)
        if diagnose:
            check('PV delta', delta)
            check('accumulated output', o)
            if start % 2048 == 0:
                print(f'block checks passed: key_start={start}', flush=True)
    return (o / l.unsqueeze(-1)).to(q.dtype).view(b, h, seq, d)
