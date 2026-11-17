"""Compare sampled rows against independent FP32 CPU attention; benchmark end-to-end."""
import argparse
import json
import math
import time
import torch
import torch_npu  # noqa: F401
from tiled_attention import causal_attention

p = argparse.ArgumentParser()
p.add_argument('--seq', type=int, default=16384)
p.add_argument('--batch', type=int, default=1)
p.add_argument('--heads', type=int, default=2)
p.add_argument('--rows', type=int, default=64)
p.add_argument('--warmup', type=int, default=1)
p.add_argument('--repeat', type=int, default=3)
p.add_argument('--output', default='tiled_hybrid_result.json')
a = p.parse_args()
if a.seq < 128 or a.seq % 128 or min(a.batch, a.heads, a.rows, a.repeat) < 1 or a.warmup < 0:
    p.error('seq must be a positive multiple of 128; batch, heads, rows and repeat must be positive')
torch.manual_seed(1234)
shape = (a.batch, a.heads, a.seq, 128)
q_cpu, k_cpu, v_cpu = [torch.randn(shape, dtype=torch.float16) for _ in range(3)]
q, k, v = [x.to('npu:0') for x in (q_cpu, k_cpu, v_cpu)]
with torch.inference_mode():
    out = causal_attention(q, k, v)
    torch.npu.synchronize()
    indices = torch.linspace(0, a.seq - 1, min(a.rows, a.seq)).long().unique()
    ref_scores = (q_cpu[:, :, indices, :].float() @ k_cpu.float().transpose(-1, -2)) / math.sqrt(128)
    mask = torch.arange(a.seq).view(1, 1, 1, -1) > indices.view(1, 1, -1, 1)
    ref = torch.softmax(ref_scores.masked_fill(mask, -float('inf')), dim=-1) @ v_cpu.float()
    actual = out[:, :, indices.to(out.device), :].cpu().float()
    error = (actual - ref).abs()
    passed = torch.allclose(actual, ref, atol=0.002, rtol=0.01)
    print(f'shape={shape} rows={len(indices)} correct={passed} max_abs={error.max().item():.8g}', flush=True)
    if not passed:
        raise SystemExit('Accuracy check failed; performance comparison aborted')
    for _ in range(a.warmup):
        causal_attention(q, k, v)
    torch.npu.synchronize()
    times = []
    for _ in range(a.repeat):
        torch.npu.synchronize()
        start = time.perf_counter()
        causal_attention(q, k, v)
        torch.npu.synchronize()
        times.append((time.perf_counter() - start) * 1000)
result = {'shape': shape, 'correct': passed, 'rows_checked': indices.tolist(),
          'max_abs_error': error.max().item(), 'wall_ms': times,
          'note': 'synchronized full hybrid call; includes Python and multiple NPU kernels'}
with open(a.output, 'w') as f:
    json.dump(result, f, indent=2)
print('synchronized full-call wall time (ms):', times, flush=True)
print('saved:', a.output, flush=True)
