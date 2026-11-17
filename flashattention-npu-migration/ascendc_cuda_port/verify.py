"""NPU correctness test; this scalar reference implementation is not a benchmark."""
import argparse
import math
import torch
import torch_npu  # noqa: F401
import flash_ascendc_probe

p = argparse.ArgumentParser()
p.add_argument('--seq', type=int, default=8, choices=(8, 32, 65, 128))
p.add_argument('--heads', type=int, default=1, choices=(1, 2))
args = p.parse_args()
if not torch.npu.is_available():
    raise RuntimeError('NPU unavailable')
for seed in (1234, 4321):
    torch.manual_seed(seed)
    shape = (1, args.heads, args.seq, 128)
    q_cpu, k_cpu, v_cpu = [torch.randn(shape, dtype=torch.float16) for _ in range(3)]
    scores = (q_cpu.float() @ k_cpu.float().transpose(-2, -1)) / math.sqrt(128)
    mask = torch.ones((args.seq, args.seq), dtype=torch.bool).triu(1)
    expected = torch.softmax(scores.masked_fill(mask, float('-inf')), -1) @ v_cpu.float()
    q, k, v = [x.to('npu:0') for x in (q_cpu, k_cpu, v_cpu)]
    actual = flash_ascendc_probe.forward(q, k, v)
    torch.npu.synchronize()
    output = actual.cpu().float()
    err = (output - expected).abs()
    passed = torch.allclose(output, expected, atol=0.002, rtol=0.01)
    print(f'seed={seed} shape={shape} pass={passed} max_abs={err.max().item():.8g}', flush=True)
    if not passed:
        raise SystemExit('Failed independent CPU FP32 reference check')
print('All rows passed. This correctness-first prototype does not use Cube/MAC.')
