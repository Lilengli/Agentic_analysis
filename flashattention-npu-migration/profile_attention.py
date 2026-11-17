"""Capture one warmed-up attention call with Ascend AI Core counters."""
import argparse
import math
from pathlib import Path

import torch
import torch_npu
import torch.nn.functional as F

from flash_attention_ascend import kernel_function


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--impl', choices=('triton', 'sdpa', 'official'), required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--heads', type=int, default=2)
    parser.add_argument('--seq', type=int, default=128)
    parser.add_argument('--dim', type=int, default=128)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--calls', type=int, default=1,
                        help='number of consecutive calls captured by the profiler')
    parser.add_argument('--block-m', type=int, default=32)
    parser.add_argument('--block-n', type=int, default=32)
    parser.add_argument('--metrics', choices=('pipe', 'memory'), default='pipe',
                        help='collect pipeline activity or memory bandwidth counters')
    args = parser.parse_args()
    if min(args.batch, args.heads, args.seq, args.dim, args.warmup, args.calls,
           args.block_m, args.block_n) < 1:
        parser.error('shape, warmup, calls and block sizes must be positive')
    if args.impl == 'official' and (args.dim not in (16, 32, 64, 128, 256)
            or args.block_m < args.block_n or args.seq % args.block_m or args.seq % args.block_n):
        parser.error('official requires supported dim, block-m >= block-n and seq divisible by both blocks')
    if not torch.npu.is_available():
        raise RuntimeError('No NPU is visible')
    output = Path(args.output).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f'Output directory is not empty: {output}; choose a new path')
    output.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(1234)
    shape = (args.batch, args.heads, args.seq, args.dim)
    q, k, v = [torch.randn(shape, dtype=torch.float16).to('npu:0') for _ in range(3)]

    def triton_call():
        return kernel_function(q, k, v)

    def sdpa_call():
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)

    def official_call():
        from official_fused_attention import attention
        return attention(q, k, v, True, 1 / math.sqrt(args.dim), args.block_m, args.block_n)

    with torch.inference_mode():
        run = {'triton': triton_call, 'sdpa': sdpa_call, 'official': official_call}[args.impl]
        actual, expected = run(), sdpa_call()
        torch.npu.synchronize()
        max_err = (actual.float() - expected.float()).abs().max().item()
        passed = torch.allclose(actual, expected, atol=2e-2, rtol=2e-2)
        print(f'shape={shape} dtype=fp16 causal=True max_abs_error={max_err:.7g} match={passed}', flush=True)
        if not passed:
            raise SystemExit('Correctness check failed; profiling skipped')
        for _ in range(args.warmup):
            result = run()
        torch.npu.synchronize()

        config = torch_npu.profiler._ExperimentalConfig(
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            aic_metrics=(torch_npu.profiler.AiCMetrics.PipeUtilization if args.metrics == 'pipe'
                         else torch_npu.profiler.AiCMetrics.Memory),
        )
        with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.CPU,
                        torch_npu.profiler.ProfilerActivity.NPU],
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(output)),
            record_shapes=True,
            with_stack=False,
            experimental_config=config,
        ):
            for _ in range(args.calls):
                result = run()
            torch.npu.synchronize()

    print(f'Profiling completed: impl={args.impl}, directory={output}', flush=True)
    for path in sorted(output.rglob('kernel_details.csv')):
        print(f'kernel_details={path}')
    for path in sorted(output.rglob('trace_view.json')):
        print(f'trace_view={path}')


if __name__ == '__main__':
    main()
