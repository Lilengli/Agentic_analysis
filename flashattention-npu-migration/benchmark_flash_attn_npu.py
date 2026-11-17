"""Compare flash-attention-npu v2 and SDPA with matching causal FP16 inputs."""
import argparse
import json
import math
import platform
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, help='flash-attention-npu source checkout')
    parser.add_argument('--seq', type=int, default=128)
    parser.add_argument('--rows', type=int, default=64)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--repeats', type=int, default=30)
    parser.add_argument('--output', default='flash_attn_npu_result.json')
    args = parser.parse_args()
    if min(args.seq, args.rows, args.warmup, args.repeats) < 1:
        parser.error('seq, rows, warmup and repeats must be positive')
    source = Path(args.source).expanduser().resolve()
    if not (source / 'flash_attn_npu' / '__init__.py').is_file():
        parser.error(f'not a flash-attention-npu source tree: {source}')
    sys.path.insert(0, str(source))

    import torch
    import torch_npu
    import torch.nn.functional as F
    from flash_attn_npu import flash_attn_func
    import flash_attn_npu
    if not torch.npu.is_available():
        raise RuntimeError('No NPU is visible')

    rev = subprocess.run(['git', '-C', str(source), 'rev-parse', 'HEAD'],
                         capture_output=True, text=True, check=False)
    result = {'config': vars(args), 'source_resolved': str(source),
              'source_commit': rev.stdout.strip() if rev.returncode == 0 else None,
              'versions': {'python': sys.version, 'platform': platform.platform(),
                           'torch': torch.__version__, 'torch_npu': torch_npu.__version__,
                           'flash_attn_npu': getattr(flash_attn_npu, '__version__', None)},
              'device': str(torch.npu.get_device_name(0)), 'methods': {}}
    torch.manual_seed(1234)
    shape = (1, 2, args.seq, 128)
    q_cpu, k_cpu, v_cpu = [torch.randn(shape, dtype=torch.float16) for _ in range(3)]
    count = args.seq if args.seq <= 512 else min(args.seq, args.rows)
    rows = torch.linspace(0, args.seq-1, count).round().long().unique()
    score = q_cpu.float().index_select(2, rows) @ k_cpu.float().transpose(-2, -1) / math.sqrt(128)
    mask = torch.arange(args.seq)[None, :] > rows[:, None]
    reference = torch.softmax(score.masked_fill(mask, float('-inf')), -1) @ v_cpu.float()
    result['reference'] = {'rows_checked': rows.tolist(), 'total_query_rows': args.seq,
                           'atol': 0.002, 'rtol': 0.01}
    q, k, v = [x.to('npu:0') for x in (q_cpu, k_cpu, v_cpu)]
    # Preconvert BSND once, then use the identical numerical tensors for both paths.
    qb, kb, vb = [x.transpose(1, 2).contiguous() for x in (q, k, v)]
    torch.npu.synchronize()

    def flash_preconverted():
        return flash_attn_func(qb, kb, vb, dropout_p=0.0, softmax_scale=1/math.sqrt(128), causal=True)

    def flash_with_layout():
        qx, kx, vx = [x.transpose(1, 2).contiguous() for x in (q, k, v)]
        return flash_attn_func(qx, kx, vx, dropout_p=0.0,
                               softmax_scale=1/math.sqrt(128), causal=True).transpose(1, 2).contiguous()

    methods = {
        'sdpa_bnsd': (lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0), False),
        'flash_npu_preconverted_bsnd': (flash_preconverted, True),
        'flash_npu_with_layout_conversion': (flash_with_layout, False),
    }
    with torch.inference_mode():
        valid = []
        for name, (fn, is_bsnd) in methods.items():
            item = result['methods'][name] = {}
            try:
                output = fn()
                torch.npu.synchronize()
                if is_bsnd:
                    output = output.transpose(1, 2)
                selected = output.cpu().float().index_select(2, rows)
                diff = selected - reference
                item['correct'] = bool(torch.allclose(selected, reference, atol=0.002, rtol=0.01))
                item['max_abs_error'] = diff.abs().max().item()
                item['normalized_rmse'] = (diff.square().mean().sqrt() /
                                           reference.square().mean().sqrt().clamp_min(1e-30)).item()
                print(f'{name}: {item}; checked {len(rows)}/{args.seq} query rows', flush=True)
                if item['correct']:
                    for _ in range(args.warmup):
                        output = fn()
                    torch.npu.synchronize()
                    item['event_us'], item['wall_us'] = [], []
                    valid.append(name)
            except Exception as exc:
                item['correct'] = False
                item['error'] = repr(exc)
                print(f'{name}: ERROR {exc!r}', flush=True)
        rng = random.Random(1234)
        for _ in range(args.repeats):
            order = valid[:]
            rng.shuffle(order)
            for name in order:
                start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
                t0 = time.perf_counter_ns()
                start.record()
                output = methods[name][0]()
                end.record()
                end.synchronize()
                result['methods'][name]['event_us'].append(start.elapsed_time(end) * 1000)
                result['methods'][name]['wall_us'].append((time.perf_counter_ns() - t0) / 1000)
        for name in valid:
            item = result['methods'][name]
            item['event_median_us'] = statistics.median(item['event_us'])
            item['wall_median_us'] = statistics.median(item['wall_us'])
            print(f"{name}: event median={item['event_median_us']:.3f} us, "
                  f"synchronized wall median={item['wall_median_us']:.3f} us", flush=True)
    Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'saved {args.output}', flush=True)
    if len(valid) != len(methods):
        raise SystemExit('Some methods failed correctness or execution; see output JSON')


if __name__ == '__main__':
    main()
