"""Check and compare three causal attention implementations on one NPU.

Measures complete calls with NPU events, not individual kernel duration.
"""
import argparse
import json
import math
import platform
import random
import statistics
import sys
import time
import traceback
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seq', type=int, default=128)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--heads', type=int, default=2)
    parser.add_argument('--dim', type=int, default=128)
    parser.add_argument('--block-m', type=int, default=32)
    parser.add_argument('--block-n', type=int, default=32)
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--repeats', type=int, default=100)
    parser.add_argument('--reference-rows', type=int, default=32,
                        help='query rows checked for sequences longer than 512')
    parser.add_argument('--output', default='attention_compare_results.json')
    a = parser.parse_args()
    if min(a.seq, a.batch, a.heads, a.dim, a.block_m, a.block_n, a.warmup, a.repeats, a.reference_rows) < 1:
        parser.error('all numeric options must be positive')
    if a.dim not in (16, 32, 64, 128, 256):
        parser.error('official example requires dim in {16,32,64,128,256}')
    if a.seq % a.block_m or a.seq % a.block_n or a.block_m < a.block_n:
        parser.error('use seq divisible by block-m and block-n, and block-m >= block-n')

    import torch
    import torch_npu
    import torch.nn.functional as F
    import triton
    if not torch.npu.is_available():
        raise RuntimeError('NPU is not available')

    result = {'config': vars(a), 'device': str(torch.npu.get_device_name(0)),
              'versions': {'python': sys.version, 'platform': platform.platform(),
                           'torch': torch.__version__, 'torch_npu': torch_npu.__version__,
                           'triton': triton.__version__}, 'implementations': {}}
    shape = (a.batch, a.heads, a.seq, a.dim)
    torch.manual_seed(1234)
    q_cpu, k_cpu, v_cpu = [torch.randn(shape, dtype=torch.float16) for _ in range(3)]
    # Check all rows at small lengths and evenly spaced rows at long lengths.
    # Avoid allocating a full 16K x 16K CPU score matrix.
    count = a.seq if a.seq <= 512 else min(a.seq, a.reference_rows)
    indices = torch.linspace(0, a.seq - 1, count).round().long().unique()
    scores = q_cpu.float().index_select(-2, indices) @ k_cpu.float().transpose(-2, -1) / math.sqrt(a.dim)
    mask = torch.arange(a.seq)[None, :] > indices[:, None]
    reference = torch.softmax(scores.masked_fill(mask, float('-inf')), dim=-1) @ v_cpu.float()
    result['reference'] = {'dtype': 'float32_cpu', 'query_rows_checked': indices.tolist(),
                           'total_query_rows': a.seq, 'atol': 0.02, 'rtol': 0.02}
    q, k, v = [t.to('npu:0') for t in (q_cpu, k_cpu, v_cpu)]
    torch.npu.synchronize()

    methods = {'sdpa': lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)}
    for name, module, attr in [('migrated', 'flash_attention_ascend', 'kernel_function'),
                               ('official', 'official_fused_attention', 'attention')]:
        try:
            fn = getattr(__import__(module), attr)
            if name == 'official':
                methods[name] = lambda fn=fn: fn(q, k, v, True, 1 / math.sqrt(a.dim), a.block_m, a.block_n)
            else:
                methods[name] = lambda fn=fn: fn(q, k, v)
        except Exception as exc:
            result['implementations'][name] = {'status': 'import_error', 'error': repr(exc)}
            traceback.print_exc()

    valid = []
    with torch.inference_mode():
        for name, fn in methods.items():
            try:
                out = fn()
                torch.npu.synchronize()
                selected = out.cpu().float().index_select(-2, indices)
                diff = (selected - reference).abs()
                passed = torch.allclose(selected, reference, atol=0.02, rtol=0.02)
                entry = {'status': 'correct' if passed else 'incorrect',
                         'max_abs_error_vs_fp32_cpu': diff.max().item()}
                result['implementations'][name] = entry
                print(f'{name}: {entry}; checked {len(indices)}/{a.seq} query rows', flush=True)
                if not passed:
                    continue
                for _ in range(a.warmup):
                    out = fn()
                torch.npu.synchronize()
                entry['event_us'] = []
                entry['wall_us'] = []
                valid.append(name)
            except Exception as exc:
                result['implementations'][name] = {'status': 'error', 'error': repr(exc)}
                traceback.print_exc()

        rng = random.Random(1234)
        for _ in range(a.repeats):
            order = valid[:]
            rng.shuffle(order)
            for name in order:
                entry = result['implementations'][name]
                try:
                    start = torch.npu.Event(enable_timing=True)
                    end = torch.npu.Event(enable_timing=True)
                    t0 = time.perf_counter_ns()
                    start.record()
                    out = methods[name]()
                    end.record()
                    end.synchronize()
                    elapsed_wall = (time.perf_counter_ns() - t0) / 1000
                    entry['event_us'].append(start.elapsed_time(end) * 1000)
                    entry['wall_us'].append(elapsed_wall)
                except Exception as exc:
                    entry.update(status='timing_error', error=repr(exc))
                    traceback.print_exc()
            valid = [n for n in valid if result['implementations'][n]['status'] == 'correct']

    for name, entry in result['implementations'].items():
        if entry.get('status') != 'correct' or not entry.get('event_us'):
            continue
        for key in ('event_us', 'wall_us'):
            values = entry[key]
            entry[key + '_summary'] = {'min': min(values), 'median': statistics.median(values),
                                       'mean': statistics.mean(values), 'max': max(values)}
        print(f"{name}: NPU-event median={entry['event_us_summary']['median']:.3f} us; "
              f"synchronized wall median={entry['wall_us_summary']['median']:.3f} us", flush=True)

    Path(a.output).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'Results saved to {a.output}', flush=True)
    if any(e['status'] != 'correct' for e in result['implementations'].values()):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
