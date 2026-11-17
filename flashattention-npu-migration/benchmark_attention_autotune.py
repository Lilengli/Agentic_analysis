"""Validate and benchmark the experimental Ascend autotune implementation."""
import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seq', type=int, default=128)
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--repeats', type=int, default=30)
    p.add_argument('--tiles', nargs='+', default=None,
                   help='candidate tiles, e.g. 128x64 128x128 256x64')
    p.add_argument('--output', default='autotune_results.json')
    a = p.parse_args()
    if min(a.seq, a.warmup, a.repeats) < 1:
        p.error('seq, warmup and repeats must be positive')
    tiles = None
    if a.tiles is not None:
        try:
            tiles = tuple(tuple(int(x) for x in t.lower().split('x')) for t in a.tiles)
        except ValueError:
            p.error('tile format must be MxN, e.g. 128x64')
        if any(len(x) != 2 or x[0] not in (16,32,64,128,256) or x[1] not in (16,32,64,128,256)
               or x[0] < x[1] for x in tiles) or len(set(tiles)) != len(tiles):
            p.error('tiles must be unique, powers of two from 16 to 256, and M >= N')

    import torch
    import torch_npu
    import torch.nn.functional as F
    import triton
    from flash_attention_ascend import kernel_function
    from flash_attention_ascend_autotune import autotuned_attention, tuning_report, _reference
    if not torch.npu.is_available():
        raise RuntimeError('No NPU is visible')
    torch.manual_seed(1234)
    shape = (1, 2, a.seq, 128)
    inputs = [torch.randn(shape, dtype=torch.float16).to('npu:0') for _ in range(3)]
    q, k, v = inputs
    results = {'config': vars(a), 'device': str(torch.npu.get_device_name(0)),
               'versions': {'python': sys.version, 'torch': torch.__version__,
                            'torch_npu': torch_npu.__version__, 'triton': triton.__version__},
               'methods': {}}
    with torch.inference_mode():
        t0 = time.monotonic()
        auto = autotuned_attention(q, k, v, candidates=tiles)
        torch.npu.synchronize()
        results['first_call_seconds'] = time.monotonic() - t0
        results['autotune'] = tuning_report(q, k, v, candidates=tiles)
        print('calibration result:', json.dumps(results['autotune'], ensure_ascii=False), flush=True)
        # Check the selected configuration against a second, different input of
        # the same shape; autotune cache should be reused without recalibration.
        torch.manual_seed(4321)
        q2, k2, v2 = [torch.randn(shape, dtype=torch.float16).to('npu:0') for _ in range(3)]
        rows, reference = _reference(q2, k2, v2, 64)
        out = autotuned_attention(q2, k2, v2, candidates=tiles)
        torch.npu.synchronize()
        selected = out.cpu().float().index_select(2, rows)
        passed = bool(torch.allclose(selected, reference, atol=0.002, rtol=0.01))
        results['second_input'] = {'correct': passed, 'rows_checked': len(rows),
                                   'max_abs_error': (selected - reference).abs().max().item()}
        print('second input:', results['second_input'], flush=True)
        if not passed:
            Path(a.output).write_text(json.dumps(results, indent=2), encoding='utf-8')
            raise SystemExit('Selected config failed second-input check; timing skipped')

        methods = {'autotune': lambda: autotuned_attention(q, k, v, candidates=tiles),
                   'sdpa': lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)}
        for bm, bn in ((32, 32), (128, 64)):
            if any(x['block_m'] == bm and x['block_n'] == bn and x['correct']
                   for x in results['autotune']['candidates']):
                methods[f'manual_{bm}x{bn}'] = lambda bm=bm, bn=bn: kernel_function(q, k, v, block_m=bm, block_n=bn)
        for name, fn in methods.items():
            for _ in range(a.warmup):
                out = fn()
            torch.npu.synchronize()
            results['methods'][name] = {'event_us': [], 'wall_us': []}
        rng = random.Random(1234)
        for _ in range(a.repeats):
            names = list(methods)
            rng.shuffle(names)
            for name in names:
                start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
                host_start = time.perf_counter_ns()
                start.record()
                out = methods[name]()
                end.record()
                end.synchronize()
                results['methods'][name]['event_us'].append(start.elapsed_time(end) * 1000)
                results['methods'][name]['wall_us'].append((time.perf_counter_ns() - host_start) / 1000)
        for name, item in results['methods'].items():
            item['event_median_us'] = statistics.median(item['event_us'])
            item['wall_median_us'] = statistics.median(item['wall_us'])
            print(f"{name}: event median={item['event_median_us']:.3f} us; "
                  f"synchronized wall median={item['wall_median_us']:.3f} us", flush=True)
    Path(a.output).write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(f'saved {a.output}', flush=True)


if __name__ == '__main__':
    main()
