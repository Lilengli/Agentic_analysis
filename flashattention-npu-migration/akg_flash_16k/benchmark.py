"""Benchmark a generated ModelNew against causal SDPA on the same NPU inputs."""
import argparse
import hashlib
import importlib.util
import json
import math
import random
import statistics
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--candidate', required=True, help='AKG-generated Python file')
    p.add_argument('--seq', type=int, default=16384)
    p.add_argument('--rows', type=int, default=64)
    p.add_argument('--warmup', type=int, default=20)
    p.add_argument('--repeat', type=int, default=100)
    p.add_argument('--output', default='akg_compare_16k.json')
    a = p.parse_args()
    if min(a.seq, a.rows, a.warmup, a.repeat) < 1:
        p.error('numeric options must be positive')
    import torch
    import torch_npu
    import torch.nn.functional as F
    if not torch.npu.is_available():
        raise RuntimeError('NPU unavailable')
    path = Path(a.candidate).resolve()
    spec = importlib.util.spec_from_file_location('akg_candidate', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.ModelNew().eval()
    shape = (1, 2, a.seq, 128)
    torch.manual_seed(1234)
    cpu = [torch.randn(shape, dtype=torch.float16) for _ in range(3)]
    idx = torch.linspace(0, a.seq - 1, min(a.rows, a.seq)).round().long().unique()
    scores = cpu[0].float().index_select(-2, idx) @ cpu[1].float().transpose(-2, -1) / math.sqrt(128)
    scores.masked_fill_(torch.arange(a.seq)[None, :] > idx[:, None], float('-inf'))
    ref = torch.softmax(scores, dim=-1) @ cpu[2].float()
    q, k, v = [x.to('npu:0') for x in cpu]
    methods = {'sdpa': lambda: F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True),
               'akg': lambda: model(q, k, v)}
    data = {'shape': shape, 'dtype': 'float16', 'layout': 'BNSD', 'causal': True,
            'dropout_p': 0.0, 'seed': 1234, 'rows_checked': idx.tolist(),
            'device': torch.npu.get_device_name(0), 'torch': torch.__version__,
            'torch_npu': torch_npu.__version__, 'candidate': str(path),
            'candidate_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'timing': 'NPU event complete call and synchronized host wall; not individual kernel',
            'results': {}}
    try:
        with torch.inference_mode():
            for name, fn in methods.items():
                out = fn()
                torch.npu.synchronize()
                if tuple(out.shape) != shape or out.dtype != torch.float16 or out.device != q.device:
                    raise RuntimeError(f'{name}: unexpected output shape/dtype/device')
                sample = out.cpu().float().index_select(-2, idx)
                correct = bool(torch.isfinite(sample).all() and torch.allclose(sample, ref, atol=0.002, rtol=0.01))
                error = (sample - ref).abs().max().item()
                data['results'][name] = {'correct_sampled_rows': correct, 'max_abs_error': error,
                                         'event_us': [], 'wall_us': []}
                print(f'{name}: correct={correct}, max_abs_error={error:.8g}', flush=True)
                if not correct:
                    raise RuntimeError(f'{name} failed sampled-row reference check')
                for _ in range(a.warmup):
                    fn()
                torch.npu.synchronize()
            rng = random.Random(1234)
            for _ in range(a.repeat):
                order = list(methods)
                rng.shuffle(order)
                for name in order:
                    start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
                    t0 = time.perf_counter_ns()
                    start.record()
                    out = methods[name]()
                    end.record()
                    end.synchronize()
                    data['results'][name]['wall_us'].append((time.perf_counter_ns() - t0) / 1000)
                    data['results'][name]['event_us'].append(start.elapsed_time(end) * 1000)
            for name, entry in data['results'].items():
                entry['event_median_us'] = statistics.median(entry['event_us'])
                entry['wall_median_us'] = statistics.median(entry['wall_us'])
                print(f"{name}: event={entry['event_median_us']:.3f} us, wall={entry['wall_median_us']:.3f} us")
    except Exception as exc:
        data['error'] = repr(exc)
        raise
    finally:
        Path(a.output).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
        print('Results saved:', a.output)


if __name__ == '__main__':
    main()
