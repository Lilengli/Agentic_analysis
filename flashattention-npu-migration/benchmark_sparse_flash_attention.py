"""Check and benchmark BSND sparse MLA single-query decode on one NPU.

The Triton baseline is kernel.py from a632589030's GitCode repository
triton-ascend-opdev-2026-sparse-flash-attention, revision 1e9acc4.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time


def inputs(torch, seq_kv, topk, dtype, seed):
    rng = torch.Generator().manual_seed(seed)
    q = torch.randn(1, 1, 128, 512, generator=rng, dtype=dtype)
    qr = torch.randn(1, 1, 128, 64, generator=rng, dtype=dtype)
    k = torch.randn(1, seq_kv, 1, 512, generator=rng, dtype=dtype)
    kr = torch.randn(1, seq_kv, 1, 64, generator=rng, dtype=dtype)
    selected = torch.randperm(seq_kv, generator=rng)[:topk].sort().values
    indices = selected.to(torch.int32).reshape(1, 1, 1, topk)
    scale = 1 / math.sqrt(576)
    chosen_k = k[0, selected, 0].float()
    chosen_kr = kr[0, selected, 0].float()
    scores = q[0, 0].float() @ chosen_k.T + qr[0, 0].float() @ chosen_kr.T
    reference = (torch.softmax(scores * scale, dim=-1) @ chosen_k).reshape(q.shape)
    return q, qr, k, kr, indices, scale, reference


def check(torch, output, reference):
    if output.shape != reference.shape:
        raise RuntimeError(f'unexpected result shape {tuple(output.shape)}')
    output = output.cpu().float()
    error = (output - reference).abs()
    matched = torch.isclose(output, reference, atol=0.05, rtol=0.05).float().mean().item()
    finite = bool(torch.isfinite(output).all())
    return {
        'correct': finite and matched >= 0.995, 'finite': finite, 'matched_ratio': matched,
        'max_abs_error': error.max().item() if finite else None,
    }


def benchmark(torch, call, warmup, repeat):
    for _ in range(warmup):
        call()
    torch.npu.synchronize()
    events, wall = [], []
    for _ in range(repeat):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        before = time.perf_counter_ns()
        start.record()
        call()
        end.record()
        torch.npu.synchronize()
        wall.append((time.perf_counter_ns() - before) / 1000)
        elapsed = start.elapsed_time(end) * 1000
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise RuntimeError(f'invalid NPU Event duration: {elapsed}')
        events.append(elapsed)
    return {
        'event_median_us': statistics.median(events), 'wall_median_us': statistics.median(wall),
        'event_samples_us': events, 'wall_samples_us': wall,
    }


def profile_once(profiler, torch, call, directory):
    with profiler.profile(
        activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.NPU],
        on_trace_ready=profiler.tensorboard_trace_handler(str(directory)),
    ):
        call()
        torch.npu.synchronize()
    return [str(path) for path in sorted(directory.rglob('kernel_details.csv'))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--impl', choices=('both', 'triton', 'cann'), default='both')
    parser.add_argument('--kernel-dir', type=Path, help='directory containing the upstream Triton kernel.py')
    parser.add_argument('--seq-kv', type=int, default=16384)
    parser.add_argument('--topk', type=int, choices=(128, 256, 512, 1024, 2048), default=2048)
    parser.add_argument('--dtype', choices=('bf16', 'fp16'), default='bf16')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--repeat', type=int, default=20)
    parser.add_argument('--profile-dir', type=Path, help='new directory for separate one-call device profiles')
    parser.add_argument('--output', type=Path, default=Path('sfa_decode.json'))
    args = parser.parse_args()
    if args.seq_kv < args.topk or args.repeat < 1 or args.warmup < 0:
        parser.error('require seq-kv >= topk, repeat >= 1 and warmup >= 0')
    if args.profile_dir and args.profile_dir.exists():
        parser.error('profile-dir already exists; choose a fresh path')

    import torch
    import torch_npu

    device = torch.device('npu:0')
    torch.npu.set_device(device)
    dtype = torch.bfloat16 if args.dtype == 'bf16' else torch.float16
    q, qr, k, kr, indices, scale, reference = inputs(torch, args.seq_kv, args.topk, dtype, args.seed)
    q, qr, k, kr, indices = (tensor.to(device) for tensor in (q, qr, k, kr, indices))
    report = {
        'device': torch.npu.get_device_name(device), 'torch': str(torch.__version__),
        'torch_npu': getattr(torch_npu, '__version__', None),
        'cann_path': os.environ.get('ASCEND_TOOLKIT_HOME') or os.environ.get('ASCEND_HOME_PATH'),
        'shape': {'batch': 1, 'seq_q': 1, 'seq_kv': args.seq_kv, 'q_heads': 128, 'kv_heads': 1,
                  'topk': args.topk, 'dtype': args.dtype},
        'seed': args.seed, 'value_aliases_key': True, 'warmup': args.warmup, 'repeat': args.repeat,
        'results': {},
    }
    print(f"device={report['device']} shape={report['shape']}", flush=True)

    def triton_runner():
        if args.kernel_dir:
            kernel_dir = args.kernel_dir.resolve()
            if not (kernel_dir / 'kernel.py').is_file():
                raise ImportError(f'kernel.py is missing from {kernel_dir}')
            sys.path.insert(0, str(kernel_dir))
        import kernel

        def run():
            return kernel.run(q, k, k, indices, scale, qr, kr)

        report['triton'] = kernel.triton.__version__
        report['triton_kernel_sha256'] = hashlib.sha256(Path(kernel.__file__).read_bytes()).hexdigest()
        return run

    def cann_call():
        return torch_npu.npu_sparse_flash_attention(
            q, k, k, indices, scale, query_rope=qr, key_rope=kr,
            sparse_mode=0, attention_mode=2, layout_query='BSND', layout_kv='BSND',
            sparse_block_size=1,
        )[0]

    with torch.inference_mode():
        for name in ('triton', 'cann'):
            if args.impl not in ('both', name):
                continue
            print(f'[{name}] first call and CPU FP32 check', flush=True)
            result = {}
            try:
                call = triton_runner() if name == 'triton' else cann_call
                result.update(check(torch, call(), reference))
                if result['correct']:
                    result.update(benchmark(torch, call, args.warmup, args.repeat))
            except (ImportError, AttributeError, RuntimeError, TypeError) as exc:
                result.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            if args.profile_dir and result.get('correct') and 'error' not in result:
                try:
                    result['profile_csv'] = profile_once(torch_npu.profiler, torch, call, args.profile_dir / name)
                except (AttributeError, RuntimeError) as exc:
                    result['profile_error'] = str(exc)
            report['results'][name] = result
            print(name, {key: value for key, value in result.items() if not key.endswith('_samples_us')}, flush=True)
            args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')

    print(f'Saved {args.output}', flush=True)
    return int(any(not value.get('correct') or 'error' in value for value in report['results'].values()))


if __name__ == '__main__':
    raise SystemExit(main())
