"""Benchmark the balanced BN512 kernel and CANN FlashAttentionScore on the same NPU inputs."""
import argparse
import json
import math
from pathlib import Path
import statistics
import tempfile
import time


SEQUENCES = tuple(1 << exponent for exponent in range(8, 21))
DEFAULT_SEQUENCES = SEQUENCES[:7]
REFERENCE_KV_CHUNK = 16384
BATCH, HEADS, HEAD_DIM = 1, 2, 128
SCALE = 1 / math.sqrt(HEAD_DIM)
GENERATED = {
    'block_m': 128,
    'block_n': 512,
    'prefetch_pv_v': True,
    'cross_query': True,
    'trim_diagonal': True,
    'vector_normalize': False,
    'balance_queries': True,
}


def causal_reference(torch, q, k, v, indices):
    query = q.index_select(2, indices).float()
    row_max = torch.full_like(query[..., :1], -math.inf)
    denominator = torch.zeros_like(row_max)
    numerator = torch.zeros_like(query)
    for start in range(0, k.shape[2], REFERENCE_KV_CHUNK):
        stop = min(start + REFERENCE_KV_CHUNK, k.shape[2])
        scores = (query @ k[:, :, start:stop].float().transpose(-2, -1)) * SCALE
        causal = torch.arange(start, stop)[None, :] > indices[:, None]
        scores.masked_fill_(causal, -math.inf)
        new_max = torch.maximum(row_max, scores.amax(dim=-1, keepdim=True))
        # The first chunk contains key 0, so every sampled query has a finite maximum.
        correction = (row_max - new_max).exp()
        weights = scores.sub_(new_max).exp_()
        numerator = numerator * correction + weights @ v[:, :, start:stop].float()
        denominator = denominator * correction + weights.sum(dim=-1, keepdim=True)
        row_max = new_max
    return numerator / denominator


def check_output(torch, output, shape, indices, reference, label):
    if tuple(output.shape) != shape:
        raise RuntimeError(f'{label}: expected shape {shape}, got {tuple(output.shape)}')
    if not torch.isfinite(output).all().item():
        raise RuntimeError(f'{label}: output contains NaN or infinity')
    selected = output.index_select(2, indices).cpu().float()
    torch.testing.assert_close(selected, reference, atol=0.002, rtol=0.01,
                               msg=lambda message: f'{label}: {message}')
    return (selected - reference).abs().max().item()


def benchmark_sequence(torch, torch_npu, fused_cube_ext, mask, device, seq, args):
    shape = (BATCH, HEADS, seq, HEAD_DIM)
    blocks = min(args.max_blocks, BATCH * HEADS * (seq // GENERATED['block_m']))
    print(f'S={seq} blocks={blocks}  verifying both kernels', flush=True)
    torch.manual_seed(args.seed)
    cpu_qkv = [torch.randn(shape, dtype=torch.float16) for _ in range(3)]
    indices = torch.linspace(0, seq - 1, min(args.rows, seq)).round().long().unique()
    reference = causal_reference(torch, *cpu_qkv, indices)
    q, k, v = [tensor.to(device) for tensor in cpu_qkv]
    device_indices = indices.to(device)
    torch.npu.synchronize()
    del cpu_qkv

    def generated():
        return fused_cube_ext.forward(q, k, v, blocks=blocks, **GENERATED)

    def cann():
        return torch_npu.npu_fusion_attention(
            q, k, v, HEADS, 'BNSD', atten_mask=mask, scale=SCALE, keep_prob=1.0,
            pre_tockens=2147483647, next_tockens=0, inner_precise=0, sparse_mode=2,
        )[0]

    methods = {'generated': generated, 'cann': cann}
    stats = {}
    for name, call in methods.items():
        stats[name] = {
            'first_max_abs_error': check_output(
                torch, call(), shape, device_indices, reference, f'S={seq} {name} first output'),
            'event_samples_us': [],
            'wall_samples_us': [],
        }
    outputs = {}
    for _ in range(args.warmup):
        for name, call in methods.items():
            outputs[name] = call()
    torch.npu.synchronize()

    for index in range(args.repeat):
        order = methods if index % 2 == 0 else reversed(methods)
        for name in order:
            start = torch.npu.Event(enable_timing=True)
            stop = torch.npu.Event(enable_timing=True)
            wall_start = time.perf_counter_ns()
            start.record()
            out = methods[name]()
            stop.record()
            stop.synchronize()
            stats[name]['wall_samples_us'].append((time.perf_counter_ns() - wall_start) / 1000)
            stats[name]['event_samples_us'].append(start.elapsed_time(stop) * 1000)
            outputs[name] = out
    for name, item in stats.items():
        item['last_max_abs_error'] = check_output(
            torch, outputs[name], shape, device_indices, reference, f'S={seq} {name} last timed output')
        item['event_median_us'] = statistics.median(item['event_samples_us'])
        item['wall_median_us'] = statistics.median(item['wall_samples_us'])
        if not math.isfinite(item['event_median_us']) or item['event_median_us'] <= 0:
            raise RuntimeError(f'S={seq} {name}: invalid NPU Event timing')

    return {
        'seq': seq, 'generated_blocks': blocks, 'query_rows_checked': indices.tolist(),
        'correct': True, 'methods': stats,
        'generated_speedup_vs_cann': stats['cann']['event_median_us'] / stats['generated']['event_median_us'],
    }


def save_report(path, report):
    # Keep previously verified lengths available if a later run is interrupted.
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, delete=False) as file:
        temporary = Path(file.name)
        try:
            json.dump(report, file, indent=2, allow_nan=False)
            file.write('\n')
            file.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seqs', nargs='+', type=int, choices=SEQUENCES, default=DEFAULT_SEQUENCES,
                        help='sequence lengths, default 256 through 16384')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--max-blocks', type=int, choices=range(1, 25), default=24)
    parser.add_argument('--rows', type=int, default=64)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--repeat', type=int, default=20)
    parser.add_argument('--output', type=Path, default=Path('flash_attention_256_to_16k.json'))
    args = parser.parse_args()
    if args.rows < 1 or args.repeat < 1 or args.warmup < 0 or args.device < 0:
        parser.error('rows and repeat must be positive; warmup and device must be nonnegative')
    if not args.output.parent.is_dir():
        parser.error(f'output parent directory does not exist: {args.output.parent}')

    import torch
    import torch_npu
    import fused_cube_ext
    from run import verify_build

    if not torch.npu.is_available():
        raise RuntimeError('No NPU is visible')
    manifest = verify_build()
    device = torch.device(f'npu:{args.device}')
    torch.npu.set_device(device)
    report = {
        'device': torch.npu.get_device_name(args.device),
        'torch': str(torch.__version__), 'torch_npu': str(torch_npu.__version__),
        'generated_module_sha256': manifest['module_sha256'],
        'batch': BATCH, 'heads': HEADS, 'head_dim': HEAD_DIM,
        'dtype': 'fp16', 'layout': 'BNSD', 'causal': True,
        'generated_config': GENERATED, 'generated_max_blocks': args.max_blocks,
        'cann_api': 'torch_npu.npu_fusion_attention',
        'reference_kv_chunk': REFERENCE_KV_CHUNK,
        'warmup': args.warmup, 'repeat': args.repeat, 'seed': args.seed,
        'requested_sequences': list(args.seqs), 'results': [],
    }
    print(f"device={report['device']} dtype=fp16 causal=True B={BATCH} H={HEADS} D={HEAD_DIM}", flush=True)
    with torch.inference_mode():
        mask = torch.ones((2048, 2048), dtype=torch.bool).triu_(1).to(device)
        for seq in args.seqs:
            row = benchmark_sequence(torch, torch_npu, fused_cube_ext, mask, device, seq, args)
            report['results'].append(row)
            save_report(args.output, report)
            generated, cann = (row['methods'][name] for name in ('generated', 'cann'))
            print(f"S={seq:5}  generated={generated['event_median_us']:9.2f} us  "
                  f"CANN={cann['event_median_us']:9.2f} us  "
                  f"CANN/generated={row['generated_speedup_vs_cann']:.3f}x", flush=True)
    print(f'Saved {args.output}', flush=True)


if __name__ == '__main__':
    main()
