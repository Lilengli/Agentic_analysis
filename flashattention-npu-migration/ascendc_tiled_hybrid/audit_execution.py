"""Source-derived memory accounting, optionally accompanied by a real NPU trace.

Logical traffic estimates are not measured HBM traffic: caches and backend
layouts can change physical traffic. Run --profile on the target for evidence
of kernel launches. This script never calls the hybrid a single fused kernel.
"""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path


def memory_accounting(batch, heads, seq, dim=128, tile=128, state_update='inplace'):
    if min(batch, heads, seq) < 1 or seq % tile:
        raise ValueError('positive shape and sequence divisible by tile required')
    blocks = seq // tile
    active_rows = batch * heads * sum(range(tile, seq + 1, tile))
    score_elements = active_rows * tile
    report = {
        'shape': [batch, heads, seq, dim],
        'state_update': state_update,
        'online_softmax': True,
        'single_fused_kernel': False,
        'score_and_probability_on_chip_only': False,
        'kv_blocks': blocks,
        'custom_stage_calls_expected': blocks,
        'framework_matmul_calls_expected': 2 * blocks,
        'peak_single_score_or_probability_tensor_bytes': batch * heads * seq * tile * 2,
        'score_and_probability_logical_read_write_bytes': score_elements * 2 * 4,
        'full_quadratic_score_tensor_bytes_if_allocated': batch * heads * seq * seq * 2,
        'state_concat_calls': 0 if state_update == 'inplace' else 3 * blocks,
        'state_concat_output_bytes': 0 if state_update == 'inplace' else blocks * batch * heads * seq * (dim + 2) * 4,
        'scope': 'source-derived logical tensor traffic; excludes other intermediates, caches, layouts and backend workspace; not a measured HBM count',
    }
    return report


def capture(args, output):
    import torch
    import torch_npu
    import torch.nn.functional as F
    from tiled_attention import causal_attention

    trace_dir = output / 'npu_trace'
    if trace_dir.exists() and any(trace_dir.iterdir()):
        raise RuntimeError('npu_trace is not empty; select a new --output directory')
    torch.manual_seed(1234)
    shape = (args.batch, args.heads, args.seq, 128)
    q, k, v = [torch.randn(shape, dtype=torch.float16).to('npu:0') for _ in range(3)]
    with torch.inference_mode():
        actual = causal_attention(q, k, v, state_update=args.state_update)
        ref = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.)
        torch.npu.synchronize()
        if not torch.allclose(actual, ref, atol=.002, rtol=.01):
            raise RuntimeError('full-output comparison against NPU SDPA failed; trace skipped')
        config = torch_npu.profiler._ExperimentalConfig(
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        )
        with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(trace_dir)),
            record_shapes=True, with_stack=False, experimental_config=config,
        ):
            actual = causal_attention(q, k, v, state_update=args.state_update)
            torch.npu.synchronize()
    files = sorted(trace_dir.rglob('kernel_details.csv'))
    if not files:
        print('No kernel_details.csv exported; inspect profiler output before drawing conclusions.')
        return {'kernel_csv_files': [], 'measured_custom_stage_kernels': None}
    names = Counter()
    for path in files:
        with path.open(encoding='utf-8-sig', newline='') as f:
            for row in csv.DictReader(f):
                name = row.get('Name') or row.get('Task Name')
                if not name:
                    raise ValueError(f'Kernel name field missing in {path}')
                names[name] += 1
    count = sum(n for name, n in names.items() if 'online_stage_kernel' in name)
    print('NPU kernel counts:')
    for name, n in names.most_common():
        print(f'{n:6d}  {name}')
    if count != args.seq // 128:
        print('WARNING: stage count differs from expected; check symbols/capture before interpreting.')
    return {'kernel_csv_files': [str(p) for p in files],
            'measured_custom_stage_kernels': count, 'kernel_name_counts': dict(names)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seq', type=int, default=16384)
    p.add_argument('--batch', type=int, default=1)
    p.add_argument('--heads', type=int, default=2)
    p.add_argument('--state-update', choices=('concat', 'inplace'), default='inplace')
    p.add_argument('--output', default='algorithm_audit')
    p.add_argument('--profile', action='store_true')
    a = p.parse_args()
    report = memory_accounting(a.batch, a.heads, a.seq, state_update=a.state_update)
    output = Path(a.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if a.profile:
        report['device_capture'] = capture(a, output)
    (output / 'algorithm_audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
