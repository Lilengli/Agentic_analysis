"""Verify, benchmark and optionally profile the one-launch mixed Cube kernel."""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

import torch
import torch_npu
import fused_cube_ext


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seq', type=int, default=16384)
    p.add_argument('--batch', type=int, default=1)
    p.add_argument('--heads', type=int, default=2)
    p.add_argument('--block-m', type=int, choices=(32, 64, 128), default=64)
    p.add_argument('--block-n', type=int, choices=(64, 128, 512), default=128)
    p.add_argument('--blocks', type=int, default=24, help='AIC groups; each has two AIV')
    pv = p.add_mutually_exclusive_group()
    pv.add_argument('--pv-row32', action='store_true', help='early PV with per-tile Fixpipe')
    pv.add_argument('--pv-row32-batch-fix', action='store_true',
                    help='early PV with one Fixpipe per 32 output rows')
    pv.add_argument('--prioritize-p-writeback', action='store_true',
                    help='for BM128/BN512, wait for the oldest P before refilling the QK queue')
    p.add_argument('--rows', type=int, default=64)
    p.add_argument('--seeds', type=int, nargs='+', default=(1234, 4321))
    p.add_argument('--warmup', type=int, default=1)
    p.add_argument('--repeat', type=int, default=3)
    p.add_argument('--verify-only', action='store_true')
    p.add_argument('--profile', help='new output directory for one-call NPU profile')
    p.add_argument('--profile-metrics', choices=('pipe', 'memory-access'), default='pipe')
    p.add_argument('--output', default='fused_cube_16k.json')
    a = p.parse_args()
    supported_tile = ((a.block_m in (32, 64, 128) and a.block_n in (64, 128)
                       and a.block_m <= a.block_n)
                      or (a.block_m == 128 and a.block_n == 512))
    if not (0 < a.seq <= 16384 and a.seq % a.block_n == 0 and 0 < a.blocks <= 24
            and supported_tile and min(a.rows, a.batch, a.heads, a.repeat) > 0 and a.warmup >= 0):
        p.error('invalid shape/count: supported tiles are BM32/64/128 with BN64/128 and BM<=BN, '
                'or experimental BM128/BN512; S must be divisible by BN, groups 1..24, counts positive')
    pv_row32 = a.pv_row32 or a.pv_row32_batch_fix
    if pv_row32 and (a.block_m, a.block_n) != (128, 512):
        p.error('row32 PV modes require --block-m 128 --block-n 512')
    if a.prioritize_p_writeback and (a.block_m, a.block_n) != (128, 512):
        p.error('--prioritize-p-writeback requires --block-m 128 --block-n 512')
    if a.profile:
        profile_dir = Path(a.profile).resolve()
        if profile_dir.exists() and any(profile_dir.iterdir()):
            p.error('profile directory must be empty; select a new path')
    print('module:', fused_cube_ext.__file__, flush=True)
    root = Path(__file__).resolve().parent
    manifest = json.loads((root/'build_manifest.json').read_text())
    for name in ('fused_attention.cpp', 'mixed_kernel.hpp', 'cube_tiles.hpp', 'geometry.hpp'):
        if manifest.get(name) != hashlib.sha256((root/name).read_bytes()).hexdigest():
            raise RuntimeError('source changed after build; rerun python build.py')
    if manifest.get('module_sha256') != hashlib.sha256(Path(fused_cube_ext.__file__).read_bytes()).hexdigest():
        raise RuntimeError('loaded extension does not match build manifest')
    implementation = dict(fused_cube_ext.metadata())
    if a.prioritize_p_writeback:
        implementation['qk_lookahead_tiles'] = 2
        implementation['qk_refill_order'] = 'after oldest P_READY and PV'
    print('implementation:', implementation, flush=True)
    from verify_products import run_checks
    run_checks()
    shape = (a.batch, a.heads, a.seq, 128)
    indices = torch.linspace(0, a.seq-1, min(a.rows, a.seq)).long().unique()
    checks = []

    def call():
        return fused_cube_ext.forward(q, k, v, a.block_m, a.block_n, a.blocks,
                                      pv_row32, a.pv_row32_batch_fix, a.prioritize_p_writeback)

    def verify(out, label):
        # Copy only sampled outputs for CPU FP32 comparison. All values in all
        # dimensions of each sampled query row are compared, not just norms.
        actual = out[:, :, indices.to(out.device), :].cpu().float()
        err = (actual - ref).abs().max().item()
        finite = torch.isfinite(out).all().item()
        correct = finite and torch.allclose(actual, ref, atol=.002, rtol=.01)
        print(f'{label}: correct={correct} finite={finite} max_abs={err:.8g}', flush=True)
        checks.append({'label': label, 'correct': correct, 'finite': finite, 'max_abs': err})
        if not correct:
            raise RuntimeError('NPU reference check failed; no performance claim or success JSON written')

    with torch.inference_mode():
        for seed in a.seeds:
            torch.manual_seed(seed)
            cpu = [torch.randn(shape, dtype=torch.float16) for _ in range(3)]
            q, k, v = [x.to('npu:0') for x in cpu]
            # No .item(), synchronize(), or reference calculation between the
            # input copies and custom call: exercise normal task-queue ordering.
            out = call()
            torch.npu.synchronize()
            scores = cpu[0][:, :, indices, :].float() @ cpu[1].float().transpose(-1, -2) / math.sqrt(128)
            mask = torch.arange(a.seq).view(1,1,1,-1) > indices.view(1,1,-1,1)
            ref = torch.softmax(scores.masked_fill(mask, -float('inf')), -1) @ cpu[2].float()
            verify(out, f'seed={seed}')
        wall_ms, event_us = [], []
        if not a.verify_only:
            for _ in range(a.warmup):
                out = call()
            torch.npu.synchronize()
            for i in range(a.repeat):
                begin = torch.npu.Event(enable_timing=True)
                end = torch.npu.Event(enable_timing=True)
                torch.npu.synchronize()
                t0 = time.perf_counter()
                begin.record()
                out = call()
                end.record()
                torch.npu.synchronize()
                wall_ms.append((time.perf_counter()-t0)*1000)
                event_us.append(begin.elapsed_time(end)*1000)
                # Check every timed result after stopping the clock, exercising
                # repeated execution and allocator reuse as well as the first call.
                verify(out, f'timed_call={i}')
        profile = None
        if a.profile:
            if a.profile_metrics == 'memory-access':
                metrics = getattr(torch_npu.profiler.AiCMetrics, 'MemoryAccess', None)
                if metrics is None:
                    raise RuntimeError('torch_npu.profiler.AiCMetrics.MemoryAccess is unavailable')
            else:
                metrics = torch_npu.profiler.AiCMetrics.PipeUtilization
            config = torch_npu.profiler._ExperimentalConfig(
                profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                aic_metrics=metrics,
            )
            with torch_npu.profiler.profile(
                activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(profile_dir)),
                record_shapes=True, with_stack=False, experimental_config=config,
            ):
                out = call()
                torch.npu.synchronize()
            verify(out, 'profiled_call')
            counts = Counter()
            kernel_rows = []
            files = sorted(profile_dir.rglob('kernel_details.csv'))
            for file in files:
                with file.open(encoding='utf-8-sig', newline='') as f:
                    for row in csv.DictReader(f):
                        name = row.get('Name') or row.get('Task Name')
                        if not name:
                            raise RuntimeError(f'kernel name field missing: {file}')
                        counts[name] += 1
                        kernel_rows.append(row)
            matched = sum(n for name, n in counts.items() if 'fused_cube_attention' in name)
            profile = {'csv_files': [str(f) for f in files], 'name_counts': dict(counts),
                       'matched_fused_kernel_tasks': matched, 'total_kernel_tasks': sum(counts.values()),
                       'one_task_verified': bool(files) and matched == 1 and sum(counts.values()) == 1}
            def number(row,key):
                try: return float(row.get(key,'0').strip())
                except (ValueError,AttributeError): return 0.0
            profile['cube_activity_verified']=any(number(row,'aic_mac_time(us)')>0 or number(row,'aic_mac_ratio')>0 or number(row,'cube_utilization(%)')>0 for row in kernel_rows)
            profile['metric_set'] = a.profile_metrics
            profile['kernel_metrics']=kernel_rows
            print('profile:', profile, flush=True)
            if a.profile_metrics == 'pipe' and not profile['cube_activity_verified']:
                print('WARNING: positive AIC MAC/cycle activity was not confirmed; inspect profiler support/CSV.',flush=True)
            if not profile['one_task_verified']:
                print('WARNING: one fused task was not confirmed; inspect exported symbols/trace.', flush=True)
    result = {'shape': shape, 'dtype': 'float16', 'causal': True, 'device': torch.npu.get_device_name(0),
              'torch_version': torch.__version__, 'torch_npu_version': getattr(torch_npu,'__version__','unknown'), 'correct': True, 'checks': checks, 'query_rows_checked': indices.tolist(),
              'block_m': a.block_m, 'block_n': a.block_n, 'blocks': a.blocks,
              'pv_row32': pv_row32, 'pv_row32_batch_fix': a.pv_row32_batch_fix,
              'prioritize_p_writeback': a.prioritize_p_writeback,
              'gm_relay_bytes': a.blocks * implementation[f'gm_relay_bytes_per_group_bm{a.block_m}_bn{a.block_n}'],
              'implementation': implementation, 'wall_ms': wall_ms, 'event_us': event_us,
              'wall_median_ms': statistics.median(wall_ms) if wall_ms else None,
              'event_median_us': statistics.median(event_us) if event_us else None,
              'profile': profile, 'note': 'Mixed Cube/Vector, bounded GM relay; event time is a stream interval; inspect CSV AIC MAC counters'}
    Path(a.output).write_text(json.dumps(result, indent=2))
    print('saved:', a.output, flush=True)
    if wall_ms:
        print('wall median (ms):', result['wall_median_ms'], 'event median (us):', result['event_median_us'])


if __name__ == '__main__':
    main()
