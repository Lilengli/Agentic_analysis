"""Verify, benchmark and optionally profile the one-launch fused AIV kernel."""
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
import fused_aiv_ext


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seq', type=int, default=16384)
    p.add_argument('--batch', type=int, default=1)
    p.add_argument('--heads', type=int, default=2)
    p.add_argument('--block-m', type=int, choices=(8, 16), default=16)
    p.add_argument('--block-n', type=int, choices=(32, 64), default=32)
    p.add_argument('--blocks', type=int, default=48)
    p.add_argument('--rows', type=int, default=64)
    p.add_argument('--seeds', type=int, nargs='+', default=(1234, 4321))
    p.add_argument('--warmup', type=int, default=1)
    p.add_argument('--repeat', type=int, default=3)
    p.add_argument('--verify-only', action='store_true')
    p.add_argument('--profile', help='new output directory for one-call NPU profile')
    p.add_argument('--output', default='fused_aiv_16k.json')
    a = p.parse_args()
    if not (0 < a.seq <= 16384 and a.seq % a.block_n == 0 and 0 < a.blocks <= 48
            and min(a.rows, a.batch, a.heads, a.repeat) > 0 and a.warmup >= 0):
        p.error('invalid shape/count: require 0<S<=16384 divisible by BN, blocks 1..48, positive counts')
    if a.profile:
        profile_dir = Path(a.profile).resolve()
        if profile_dir.exists() and any(profile_dir.iterdir()):
            p.error('profile directory must be empty; select a new path')
    print('module:', fused_aiv_ext.__file__, flush=True)
    root = Path(__file__).resolve().parent
    manifest = json.loads((root/'build_manifest.json').read_text())
    for name in ('fused_attention.cpp', 'fused_kernel.hpp'):
        if manifest.get(name) != hashlib.sha256((root/name).read_bytes()).hexdigest():
            raise RuntimeError('source changed after build; rerun python build.py')
    if manifest.get('module_sha256') != hashlib.sha256(Path(fused_aiv_ext.__file__).read_bytes()).hexdigest():
        raise RuntimeError('loaded extension does not match build manifest')
    print('implementation:', dict(fused_aiv_ext.metadata()), flush=True)
    shape = (a.batch, a.heads, a.seq, 128)
    indices = torch.linspace(0, a.seq-1, min(a.rows, a.seq)).long().unique()
    checks = []

    def call():
        return fused_aiv_ext.forward(q, k, v, a.block_m, a.block_n, a.blocks)

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
            config = torch_npu.profiler._ExperimentalConfig(
                profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
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
            files = sorted(profile_dir.rglob('kernel_details.csv'))
            for file in files:
                with file.open(encoding='utf-8-sig', newline='') as f:
                    for row in csv.DictReader(f):
                        name = row.get('Name') or row.get('Task Name')
                        if not name:
                            raise RuntimeError(f'kernel name field missing: {file}')
                        counts[name] += 1
            matched = sum(n for name, n in counts.items() if 'fused_causal_attention' in name)
            profile = {'csv_files': [str(f) for f in files], 'name_counts': dict(counts),
                       'matched_fused_kernel_tasks': matched, 'total_kernel_tasks': sum(counts.values()),
                       'one_task_verified': bool(files) and matched == 1 and sum(counts.values()) == 1}
            print('profile:', profile, flush=True)
            if not profile['one_task_verified']:
                print('WARNING: one fused task was not confirmed; inspect exported symbols/trace.', flush=True)
    result = {'shape': shape, 'correct': True, 'checks': checks, 'query_rows_checked': indices.tolist(),
              'block_m': a.block_m, 'block_n': a.block_n, 'blocks': a.blocks,
              'implementation': dict(fused_aiv_ext.metadata()), 'wall_ms': wall_ms, 'event_us': event_us,
              'wall_median_ms': statistics.median(wall_ms) if wall_ms else None,
              'event_median_us': statistics.median(event_us) if event_us else None,
              'profile': profile, 'note': 'AIV-only, not Cube; event time is a stream interval, use CSV for task duration'}
    Path(a.output).write_text(json.dumps(result, indent=2))
    print('saved:', a.output, flush=True)
    if wall_ms:
        print('wall median (ms):', result['wall_median_ms'], 'event median (us):', result['event_median_us'])


if __name__ == '__main__':
    main()
