"""Sweep causal attention context lengths with isolated runs and checkpoints."""
import argparse
import csv
import json
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LENGTHS = (128, 256, 512, 1024, 2048, 4096, 8192, 16384)
KINDS = ('sdpa', 'migrated', 'official')
MATCH = {'sdpa': 'flashattentionscore', 'migrated': '_causal_attention_kernel',
         'official': '_attn_fwd'}
FIELDS = ('sequence', 'implementation', 'correctness', 'reference_rows_checked',
          'max_abs_error', 'event_median_us', 'wall_median_us', 'timing_samples',
          'kernel_status', 'kernel_samples', 'kernel_median_us', 'kernel_min_us',
          'kernel_max_us', 'error')


def execute(command, logfile):
    with logfile.open('w', encoding='utf-8') as log:
        proc = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
    return proc.returncode


def kernel_result(directory, kind):
    files = list(directory.rglob('kernel_details.csv'))
    if len(files) != 1:
        return {'kernel_status': f'csv_count_{len(files)}'}
    with files[0].open(newline='', encoding='utf-8-sig') as f:
        rows = [r for r in csv.DictReader(f) if MATCH[kind] in r.get('Name', '').lower()]
    if not rows:
        return {'kernel_status': 'main_kernel_not_matched'}
    durations = [float(r['Duration(us)'].strip()) for r in rows]
    return {'kernel_status': 'ok', 'kernel_samples': len(durations),
            'kernel_median_us': statistics.median(durations), 'kernel_min_us': min(durations),
            'kernel_max_us': max(durations)}


def save_csv(path, rows):
    temp = path.with_suffix('.tmp')
    with temp.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--lengths', type=int, nargs='+', default=LENGTHS)
    p.add_argument('--warmup', type=int, default=20)
    p.add_argument('--repeats', type=int, default=100)
    p.add_argument('--profile-calls', type=int, default=1,
                   help='calls per profiler capture; validate captured count before interpreting')
    p.add_argument('--skip-profile', action='store_true')
    p.add_argument('--output', default=None, help='new output directory')
    a = p.parse_args()
    if min(*a.lengths, a.warmup, a.repeats, a.profile_calls) < 1:
        p.error('lengths, warmup, repeats and profile-calls must be positive')
    if any(n % 32 for n in a.lengths):
        p.error('sequence lengths must be divisible by the official 32x32 tile')
    target = Path(a.output or ('attention_sweep_' + datetime.now().strftime('%Y%m%d_%H%M%S'))).resolve()
    target.mkdir(parents=True, exist_ok=False)
    results = []
    (target/'settings.json').write_text(json.dumps(vars(a), indent=2), encoding='utf-8')
    for length in a.lengths:
        print(f'\n=== sequence={length} ===', flush=True)
        output = target / f'length_{length}.json'
        code = execute([sys.executable, str(ROOT/'compare_attention.py'), '--seq', str(length),
                        '--warmup', str(a.warmup), '--repeats', str(a.repeats), '--output', str(output)],
                       target/f'length_{length}.log')
        data = json.loads(output.read_text(encoding='utf-8')) if output.exists() else {}
        for kind in KINDS:
            info = data.get('implementations', {}).get(kind, {})
            row = {'sequence': length, 'implementation': kind,
                   'correctness': info.get('status', 'process_failed' if code else 'missing'),
                   'reference_rows_checked': len(data.get('reference', {}).get('query_rows_checked', [])),
                   'max_abs_error': info.get('max_abs_error_vs_fp32_cpu', ''),
                   'event_median_us': info.get('event_us_summary', {}).get('median', ''),
                   'wall_median_us': info.get('wall_us_summary', {}).get('median', ''),
                   'timing_samples': len(info.get('event_us', [])), 'error': info.get('error', '')}
            if info.get('status') == 'correct' and not a.skip_profile:
                profile = target/'profiles'/f'{length}_{kind}'
                impl = 'triton' if kind == 'migrated' else kind
                print(f'  profiling {kind}...', flush=True)
                ret = execute([sys.executable, str(ROOT/'profile_attention.py'), '--impl', impl,
                               '--seq', str(length), '--warmup', str(a.warmup), '--calls', str(a.profile_calls),
                               '--output', str(profile)], target/f'profile_{length}_{kind}.log')
                try:
                    row.update(kernel_result(profile, kind) if ret == 0 else {'kernel_status': f'profile_exit_{ret}'})
                    if row.get('kernel_status') == 'ok' and row['kernel_samples'] != a.profile_calls:
                        row['kernel_status'] = 'capture_count_mismatch'
                except (ValueError, KeyError, OSError) as exc:
                    row.update(kernel_status='parse_error', error=str(exc))
            else:
                row['kernel_status'] = 'skipped'
            results.append(row)
            save_csv(target/'summary.csv', results)
            print(f"  {kind}: status={row['correctness']} event_us={row['event_median_us']} "
                  f"wall_us={row['wall_median_us']} kernel_us={row.get('kernel_median_us', '')} "
                  f"kernel_samples={row.get('kernel_samples', '')}", flush=True)
    print(f'\nResults: {target / "summary.csv"}', flush=True)
    if any(r['correctness'] != 'correct' or (not a.skip_profile and r['kernel_status'] != 'ok') for r in results):
        raise SystemExit('Some measurements failed; see summary.csv and the per-length logs')


if __name__ == '__main__':
    main()
