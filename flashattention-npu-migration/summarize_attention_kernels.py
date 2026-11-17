"""Summarize captured main-kernel durations without counting setup kernels."""
import argparse
import csv
import statistics
from collections import Counter
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path)
    p.add_argument('--impl', choices=('triton', 'sdpa', 'official'), required=True)
    p.add_argument('--expected', type=int, default=20)
    args = p.parse_args()
    files = list(args.directory.rglob('kernel_details.csv'))
    if len(files) != 1:
        p.error(f'expected one kernel_details.csv, found {len(files)}: {files}')
    with files[0].open(newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    needles = {'triton': '_causal_attention_kernel', 'sdpa': 'flashattentionscore',
               'official': '_attn_fwd'}
    needle = needles[args.impl].lower()
    selected = [r for r in rows if needle in r.get('Name', '').lower()]
    print(f'file: {files[0]}')
    print('all kernel names and counts:')
    for name, count in Counter(r.get('Name', '') for r in rows).most_common():
        print(f'  {count:5d}  {name}')
    if not selected:
        raise SystemExit(f'No kernel name matched {needle!r}; inspect names above')
    durations = [float(r['Duration(us)'].strip()) for r in selected]
    print(f'\nmain kernel match: {needle}; count={len(durations)}; expected={args.expected}')
    if len(durations) != args.expected:
        print('WARNING: count differs from expected; inspect kernel names and capture before comparing')
    print(f'min={min(durations):.3f} us  median={statistics.median(durations):.3f} us  '
          f'mean={statistics.mean(durations):.3f} us  max={max(durations):.3f} us')
    for r in selected[:1]:
        print('example:', r.get('Name'), 'input shapes:', r.get('Input Shapes'),
              'input dtypes:', r.get('Input Data Types'))


if __name__ == '__main__':
    main()
