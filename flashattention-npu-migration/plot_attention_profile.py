"""Plot profiler kernel intervals and per-kernel Ascend pipeline ratios."""
import argparse
import csv
from decimal import Decimal, InvalidOperation
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle


def number(value):
    try:
        x = Decimal(str(value).strip())
        return x if x.is_finite() else None
    except (InvalidOperation, ValueError):
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('profile_dir', help='profiler output directory for one implementation')
    parser.add_argument('--output', required=True, help='output PNG path')
    parser.add_argument('--title', default='Attention profile')
    args = parser.parse_args()
    files = sorted(Path(args.profile_dir).rglob('kernel_details.csv'))
    if len(files) != 1:
        raise SystemExit(f'Expected exactly one kernel_details.csv; found {len(files)}: {files}')
    with files[0].open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    rows = [r for r in rows if number(r.get('Start Time(us)')) is not None
            and number(r.get('Duration(us)')) is not None and number(r['Duration(us)']) > 0]
    rows.sort(key=lambda r: number(r['Start Time(us)']))
    if not rows:
        raise SystemExit('No timed kernels were found')
    origin = number(rows[0]['Start Time(us)'])
    fields = [('aic_mac_ratio','AIC MAC','aic_total_cycles'),
              ('aic_scalar_ratio','AIC Scalar','aic_total_cycles'),
              ('aic_mte1_ratio','AIC MTE1','aic_total_cycles'),
              ('aic_mte2_ratio','AIC MTE2','aic_total_cycles'),
              ('aic_fixpipe_ratio','AIC Fixpipe','aic_total_cycles'),
              ('aiv_vec_ratio','AIV Vector','aiv_total_cycles'),
              ('aiv_scalar_ratio','AIV Scalar','aiv_total_cycles'),
              ('aiv_mte2_ratio','AIV MTE2','aiv_total_cycles'),
              ('aiv_mte3_ratio','AIV MTE3','aiv_total_cycles')]
    if not all(f in rows[0] and c in rows[0] for f, _, c in fields):
        raise SystemExit('AI Core ratio columns are absent; check Level1/PipeUtilization support')

    fig = plt.figure(figsize=(18, 8), layout='constrained')
    gs = fig.add_gridspec(2, 1, height_ratios=[5, 1.4])
    ax = fig.add_subplot(gs[0])
    bottom = fig.add_subplot(gs[1], sharex=ax)
    cmap = plt.colormaps['YlGnBu']
    norm = Normalize(0, 100)
    times = [(float(number(r['Start Time(us)']) - origin), float(number(r['Duration(us)']))) for r in rows]
    total = max(start + dur for start, dur in times)
    for y, (field, _, cycles) in enumerate(fields):
        for r, (start, duration) in zip(rows, times):
            value, count = number(r.get(field)), number(r.get(cycles))
            if value is None or count is None or count <= 0:
                continue
            ratio = float(value * 100)
            ax.add_patch(Rectangle((start, y-.36), duration, .72, facecolor=cmap(norm(ratio)), edgecolor='white', linewidth=.3))
            if duration >= total / 18:
                ax.text(start+duration/2, y, f'{ratio:.0f}%', ha='center', va='center', fontsize=7,
                        color='white' if ratio >= 65 else 'black', clip_on=True)
    ax.set_yticks(range(len(fields)), [x[1] for x in fields])
    ax.set_ylim(len(fields)-.5, -.5)
    ax.set_title('Per-kernel ratios: colored spans are aggregates, not time-resolved samples', loc='left')
    ax.tick_params(labelbottom=False)
    ax.grid(axis='x', alpha=.2)
    for i, (r, (start, duration)) in enumerate(zip(rows, times), 1):
        lane=(i-1) % 3
        bottom.barh(lane, duration, left=start, height=.65, color='#0072b2')
        if duration >= total / 40:
            bottom.text(start+duration/2, lane, str(i), ha='center', va='center', fontsize=8, color='white')
    bottom.set_yticks([])
    bottom.set_ylim(2.6, -.7)
    bottom.set_xlim(0, total * 1.02)
    bottom.set_xlabel('Time from first recorded kernel start (µs)')
    bottom.set_title('Kernel execution intervals; IDs refer to the CSV table', loc='left')
    bottom.grid(axis='x', alpha=.2)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label='Per-kernel pipeline ratio (%)', pad=.01)
    fig.suptitle(args.title)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    table = out.with_suffix('.kernels.csv')
    with table.open('w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['index', 'name', 'type', 'stream', 'relative_start_us', 'duration_us', 'input_shapes'])
        for i, (r, (start, duration)) in enumerate(zip(rows, times), 1):
            w.writerow([i, r.get('Name'), r.get('Type'), r.get('Stream ID'), f'{start:.3f}', f'{duration:.3f}', r.get('Input Shapes')])
    print(f'Input: {files[0]}\nGraph: {out}\nKernel table: {table}\nKernels: {len(rows)}')


if __name__ == '__main__':
    main()
