"""Combine the middle 20% of two complete CAModel core-group traces."""
import argparse
import json
from pathlib import Path


CORES = (('cubecore0', 'Cube'), ('veccore0', 'Vector 0'), ('veccore1', 'Vector 1'))
COMPUTE_PIPES = {'CUBE', 'VECTOR'}
TRANSFER_PIPES = {'MTE1', 'MTE2', 'MTE3', 'FIXP'}


def load_group(directory, core):
    group = []
    for suffix, label in CORES:
        matches = sorted(directory.rglob(f'core{core}.{suffix}/trace.json'))
        if len(matches) != 1:
            raise ValueError(f'expected one core{core}.{suffix}/trace.json under {directory}, found {len(matches)}')
        path, = matches
        raw = json.loads(path.read_text())
        events = raw if isinstance(raw, list) else raw['traceEvents']
        pipes = {e['pid']: e['args']['name'] for e in events
                 if e.get('ph') == 'M' and e.get('name') == 'process_name'}
        threads = {(e['pid'], e['tid']): e['args']['name'] for e in events
                   if e.get('ph') == 'M' and e.get('name') == 'thread_name'}
        slices = [e for e in events if e.get('ph') == 'X' and e.get('dur', 0) > 0
                  and pipes.get(e['pid']) != 'CACHEMISS']
        if not slices:
            raise ValueError(f'no instruction slices in {path}')
        group.append((label, pipes, threads, slices))
    return group


def crop(group, label, first_pid):
    first = min(e['ts'] for _, _, _, slices in group for e in slices)
    last = max(e['ts'] + e['dur'] for _, _, _, slices in group for e in slices)
    begin = first + 0.4 * (last - first)
    end = first + 0.6 * (last - first)
    output = []
    for index, (role, pipes, threads, slices) in enumerate(group):
        pid = first_pid + index
        output.append({'ph': 'M', 'name': 'process_name', 'pid': pid, 'tid': 0,
                       'args': {'name': f'{label} {role}'}})
        output.append({'ph': 'M', 'name': 'process_sort_index', 'pid': pid, 'tid': 0,
                       'args': {'sort_index': pid}})
        tracks = {}
        for event in slices:
            start = max(begin, event['ts'])
            finish = min(end, event['ts'] + event['dur'])
            if start >= finish:
                continue
            pipe = pipes.get(event['pid'], '')
            name = event['name']
            if name.startswith(('WAIT_FLAG', 'SET_FLAG')) or name == 'BAR':
                category = 'sync'
            elif pipe in COMPUTE_PIPES:
                category = 'compute'
            elif pipe in TRANSFER_PIPES:
                category = 'move'
            else:
                continue
            key = (event['pid'], event['tid'])
            if key not in tracks:
                tracks[key] = len(tracks) + 1
                output.append({'ph': 'M', 'name': 'thread_name', 'pid': pid, 'tid': tracks[key],
                               'args': {'name': threads.get(key, pipe)}})
            args = {**event.get('args', {}), 'pipe': pipe, 'original_start_us': event['ts'],
                    'original_duration_us': event['dur']}
            output.append({**event, 'pid': pid, 'tid': tracks[key], 'ts': start - begin,
                           'dur': finish - start, 'cat': category, 'args': args})
    return output, {'core_span_us': [first, last], 'selected_span_us': [begin, end]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cann', type=Path, help='complete CANN CAModel output directory')
    parser.add_argument('generated', type=Path, help='complete generated CAModel output directory')
    parser.add_argument('--core', type=int, default=15)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    official, official_window = crop(load_group(args.cann, args.core), 'CANN', 1)
    generated, generated_window = crop(load_group(args.generated, args.core), 'Generated BN1024', 4)
    result = {'displayTimeUnit': 'ns', 'traceEvents': official + generated,
              'core': args.core, 'cann_window': official_window, 'generated_window': generated_window}
    args.output.write_text(json.dumps(result))
    print(f'Saved {args.output}')


if __name__ == '__main__':
    main()
