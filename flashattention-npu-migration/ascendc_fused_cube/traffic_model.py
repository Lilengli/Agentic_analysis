"""Logical GM traffic for the causal fused Cube schedule; not physical HBM traffic."""
import argparse
import json


def estimate(seq, heads, bm, bn, dim=128):
    if min(seq, heads) <= 0 or bm not in (32, 64, 128) or bn not in (64, 128) or bm > bn or seq % bn:
        raise ValueError('require positive heads and sequence divisible by BN, BM<=BN')
    tasks = heads * (seq // bm)
    tiles = heads * sum(qs // bn + 1 for qs in range(0, seq, bm))
    # Every query tile is loaded once to L1. K, V, P and relay results are
    # transferred once per KV iteration. Counters count logical GM transfers.
    q = tasks * bm * dim * 2
    k_v = tiles * 2 * bn * dim * 2
    scores = tiles * bm * bn * 4
    probs = tiles * bm * bn * 2
    pv = tiles * bm * dim * 4
    out = tasks * bm * dim * 2
    result = {'seq': seq, 'heads_including_batch': heads, 'block_m': bm, 'block_n': bn,
              'query_tiles': tasks, 'kv_tile_iterations': tiles,
              'q_read_bytes': q, 'kv_read_bytes': k_v,
              'score_relay_each_direction_bytes': scores,
              'probability_relay_each_direction_bytes': probs,
              'pv_relay_each_direction_bytes': pv, 'output_write_bytes': out,
              'total_logical_read_bytes': q + k_v + scores + probs + pv,
              'total_logical_write_bytes': scores + probs + pv + out,
              'note': 'Source-derived logical GM transfers, not measured HBM transactions; excludes cache effects and ancillary data.'}
    result['total_logical_transfer_bytes'] = result['total_logical_read_bytes'] + result['total_logical_write_bytes']
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seq', type=int, default=16384)
    p.add_argument('--heads', type=int, default=2)
    a = p.parse_args()
    baseline = estimate(a.seq, a.heads, 64, 128)
    candidate = estimate(a.seq, a.heads, 128, 128)
    print(json.dumps({'baseline': baseline, 'candidate': candidate,
                      'estimated_total_reduction_fraction': 1 - candidate['total_logical_transfer_bytes']/baseline['total_logical_transfer_bytes']}, indent=2))


if __name__ == '__main__':
    main()
