#!/usr/bin/env python3
"""Join installed wide-Q8 event durations to preceding recorder-sized trace gaps."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean, median

from decode_trace import family


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def token_cases(rows, token):
    rows = sorted(rows, key=lambda row: int(row['Start_Timestamp']))
    if len(rows) != 1565 or {row['Queue_Id'] for row in rows} != {'2'}:
        raise ValueError('unexpected decode span / queue')
    routers = [i for i, row in enumerate(rows) if family(row) == 'router/topk']
    wide = [i for i, row in enumerate(rows) if family(row) == 'Q8_0/out8192']
    if len(routers) != 40 or len(wide) != 40:
        raise ValueError('unexpected router / wide-Q8 inventory')
    gaps = [i for i in range(len(rows) - 1)
            if int(rows[i + 1]['Start_Timestamp']) - int(rows[i]['End_Timestamp']) >= 500000]
    cases = []
    for layer, stop in enumerate(routers):
        start = routers[layer - 1] + 1 if layer else 0
        found = [i for i in wide if start <= i < stop]
        if len(found) != 1:
            raise ValueError(f'Q8 layer alignment failed: token {token} layer {layer}')
        i = found[0]
        row = rows[i]
        duration = (int(row['End_Timestamp']) - int(row['Start_Timestamp'])) / 1000
        if duration < 0:
            raise ValueError('negative event duration')
        preceding = [i - gap for gap in gaps if gap < i]
        cases.append({'token': token, 'layer': layer, 'ordinal': i, 'router_ordinal': stop,
                      'duration_us': duration, 'slow': duration > 105,
                      'distance_after_long_gap': min(preceding) if preceding else None})
    return gaps, cases


def panel(cases):
    result = {'calls': len(cases), 'slow': sum(c['slow'] for c in cases),
              'mean_us': mean(c['duration_us'] for c in cases),
              'median_us': median(c['duration_us'] for c in cases)}
    for distance in (8, 16, 32, 64, 96, 128):
        near = [c for c in cases if c['distance_after_long_gap'] is not None
                and c['distance_after_long_gap'] <= distance]
        far = [c for c in cases if c['distance_after_long_gap'] is None
               or c['distance_after_long_gap'] > distance]
        result[f'within_{distance}_dispatches'] = {
            'near_calls': len(near), 'near_slow': sum(c['slow'] for c in near),
            'near_mean_us': mean(c['duration_us'] for c in near) if near else None,
            'far_calls': len(far), 'far_slow': sum(c['slow'] for c in far),
            'far_mean_us': mean(c['duration_us'] for c in far) if far else None,
        }
    return result


def read_trace(trace, receipt, expected_heads):
    original = json.loads(receipt.read_text())
    relative = str(trace.relative_to(trace.parents[2]))
    if original['files'][relative] != sha(trace):
        raise ValueError('trace does not match original installed receipt')
    rows = list(csv.DictReader(trace.open(newline='')))
    heads = [i for i, row in enumerate(rows) if family(row) == 'Q6_K/out248320']
    if len(heads) != expected_heads:
        raise ValueError(f'expected {expected_heads} complete heads, found {len(heads)}')
    tokens = []
    end_heads = heads[-8:] if expected_heads == 10 else heads[1:]
    for n, b in enumerate(end_heads):
        token = n if expected_heads == 10 else n + 1
        a = b - 1564
        if n and end_heads[n - 1] != a - 1:
            raise ValueError('non-contiguous decode token spans')
        gaps, cases = token_cases(rows[a:b + 1], token)
        tokens.append({'token': token, 'long_gap_ordinals': gaps, 'cases': cases})
    return {'trace_sha256': sha(trace), 'receipt_sha256': sha(receipt), 'tokens': tokens}


def analyze(trace, receipt, depth0_trace, depth0_receipt):
    original = json.loads(receipt.read_text())
    second = json.loads(depth0_receipt.read_text())
    if second['binary_sha256'] != original['binary_hashes']['llama-bench']:
        raise ValueError('depth-0 experiment used a different executable')
    first = read_trace(trace, receipt, 10)
    depth0 = read_trace(depth0_trace, depth0_receipt, 8)
    train = [c for t in first['tokens'][1:4] for c in t['cases']]
    held = [c for t in first['tokens'][4:] for c in t['cases']]
    independent = [c for t in depth0['tokens'] for c in t['cases']]
    return {
        'contract': 'Two same-binary counter-free HIP-profiler traces at synthetic occupied depth 1024 and 0. Two setup heads (depth 1024) and cold token 0 (both) excluded. Previous Q8 slow cutoff >105 us, expert-gap window <=64 dispatches and recorder-gap cutoff >=0.5 ms frozen before this join. One pre-router wide Q8 per layer. Association is neither causation, unprofiled device time, DRAM traffic nor wall speed.',
        'source_commit': original['source_commit'], 'model_sha256': original['model_sha256'],
        'binary_hashes': original['binary_hashes'], 'script_sha256': sha(Path(__file__)),
        'depth1024': first, 'depth0': depth0,
        'inspection': panel(train), 'held': panel(held), 'independent_depth0': panel(independent),
    }


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('trace', type=Path)
    p.add_argument('--receipt', type=Path, required=True)
    p.add_argument('--depth0-trace', type=Path, required=True)
    p.add_argument('--depth0-receipt', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    result = analyze(args.trace, args.receipt, args.depth0_trace, args.depth0_receipt)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('inspection', 'held', 'independent_depth0')}, indent=2))
