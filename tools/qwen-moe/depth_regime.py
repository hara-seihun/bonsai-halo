#!/usr/bin/env python3
"""Compare same-binary Qwen expert regimes across synthetic occupied depths."""
import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path

from decode_trace import family
from slow_regime import paired


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows_by_token(path, setup_heads):
    rows = list(csv.DictReader(path.open(newline='')))
    heads = [i for i, row in enumerate(rows) if family(row) == 'Q6_K/out248320']
    if len(heads) != setup_heads + 8:
        raise ValueError(f'{path}: expected {setup_heads + 8} heads, got {len(heads)}')
    cases = []
    for token in range(1, 8):
        head = heads[setup_heads + token]
        before = heads[setup_heads + token - 1]
        span = rows[before + 1:head + 1]
        if len(span) != 1565 or len({row['Queue_Id'] for row in span}) != 1:
            raise ValueError(f'{path}: token {token} changed dispatch/queue inventory')
        span.sort(key=lambda row: int(row['Start_Timestamp']))
        cases.extend([dict(case, token=token) for case in paired(span)])
    return cases


def mean(cases, key):
    return statistics.mean(case[key] for case in cases)


def counts(cases):
    slow = [c for c in cases if c['gate_up_us'] > 80]
    ordinary = [c for c in cases if c['gate_up_us'] <= 80]
    return {'layers': len(cases), 'slow_q4': len(slow),
            'slow_q4_layers_by_token': [[c['layer'] for c in cases if c['token'] == t and c['gate_up_us'] > 80]
                                        for t in range(4, 8)],
            'ordinary_q4_mean_us': mean(ordinary, 'gate_up_us'),
            'slow_q4_mean_us': mean(slow, 'gate_up_us') if slow else None,
            'q4_sum_us_per_token': sum(c['gate_up_us'] for c in cases) / len({c['token'] for c in cases}),
            'router_mean_us_on_slow_q4': mean(slow, 'router_us') if slow else None,
            'router_mean_us_on_ordinary_q4': mean(ordinary, 'router_us'),
            'slow_router_on_slow_q4': sum(c['router_us'] > 10 for c in slow),
            'slow_down_q5_on_slow_q4': sum(c['down_family'] == 'Q5_K/out2048' and c['down_us'] > 65 for c in slow),
            'q5_slow_q4_count': sum(c['down_family'] == 'Q5_K/out2048' for c in slow)}


def analyze(shallow, deep, wall):
    a, b = rows_by_token(shallow, 0), rows_by_token(deep, 2)
    if len(a) != len(b) or [c['layer'] for c in a] != [c['layer'] for c in b]:
        raise ValueError('layer alignments differ')
    pairs = [dict(token=x['token'], layer=x['layer'], shallow=x['gate_up_us'], deep=y['gate_up_us'],
                  shallow_router=x['router_us'], deep_router=y['router_us'],
                  shallow_down=x['down_us'], deep_down=y['down_us']) for x, y in zip(a, b)]
    result = {'trace_depth0_sha256': digest(shallow), 'trace_depth1024_sha256': digest(deep),
              'wall_sha256': digest(wall), 'analyzer_sha256': digest(Path(__file__)),
              'depth0_held': counts([c for c in a if c['token'] >= 4]),
              'depth1024_held': counts([c for c in b if c['token'] >= 4]),
              'depth0_all': counts(a), 'depth1024_all': counts(b),
              'paired_warm': pairs}
    held = [p for p in pairs if p['token'] >= 4]
    outliers = [p for p in held if p['shallow'] > 200 or p['deep'] > 200]
    typical = [p for p in held if p['shallow'] <= 200 and p['deep'] <= 200]
    result['held_cross_depth'] = {
        'outliers_over_200us': outliers,
        'typical_duration_correlation': statistics.correlation([p['shallow'] for p in typical],
                                                               [p['deep'] for p in typical]),
        'typical_depth0_q4_us_per_token': sum(p['shallow'] for p in typical) / 4,
        'typical_depth1024_q4_us_per_token': sum(p['deep'] for p in typical) / 4,
        'slow_both': sum(p['shallow'] > 80 and p['deep'] > 80 for p in held),
        'slow_only_depth0': sum(p['shallow'] > 80 and p['deep'] <= 80 for p in held),
        'slow_only_depth1024': sum(p['shallow'] <= 80 and p['deep'] > 80 for p in held),
        'ordinary_both': sum(p['shallow'] <= 80 and p['deep'] <= 80 for p in held),
        'all_duration_correlation_including_outlier': statistics.correlation([p['shallow'] for p in held],
                                                                             [p['deep'] for p in held]),
        'depth0_gate_us': sum(p['shallow'] for p in held) / 4,
        'depth1024_gate_us': sum(p['deep'] for p in held) / 4}
    result['wall_jsonl'] = [json.loads(line) for line in wall.read_text().splitlines() if line.startswith('{')]
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('depth0', type=Path)
    parser.add_argument('depth1024', type=Path)
    parser.add_argument('wall', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.depth0, args.depth1024, args.wall)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in ('depth0_held', 'depth1024_held', 'held_cross_depth')}, indent=2))
