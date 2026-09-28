#!/usr/bin/env python3
"""Test whether traced overlap or a shared launch regime explains slow MoE matvecs."""

import argparse
import csv
import json
import statistics
from pathlib import Path

from decode_trace import family, sha256


TARGETS = {'Q8_0/out8192': 105, 'Q4_K/out512': 80}


def analyze(trace, receipt, installed):
    if installed['files'][str(trace.relative_to(trace.parents[2]))] != sha256(trace):
        raise ValueError('trace differs from installed receipt')
    with trace.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    heads = [i for i, row in enumerate(rows) if family(row) == 'Q6_K/out248320']
    if len(heads) != 10:
        raise ValueError(f'expected two setup and eight decode heads, got {len(heads)}')
    events = []
    spans = []
    for token, (a, b) in enumerate(zip(heads[-9:-1], heads[-8:])):
        ordered = sorted(rows[a + 1:b + 1], key=lambda r: int(r['Start_Timestamp']))
        if len(ordered) != 1565 or {r['Queue_Id'] for r in ordered} != {'2'}:
            raise ValueError(f'decode token {token} changed dispatch inventory or queue')
        gaps = [int(y['Start_Timestamp']) - int(x['End_Timestamp'])
                for x, y in zip(ordered, ordered[1:])]
        duration = sum(int(r['End_Timestamp']) - int(r['Start_Timestamp']) for r in ordered)
        elapsed = int(ordered[-1]['End_Timestamp']) - int(ordered[0]['Start_Timestamp'])
        if duration + sum(gaps) != elapsed:
            raise ValueError('event interval accounting failed')
        spans.append({'token': token, 'dispatches': len(ordered), 'overlapping_pairs': sum(g < 0 for g in gaps),
                      'device_ms': duration / 1e6, 'idle_ms': sum(max(0, g) for g in gaps) / 1e6,
                      'elapsed_ms': elapsed / 1e6})
        for i, row in enumerate(ordered):
            kind = family(row)
            if kind not in TARGETS:
                continue
            if i == 0 or family(ordered[i - 1]) != 'activation/q8':
                raise ValueError(f'{kind} has no immediately preceding activation/q8')
            predecessor = ordered[i - 1]
            events.append({'token': token, 'family': kind,
                           'kernel_us': (int(row['End_Timestamp']) - int(row['Start_Timestamp'])) / 1000,
                           'activation_us': (int(predecessor['End_Timestamp']) - int(predecessor['Start_Timestamp'])) / 1000,
                           'gap_us': (int(row['Start_Timestamp']) - int(predecessor['End_Timestamp'])) / 1000,
                           'activation_grid': predecessor['Grid_Size_X'], 'activation_kernel': predecessor['Kernel_Name']})
    summary = {}
    for kind, cutoff in TARGETS.items():
        arm = [e for e in events if e['family'] == kind and e['token'] > 0]
        if len(arm) != 280 or len({e['activation_grid'] for e in arm}) != 1 or len({e['activation_kernel'] for e in arm}) != 1:
            raise ValueError(f'{kind} does not have 40 identical predecessors per warm token')
        slow = [e for e in arm if e['kernel_us'] > cutoff]
        ordinary = [e for e in arm if e['kernel_us'] <= cutoff]
        if not slow or not ordinary:
            raise ValueError('no comparison group')
        groups = {}
        for label, subset in [('slow', slow), ('ordinary', ordinary)]:
            groups[label] = {'count': len(subset), 'kernel_mean_us': statistics.mean(e['kernel_us'] for e in subset),
                             'activation_mean_us': statistics.mean(e['activation_us'] for e in subset),
                             'predecessor_gap_mean_us': statistics.mean(e['gap_us'] for e in subset)}
        held = [e for e in arm if e['token'] >= 4]
        train = [e for e in arm if e['token'] <= 3]
        def table(pool):
            return {'slow_long_activation': sum(e['kernel_us'] > cutoff and e['activation_us'] > 3 for e in pool),
                    'slow_short_activation': sum(e['kernel_us'] > cutoff and e['activation_us'] <= 3 for e in pool),
                    'ordinary_long_activation': sum(e['kernel_us'] <= cutoff and e['activation_us'] > 3 for e in pool),
                    'ordinary_short_activation': sum(e['kernel_us'] <= cutoff and e['activation_us'] <= 3 for e in pool)}
        summary[kind] = {'slow_threshold_us': cutoff, 'activation_threshold_us': 3,
                         'activation_grid': arm[0]['activation_grid'], 'activation_kernel': arm[0]['activation_kernel'],
                         'group_means': groups, 'train_contingency': table(train), 'held_contingency': table(held),
                         'activation_kernel_pearson': statistics.correlation(
                             [e['activation_us'] for e in arm], [e['kernel_us'] for e in arm])}
    return {'source_commit': installed['source_commit'], 'model_sha256': installed['model_sha256'],
            'binary_hashes': installed['binary_hashes'], 'trace_sha256': sha256(trace),
            'installed_receipt_sha256': sha256(receipt), 'script_sha256': sha256(Path(__file__)),
            'contract': 'One counter-free profiled depth-1024 single-row decode, sorted device event intervals. Token 0 cold; tokens 1-3 train, 4-7 held. Events are not unprofiled wall times or DRAM bytes. No causal intervention.',
            'spans': spans, 'summary': summary, 'events': events}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace', type=Path)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    installed = json.loads(args.receipt.read_text())
    result = analyze(args.trace, args.receipt, installed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'spans': result['spans'], 'summary': result['summary']}, indent=2))
