#!/usr/bin/env python3
"""Measure whether slow routed expert calls are confined to the expert consumer.

This is a device-event analysis, not a causal intervention or wall-time estimate.
"""
import argparse
import csv
import json
import statistics
from pathlib import Path

from decode_trace import family, sha256


def duration(row):
    return (int(row['End_Timestamp']) - int(row['Start_Timestamp'])) / 1000


def paired(rows):
    kinds = [family(row) for row in rows]
    expert = [i for i, kind in enumerate(kinds) if kind == 'Q4_K/out512']
    q8 = [i for i, kind in enumerate(kinds) if kind == 'Q8_0/out8192']
    routers = [i for i, kind in enumerate(kinds) if kind == 'router/topk']
    if not (len(expert) == len(q8) == len(routers) == 40):
        raise ValueError('expected forty layers, routers and both matvecs')
    result = []
    for layer, (e, w, r) in enumerate(zip(expert, q8, routers)):
        if not (w < r < e and (layer == 39 or e < q8[layer + 1])):
            raise ValueError(f'layer {layer} dispatch order changed')
        if kinds[e - 1] != 'activation/q8':
            raise ValueError('expert input coder changed')
        downs = [i for i in range(e + 1, q8[layer + 1] if layer < 39 else len(rows))
                 if kinds[i] in ('Q5_K/out2048', 'Q6_K/out2048')]
        if len(downs) != 1 or kinds[downs[0] - 1] != 'activation/q8':
            raise ValueError(f'layer {layer} down dispatch changed')
        d = downs[0]
        result.append({'layer': layer, 'router_us': duration(rows[r]), 'gate_up_us': duration(rows[e]),
                       'down_us': duration(rows[d]), 'down_family': kinds[d],
                       'gate_up_activation_us': duration(rows[e - 1]),
                       'down_activation_us': duration(rows[d - 1]), 'q8_us': duration(rows[w])})
    return result


def contingency(cases, left, threshold_left, right, threshold_right):
    counts = [[0, 0], [0, 0]]
    for row in cases:
        counts[int(row[left] > threshold_left)][int(row[right] > threshold_right)] += 1
    return counts


def metrics(cases):
    slow = [row for row in cases if row['gate_up_us'] > 80]
    ordinary = [row for row in cases if row['gate_up_us'] <= 80]
    return {
        'calls': len(cases), 'gate_up_slow': len(slow),
        'router_10us_by_gate_up_80us': contingency(cases, 'gate_up_us', 80, 'router_us', 10),
        'down_q5_65us_by_gate_up_80us': contingency(
            [row for row in cases if row['down_family'] == 'Q5_K/out2048'],
            'gate_up_us', 80, 'down_us', 65),
        'next_layer_q8_105us_by_gate_up_80us': contingency(
            [dict(row, next_q8_us=cases[i + 1]['q8_us']) for i, row in enumerate(cases[:-1])
             if row['layer'] < 39 and cases[i + 1]['layer'] == row['layer'] + 1],
            'gate_up_us', 80, 'next_q8_us', 105),
        'same_layer_q8_105us_by_gate_up_80us': contingency(cases, 'gate_up_us', 80, 'q8_us', 105),
        'device_duration_means_us': {
            name: {'slow_gate_up_layers': statistics.mean(row[name] for row in slow),
                   'ordinary_gate_up_layers': statistics.mean(row[name] for row in ordinary)}
            for name in ('gate_up_us', 'router_us', 'down_us', 'gate_up_activation_us', 'down_activation_us')},
    }


def analyze(trace, receipt):
    installed = json.loads(receipt.read_text())
    if installed['files'][str(trace.relative_to(trace.parents[2]))] != sha256(trace):
        raise ValueError('trace differs from installed receipt')
    with trace.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    heads = [i for i, row in enumerate(rows) if family(row) == 'Q6_K/out248320']
    if len(heads) != 10:
        raise ValueError(f'expected two setup and eight measured heads, got {len(heads)}')
    tokens = []
    for token, (a, b) in enumerate(zip(heads[-9:-1], heads[-8:])):
        span = sorted(rows[a + 1:b + 1], key=lambda row: int(row['Start_Timestamp']))
        if len(span) != 1565 or {row['Queue_Id'] for row in span} != {'2'}:
            raise ValueError('decode dispatch inventory changed')
        tokens.extend([dict(case, token=token) for case in paired(span)])
    train = [row for row in tokens if 1 <= row['token'] <= 3]
    held = [row for row in tokens if 4 <= row['token'] <= 7]
    return {'source_commit': installed['source_commit'], 'model_sha256': installed['model_sha256'],
            'binary_hashes': installed['binary_hashes'], 'trace_sha256': sha256(trace),
            'installed_receipt_sha256': sha256(receipt), 'script_sha256': sha256(Path(__file__)),
            'contract': 'One counter-free profiled decode at depth 1024. Two setup heads excluded; token 0 cold excluded. Fixed training thresholds: gate/up 80us, router 10us, Q5 down 65us, wide Q8 105us. Conditional associations, not causal time savings or unprofiled wall times.',
            'train': metrics(train), 'held': metrics(held), 'cases': tokens}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace', type=Path)
    parser.add_argument('--receipt', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = analyze(args.trace, args.receipt)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'train': result['train'], 'held': result['held']}, indent=2))
