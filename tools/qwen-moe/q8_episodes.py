#!/usr/bin/env python3
"""Separate persistent wide-Q8 latency from the ordinary installed MoE decode path."""

import argparse
import csv
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path

from decode_trace import family


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def us(row):
    return (int(row['End_Timestamp']) - int(row['Start_Timestamp'])) / 1000


def analyze(path, receipt, cutoff=105):
    rows = list(csv.DictReader(path.open(newline='')))
    heads = [i for i, row in enumerate(rows) if family(row) == 'Q6_K/out248320']
    spans = [rows[a + 1:b + 1] for a, b in zip(heads[-9:-1], heads[-8:])]
    if len(spans) != 8 or any(len(span) != 1565 for span in spans):
        raise ValueError('expected eight depth-1024 decode spans of 1565 dispatches')
    observed = []
    for token, span in enumerate(spans):
        routers = [i for i, row in enumerate(span) if family(row) == 'router/topk']
        if len(routers) != 40:
            raise ValueError('expected 40 layer boundaries')
        q8 = []
        q4 = [row for row in span if family(row) == 'Q4_K/out512']
        if len(q4) != 40:
            raise ValueError('expected forty routed gate/up calls')
        for layer, stop in enumerate(routers):
            start = 0 if layer == 0 else routers[layer - 1] + 1
            wide = [row for row in span[start:stop] if family(row) == 'Q8_0/out8192']
            if len(wide) != 1:
                raise ValueError(f'wrong wide Q8 alignment at token {token}, layer {layer}')
            q8.append(wide[0])
        for layer, (wide, expert) in enumerate(zip(q8, q4)):
            observed.append({'token': token, 'layer': layer, 'q8_us': us(wide),
                             'q4_us': us(expert), 'q8_grid': wide['Grid_Size_X'],
                             'q8_kernel': wide['Kernel_Name'], 'q8_queue': wide['Queue_Id']})
    if {o['q8_grid'] for o in observed} != {'262144'} or len({o['q8_kernel'] for o in observed}) != 1:
        raise ValueError('wide Q8 geometry/template varies')
    train = [o for o in observed if 1 <= o['token'] <= 3]
    held = [o for o in observed if 4 <= o['token'] <= 7]
    warm = train + held
    persistent = sorted({layer for layer in range(40) if sum(o['q8_us'] > cutoff for o in train
                                                             if o['layer'] == layer) >= 2})
    ordinary = [o['q8_us'] for o in warm if o['q8_us'] <= cutoff]
    slow = [o['q8_us'] for o in warm if o['q8_us'] > cutoff]
    baseline = statistics.median(ordinary)
    contingency = Counter(('wide_slow' if o['q8_us'] > cutoff else 'wide_ordinary',
                           'expert_slow' if o['q4_us'] > 80 else 'expert_ordinary') for o in warm)
    p = json.loads(receipt.read_text())
    if p['files']['trace/gpu-host/1886571_kernel_trace.csv'] != digest(path):
        raise ValueError('trace does not match installed receipt')
    return {
        'trace_sha256': digest(path), 'installed_receipt_sha256': digest(receipt),
        'script_sha256': digest(Path(__file__)), 'source_commit': p['source_commit'],
        'model_sha256': p['model_sha256'], 'binary_hashes': p['binary_hashes'],
        'contract': 'HIP event durations for depth-1024 single-row decode. One pre-router width-8192 Q8 per router; Q4 paired by ordered gate/up call index, since graph execution can schedule a Q4 before its router. Token 0 excluded from warm fit. No wall or DRAM-byte inference.',
        'threshold_us': cutoff, 'ordinary_median_us': baseline,
        'warm': {'observations': len(warm), 'wide_slow': len(slow),
                 'wide_ordinary_mean_us': statistics.mean(ordinary),
                 'wide_slow_mean_us': statistics.mean(slow),
                 'wide_mean_us': statistics.mean(o['q8_us'] for o in warm),
                 'wide_device_ms_per_token': sum(o['q8_us'] for o in warm) / 7000,
                 'episode_excess_vs_ordinary_median_ms_per_token': sum(max(0, v - baseline) for v in slow) / 7000,
                 'q8_q4_contingency': [{'q8': k[0], 'q4': k[1], 'count': n}
                                       for k, n in sorted(contingency.items())]},
        'held_prediction': {
            'train_tokens': [1, 2, 3], 'held_tokens': [4, 5, 6, 7],
            'persistent_layers': persistent,
            'true_positive': sum(o['layer'] in persistent and o['q8_us'] > cutoff for o in held),
            'false_positive': sum(o['layer'] in persistent and o['q8_us'] <= cutoff for o in held),
            'false_negative': sum(o['layer'] not in persistent and o['q8_us'] > cutoff for o in held),
            'true_negative': sum(o['layer'] not in persistent and o['q8_us'] <= cutoff for o in held),
        },
        'per_layer': [{'layer': layer,
                       'wide_median_us': statistics.median(o['q8_us'] for o in warm if o['layer'] == layer),
                       'expert_median_us': statistics.median(o['q4_us'] for o in warm if o['layer'] == layer)}
                      for layer in range(40)],
        'rows': observed,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace', type=Path)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(analyze(args.trace, args.receipt), indent=2) + '\n')
