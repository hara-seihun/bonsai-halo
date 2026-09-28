#!/usr/bin/env python3
"""Locate layer-persistent slow dispatches in the installed Qwen decode trace."""
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path

from decode_trace import family


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def duration(row):
    return (int(row['End_Timestamp']) - int(row['Start_Timestamp'])) / 1000


def median(xs):
    return statistics.median(xs)


def analyze(path):
    rows = list(csv.DictReader(path.open(newline='')))
    heads = [i for i, row in enumerate(rows) if family(row) == 'Q6_K/out248320']
    spans = [rows[a + 1:b + 1] for a, b in zip(heads[-9:-1], heads[-8:])]
    if len(spans) != 8 or any(len(s) != 1565 for s in spans):
        raise ValueError('expected eight installed depth-1024 decode spans')
    observations = []
    for token, span in enumerate(spans):
        routers = [i for i, row in enumerate(span) if family(row) == 'router/topk']
        q4 = [row for row in span if family(row) == 'Q4_K/out512']
        downs = [row for row in span if family(row) in ('Q5_K/out2048', 'Q6_K/out2048')]
        shared = [row for row in span if family(row) == 'Q8_0/out512' and 'true, true' in row['Kernel_Name']]
        if any(len(group) != 40 for group in (routers, q4, downs, shared)):
            raise ValueError('missing per-layer dispatch')
        for layer, i in enumerate(routers):
            r = span[i]
            g = q4[layer]
            # Compare one identical upstream operation, not a median of
            # unrelated elementwise kernels or an expert-dependent gather.
            before = [x for x in span[max(0, i - 3):i]
                      if x['Kernel_Name'].startswith('void rms_norm_f32<1024, true, false>')
                      and x['Grid_Size_X'] == '1024']
            if len(before) > 1:
                raise ValueError('ambiguous pre-router RMS norm')
            observations.append({
                'token': token, 'layer': layer,
                'slow': duration(g) > 80,
                'router_us': duration(r), 'gate_up_us': duration(g),
                'down_us': duration(downs[layer]), 'shared_us': duration(shared[layer]),
                'prerouter_rms_us': duration(before[0]) if before else None,
                'router_gap_us': (int(r['Start_Timestamp']) - int(span[i - 1]['End_Timestamp'])) / 1000,
                'router_queue': r['Queue_Id'], 'expert_queue': g['Queue_Id'],
                'expert_grid': [g[f'Grid_Size_{c}'] for c in 'XYZ'],
            })
    train = observations[40:160]  # tokens 1,2,3; skip cold first token
    test = observations[160:]      # tokens 4..7
    persistent = sorted({layer for layer in range(40) if sum(o['slow'] for o in train if o['layer'] == layer) >= 2})
    prediction = {'train_tokens': [1, 2, 3], 'test_tokens': [4, 5, 6, 7],
                  'selected_layers': persistent,
                  'true_positive': sum(o['slow'] and o['layer'] in persistent for o in test),
                  'false_positive': sum(not o['slow'] and o['layer'] in persistent for o in test),
                  'false_negative': sum(o['slow'] and o['layer'] not in persistent for o in test),
                  'true_negative': sum(not o['slow'] and o['layer'] not in persistent for o in test)}
    metrics = {}
    for key in ('router_us', 'gate_up_us', 'down_us', 'shared_us', 'prerouter_rms_us', 'router_gap_us'):
        metrics[key] = {}
        for slow in (False, True):
            vals = [o[key] for o in observations[40:] if o['slow'] == slow and o[key] is not None]
            metrics[key]['slow' if slow else 'ordinary'] = {'count': len(vals), 'median_us': median(vals), 'mean_us': statistics.mean(vals)}
    by_layer = []
    for layer in range(40):
        sample = [o for o in observations[40:] if o['layer'] == layer]
        by_layer.append({'layer': layer, 'slow_count_warm': sum(o['slow'] for o in sample),
                         'gate_up_median_us': median([o['gate_up_us'] for o in sample]),
                         'router_median_us': median([o['router_us'] for o in sample])})
    parent_receipt = path.parents[2] / 'receipt.json'
    return {'trace': str(path.resolve()), 'trace_sha256': sha(path),
            'script_sha256': sha(Path(__file__)),
            'installed_receipt_sha256': sha(parent_receipt),
            'contract': 'HIP kernel-event durations, 8 decode tokens at depth 1024; token 0 excluded from warm comparisons; gate/up >80 us labels slow; no physical byte or wall-time inference',
            'prediction': prediction, 'warm_by_mode': metrics, 'by_layer': by_layer,
            'warm_tokens': [{'token': t, 'slow_layers': [o['layer'] for o in observations if o['token'] == t and o['slow']]} for t in range(1, 8)],
            'queues': sorted({(o['router_queue'], o['expert_queue']) for o in observations}),
            'expert_grids': sorted({tuple(o['expert_grid']) for o in observations})}


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('trace', type=Path)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    result = json.dumps(analyze(args.trace), indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result)
    else:
        print(result, end='')
