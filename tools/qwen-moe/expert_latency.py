#!/usr/bin/env python3
"""Paired, per-layer diagnosis of routed expert timing in a measured decode trace."""
import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path

from decode_trace import family


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def us(row):
    return (int(row['End_Timestamp']) - int(row['Start_Timestamp'])) / 1000


def summarize(values):
    return {'count': len(values), 'mean_us': statistics.mean(values),
            'median_us': statistics.median(values), 'min_us': min(values),
            'max_us': max(values), 'sum_us': sum(values)}


def analyze(path, tokens=8):
    rows = list(csv.DictReader(path.open(newline='')))
    heads = [i for i, row in enumerate(rows) if family(row) == 'Q6_K/out248320']
    if len(heads) < tokens + 1:
        raise ValueError('missing setup head or measured token')
    spans = [rows[a+1:b+1] for a, b in zip(heads[-tokens-1:-1], heads[-tokens:])]
    if any(len(span) != 1565 for span in spans):
        raise ValueError('not the installed depth-1024 decode shape')
    observations = []
    for token, span in enumerate(spans):
        routers = [i for i, row in enumerate(span) if family(row) == 'router/topk']
        if len(routers) != 40:
            raise ValueError('expected one router per layer')
        gate_up = [row for row in span if family(row) == 'Q4_K/out512']
        down = [row for row in span if family(row) in ('Q5_K/out2048', 'Q6_K/out2048')]
        shared = [row for row in span if family(row) == 'Q8_0/out512' and
                  'true, true' in row['Kernel_Name']]
        if not all(len(group) == 40 for group in (gate_up, down, shared)):
            raise ValueError(f'token {token}: expected 40 calls of each layer operation')
        # Independent Q8 and routed operations can interleave across adjacent
        # router boundaries. Pair ordinal layers, not router-delimited spans.
        for layer, start in enumerate(routers):
            router = span[start]
            if not router['Kernel_Name'].startswith('void topk_moe_cuda<256, false>') or (router['Grid_Size_X'], router['Grid_Size_Y'], router['Workgroup_Size_X']) != ('32', '4', '32'):
                raise ValueError('router shape changed')
            observations.append({'token': token, 'layer': layer, 'router_us': us(router),
                                 'gate_up_us': us(gate_up[layer]), 'down_us': us(down[layer]),
                                 'shared_us': us(shared[layer]), 'down_type': family(down[layer])})
    groups = {'ordinary_gate_up': [], 'slow_gate_up': []}
    # Q4 timings form two separated modes near 50 and 110 us. This split
    # describes the observed mixture; it does not diagnose the hardware cause.
    for entry in observations:
        groups['slow_gate_up' if entry['gate_up_us'] > 80 else 'ordinary_gate_up'].append(entry)
    result = {'trace': str(path.resolve()), 'trace_sha256': digest(path),
              'analyzer_sha256': digest(Path(__file__)),
              'parent_receipt_sha256': digest(path.parent.parent.parent / 'receipt.json'),
              'domain': {'tokens': tokens, 'layers': 40, 'measured_rows': len(spans[0]),
                         'gate_up_split_us': 80, 'router_slow_threshold_us': 10,
                         'setup_heads_excluded': len(heads)-tokens},
              'groups': {}, 'slow_layers_by_token': [], 'per_layer': []}
    for name, entries in groups.items():
        result['groups'][name] = {key: summarize([e[key] for e in entries]) for key in
                                  ('router_us', 'gate_up_us', 'down_us', 'shared_us')}
        result['groups'][name]['router_over_10us'] = sum(e['router_us'] > 10 for e in entries)
        result['groups'][name]['down_over_65us'] = sum(e['down_us'] > 65 for e in entries)
    for token in range(tokens):
        result['slow_layers_by_token'].append([e['layer'] for e in groups['slow_gate_up'] if e['token'] == token])
    for layer in range(40):
        layer_entries = [e for e in observations if e['layer'] == layer]
        result['per_layer'].append({'layer': layer, 'slow_count': sum(e['gate_up_us'] > 80 for e in layer_entries),
                                    'gate_up_mean_us': statistics.mean(e['gate_up_us'] for e in layer_entries),
                                    'down_mean_us': statistics.mean(e['down_us'] for e in layer_entries)})
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('trace', type=Path)
    p.add_argument('--tokens', type=int, default=8)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    result = analyze(args.trace, args.tokens)
    text = json.dumps(result, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    else:
        print(text, end='')


if __name__ == '__main__':
    main()
