#!/usr/bin/env python3
"""Test whether long profiler dispatch gaps alone locate slow Qwen expert calls."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from decode_trace import family


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def span(rows):
    kinds = [family(row) for row in rows]
    gates = [i for i, kind in enumerate(kinds) if kind == 'Q4_K/out512']
    routers = [i for i, kind in enumerate(kinds) if kind == 'router/topk']
    downs = [i for i, kind in enumerate(kinds) if kind in ('Q5_K/out2048', 'Q6_K/out2048')]
    if len(rows) != 1565 or not (len(gates) == len(routers) == len(downs) == 40):
        raise ValueError('unexpected full-decode dispatch inventory')
    gaps = [i for i in range(len(rows) - 1)
            if int(rows[i + 1]['Start_Timestamp']) - int(rows[i]['End_Timestamp']) >= 500000]
    cases = []
    for layer, (r, g, d) in enumerate(zip(routers, gates, downs)):
        if not r < g < d:
            raise ValueError(f'layer {layer} changed dispatch order')
        gate_us = (int(rows[g]['End_Timestamp']) - int(rows[g]['Start_Timestamp'])) / 1000
        # Only preceding gaps can possibly make a subsequent call slow.
        preceding = [g - p for p in gaps if p < g]
        cases.append({'layer': layer, 'gate_ordinal': g, 'router_ordinal': r,
                      'down_ordinal': d, 'gate_us': gate_us, 'slow': gate_us > 80,
                      'distance_after_long_gap': min(preceding) if preceding else None})
    return gaps, cases


def panel(cases):
    out = {'calls': len(cases), 'slow': sum(c['slow'] for c in cases)}
    for n in (1, 2, 4, 8, 16, 32, 64):
        near = [c for c in cases if c['distance_after_long_gap'] is not None
                and c['distance_after_long_gap'] <= n]
        far = [c for c in cases if c not in near]
        out[f'within_{n}_dispatches'] = {
            'near_calls': len(near), 'near_slow': sum(c['slow'] for c in near),
            'far_calls': len(far), 'far_slow': sum(c['slow'] for c in far),
        }
    return out


def analyze(trace, receipt):
    original = json.loads(receipt.read_text())
    relative = str(trace.relative_to(trace.parents[2]))
    if original['files'][relative] != sha(trace):
        raise ValueError('trace not covered by original measurement receipt')
    rows = list(csv.DictReader(trace.open(newline='')))
    heads = [i for i, row in enumerate(rows) if family(row) == 'Q6_K/out248320']
    if len(heads) != 10:
        raise ValueError('expected two synthetic-depth heads and eight measured heads')
    tokens = []
    for token, (a, b) in enumerate(zip(heads[-9:-1], heads[-8:])):
        part = sorted(rows[a + 1:b + 1], key=lambda row: int(row['Start_Timestamp']))
        if {row['Queue_Id'] for row in part} != {'2'}:
            raise ValueError('not a single-queue trace')
        gaps, cases = span(part)
        tokens.append({'token': token, 'long_gap_ordinals': gaps, 'cases': cases})
    train = [c for t in tokens if 1 <= t['token'] <= 3 for c in t['cases']]
    held = [c for t in tokens if 4 <= t['token'] <= 7 for c in t['cases']]
    return {'contract': 'One counter-free PROFILed depth-1024 selected Qwen decode; gaps >=0.5 ms; Q4 slow >80 us as preexisting frozen threshold. Preceding-gap distance is in dispatches, not wall time. Token 0 cold and two setup heads excluded. Association is not a native-time or causal intervention.',
            'source_commit': original['source_commit'], 'model_sha256': original['model_sha256'],
            'binary_hashes': original['binary_hashes'], 'trace_sha256': sha(trace),
            'original_receipt_sha256': sha(receipt), 'script_sha256': sha(Path(__file__)),
            'train': panel(train), 'held': panel(held), 'tokens': tokens}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace', type=Path)
    parser.add_argument('--receipt', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    result = analyze(args.trace, args.receipt)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'train': result['train'], 'held': result['held']}, indent=2))
