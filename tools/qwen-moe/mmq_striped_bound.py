#!/usr/bin/env python3
"""Exact static CTA/chain frontier for builder-free striped routed MMQ.

Each (expert, J body, stripe) CTA owns tile indices s, s+k, ... . Down
stripes its own doubled row tiles. No native timing or FP32 claim follows.
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from expert_occupancy import route_rows
from mmq_indirect import construct, decode, encode

DEFAULT = Path('../../data/qwen-moe/all-layer-routes')
STRIPES = (1, 2, 4, 8, 16, 32)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def stripe_words(counts, k, down=False):
    by_body = {16: [], 64: []}
    active = {16: 0, 64: 0}
    deepest = {16: 0, 64: 0}
    for j in (16, 64):
        rows = 16 if j == 16 else 8
        for expert, n in enumerate(counts[j]):
            tasks = n * (2 if down else 1)
            active[j] += min(tasks, k)
            deepest[j] = max(deepest[j], (tasks + k - 1) // k)
            for stripe in range(k):
                for t in range(stripe, tasks, k):
                    if down:
                        column, row = divmod(t, rows * 2)
                        by_body[j].append((expert, column, row))
                    else:
                        column, row = divmod(t, rows)
                        by_body[j].append(encode(expert, column, row, j))
    return by_body, active, deepest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture-dir', type=Path, default=DEFAULT)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    raw = (args.capture_dir / 'occupancy.json').read_bytes()
    occupancy = json.loads(raw)
    result = {
        'contract': 'For each J16/J64 body launch 256*k CTAs. CTA (expert e,stripe s) reads native expert_bounds, chooses same J, and executes disjoint native tile indices s,s+k,...; down stripes its twofold row list separately. Fixed k per projection, no descriptor builder or atomic claims. Static task equality and CTA/chain counts, not native timing or FP32 acceptance.',
        'source_sha256': sha(Path(__file__).read_bytes()),
        'occupancy_sha256': sha(raw),
        'model_sha256': occupancy['model_sha256'],
        'native_source_revision': occupancy['native_source_revision'],
        'native_source_sha256': occupancy['native_source_sha256'],
        'capture_receipt_sha256': occupancy['capture_receipt_sha256'],
        'splits': {},
    }
    for split, record in occupancy['splits'].items():
        totals = {str(k): {'gate': Counter(), 'down': Counter(), 'min_j64_active_per_layer': 256 * k,
                           'max_j64_active_per_layer': 0} for k in STRIPES}
        layers = []
        for layer, expected in enumerate(record['capture_sha256']['layers']):
            path = args.capture_dir / f'{split}.layer-{layer}.i32'
            assert sha(path.read_bytes()) == expected
            rows = route_rows(path, record['tokens'])
            for first in range(0, len(rows), 128):
                counts, _, words, _ = construct(rows[first:first + 128])
                reference = {j: set(words[j]) for j in (16, 64)}
                assert all(len(reference[j]) == len(words[j]) for j in (16, 64))
                expected_down = {j: {(e, c, 2*r + parity) for w in words[j]
                                     for e, c, r, _ in (decode(w),) for parity in (0, 1)}
                                 for j in (16, 64)}
                layer_record = {'layer': layer, 'first': first, 'tasks': {str(j): len(words[j]) for j in (16, 64)}, 'stripes': {}}
                for k in STRIPES:
                    gate, ga, gd = stripe_words(counts, k)
                    down, da, dd = stripe_words(counts, k, True)
                    for j in (16, 64):
                        assert len(gate[j]) == len(reference[j]) and set(gate[j]) == reference[j]
                        # Down tile (column,row) = (column,2*gate_row+parity).
                        assert len(down[j]) == 2 * len(words[j]) and set(down[j]) == expected_down[j]
                    item = totals[str(k)]
                    item['min_j64_active_per_layer'] = min(item['min_j64_active_per_layer'], ga[64])
                    item['max_j64_active_per_layer'] = max(item['max_j64_active_per_layer'], ga[64])
                    for label, active, depth, factor in (('gate', ga, gd, 1), ('down', da, dd, 2)):
                        target = item[label]
                        target['launched'] += 512 * k
                        target['active'] += sum(active.values())
                        target['tasks'] += sum(counts[j][e] for j in (16, 64) for e in range(256)) * factor
                        target['max_chain'] = max(target['max_chain'], *depth.values())
                        target['j64_active'] += active[64]
                        target['j64_tasks'] += sum(counts[64]) * factor
                        target['j64_max_chain'] = max(target['j64_max_chain'], depth[64])
                    layer_record['stripes'][str(k)] = {'gate_j64_active': ga[64], 'gate_max_chain': max(gd.values()),
                                                       'down_max_chain': max(dd.values())}
                layers.append(layer_record)
        result['splits'][split] = {'stripes': {k: {**{'gate': dict(v['gate']), 'down': dict(v['down'])},
                                                  'min_j64_active_per_layer': v['min_j64_active_per_layer'],
                                                  'max_j64_active_per_layer': v['max_j64_active_per_layer']}
                                             for k, v in totals.items()}, 'layers': layers}
    text = json.dumps(result, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    else:
        print(text)


if __name__ == '__main__':
    main()
