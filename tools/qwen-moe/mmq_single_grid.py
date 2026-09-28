#!/usr/bin/env python3
"""Check a one-grid heterogeneous J16/J64 stripe map on captured routes.

This is a dispatch-coordinate construction, not a GPU kernel. J16 has 128
threads and J64 has 256: the proposed 256-thread CTA gives two independent
128-thread J16 half-CTAs one native tile each per iteration.
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from expert_occupancy import route_rows
from mmq_indirect import construct, decode

DEFAULT = Path('../../data/qwen-moe/all-layer-routes')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def visit(counts, k, down=False):
    """One grid (expert, stripe); J16 stripes carry two independent half-CTAs."""
    emitted = {16: [], 64: []}
    active = Counter()
    chain = Counter()
    for e in range(256):
        bodies = [j for j in (16, 64) if counts[j][e]]
        assert len(bodies) <= 1
        if not bodies:
            continue
        j = bodies[0]
        nrow = (16 if j == 16 else 8) * (2 if down else 1)
        n = counts[j][e] * (2 if down else 1)
        for stripe in range(k):
            lanes = 2 if j == 16 else 1
            lanes_active = False
            for half in range(lanes):
                tiles = list(range(stripe * lanes + half, n, k * lanes))
                if tiles:
                    lanes_active = True
                    chain[j] = max(chain[j], len(tiles))
                for t in tiles:
                    column, row = divmod(t, nrow)
                    emitted[j].append((e, column, row))
            active[j] += lanes_active
    return emitted, active, chain


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture-dir', type=Path, default=DEFAULT)
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    raw = (a.capture_dir / 'occupancy.json').read_bytes()
    occupancy = json.loads(raw)
    result = {
        'contract': 'One 256*k, 256-thread CTA grid per projection; CTA(e,s) chooses its unique J16/J64 group from expert_bounds. J64 visits s,s+k,...; two disjoint 128-thread J16 halves visit 2s+h,2s+h+2k,... for h=0,1. Down has twice the output row tiles. CPU coordinate equality only. Native J16 uses 128 threads and J64 uses 256: a real fused kernel requires separate half-CTA synchronization/LDS and a measured union resource budget. No source-level fusion, native time or FP32 bit proof.',
        'source_sha256': sha(Path(__file__).read_bytes()),
        'occupancy_sha256': sha(raw),
        'model_sha256': occupancy['model_sha256'],
        'native_source_revision': occupancy['native_source_revision'],
        'native_source_sha256': occupancy['native_source_sha256'],
        'capture_receipt_sha256': occupancy['capture_receipt_sha256'],
        'splits': {},
    }
    for split, record in occupancy['splits'].items():
        metrics = {}
        for k in (4, 8, 16):
            metrics[str(k)] = {'gate': Counter(), 'down': Counter(), 'min_j64_active_per_layer': 256*k,
                               'max_j64_active_per_layer': 0, 'min_total_active_per_layer': 256*k}
        digest = hashlib.sha256()
        for layer, expected in enumerate(record['capture_sha256']['layers']):
            data = (a.capture_dir / f'{split}.layer-{layer}.i32').read_bytes()
            assert sha(data) == expected
            rows = route_rows(a.capture_dir / f'{split}.layer-{layer}.i32', record['tokens'])
            for first in range(0, len(rows), 128):
                counts, _, words, _ = construct(rows[first:first+128])
                reference = {j: [(e, col, row) for word in words[j]
                                 for e, col, row, body in (decode(word),) if body == j] for j in (16, 64)}
                assert all(len(set(reference[j])) == len(reference[j]) for j in (16, 64))
                for k, item in metrics.items():
                    k = int(k)
                    for name, is_down in (('gate', False), ('down', True)):
                        emitted, active, chain = visit(counts, k, is_down)
                        for j in (16, 64):
                            expected_tiles = ([(e, col, 2*row+p) for e, col, row in reference[j]
                                               for p in (0, 1)] if is_down else reference[j])
                            assert len(emitted[j]) == len(expected_tiles)
                            assert set(emitted[j]) == set(expected_tiles)
                            digest.update(bytes((layer, first, k, int(is_down), j)))
                            for e, col, row in emitted[j]:
                                digest.update(bytes((e, col, row)))
                        m = item[name]
                        m['launched'] += 256*k
                        m['active'] += sum(active.values())
                        m['tasks'] += sum(len(v) for v in emitted.values())
                        m['max_chain'] = max(m['max_chain'], *chain.values())
                        m['j16_active'] += active[16]
                        m['j16_max_chain'] = max(m['j16_max_chain'], chain[16])
                        m['j64_active'] += active[64]
                        m['j64_max_chain'] = max(m['j64_max_chain'], chain[64])
                        if not is_down:
                            item['min_j64_active_per_layer'] = min(item['min_j64_active_per_layer'], active[64])
                            item['max_j64_active_per_layer'] = max(item['max_j64_active_per_layer'], active[64])
                            item['min_total_active_per_layer'] = min(item['min_total_active_per_layer'], sum(active.values()))
        result['splits'][split] = {'coordinate_sha256': digest.hexdigest(),
                                   'stripes': {k: {'gate': dict(v['gate']), 'down': dict(v['down']),
                                                   'min_j64_active_per_layer': v['min_j64_active_per_layer'],
                                                   'max_j64_active_per_layer': v['max_j64_active_per_layer'],
                                                   'min_total_active_per_layer': v['min_total_active_per_layer']}
                                               for k, v in metrics.items()}}
    text = json.dumps(result, indent=2) + '\n'
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(text)
    else:
        print(text)


if __name__ == '__main__':
    main()
