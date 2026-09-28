#!/usr/bin/env python3
"""Test train-fitted down-column zero layouts on actual forty-layer MoE producers."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

CAPTURE = Path('../../data/qwen-moe/all-down-zero')
TRAFFIC = Path('../../data/qwen-moe/traffic.json')
COMPLETE_BYTES = 2_626_187_904


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def capture(root, split, layer, hashes):
    paths = [root / f'{split}.layer-{layer}.ffn_moe_topk.i32',
             root / f'{split}.layer-{layer}.ffn_moe_swiglu.f32']
    for p in paths:
        hashes[p.name] = digest(p)
    ids = np.fromfile(paths[0], '<i4').reshape(64, 8)
    hidden = np.fromfile(paths[1], '<f4').reshape(64, 8, 512)
    assert np.all((ids >= 0) & (ids < 256)) and np.isfinite(hidden).all()
    return ids.reshape(-1), hidden.reshape(-1, 512) == 0


def panel(train_id, train_z, held_id, held_z, width):
    counts = {k: 0 for k in ('original', 'static', 'oracle', 'tiles', 'slots', 'unseen_slots',
                              'shared_experts', 'shared_conflicts', 'train_varying', 'held_varying',
                              'train_seen', 'held_seen', 'static_shared', 'oracle_shared')}
    for expert in range(256):
        tr = train_z[train_id == expert]
        he = held_z[held_id == expert]
        if len(tr):
            counts['train_seen'] += 1
            counts['train_varying'] += int(np.any(tr != tr[0]))
        if not len(he):
            continue
        counts['held_seen'] += 1
        counts['held_varying'] += int(np.any(he != he[0]))
        if len(tr):
            counts['shared_experts'] += 1
            counts['shared_conflicts'] += int(np.any(tr[0] != he[0]))
            # Least frequently active channels first; stable index tie-breaking.
            perm = np.argsort((~tr).sum(axis=0), kind='stable')
        else:
            perm = np.arange(512)
            counts['unseen_slots'] += len(he)
        for start in range(0, len(he), width):
            tile = he[start:start + width]
            common = np.all(tile, axis=0)
            original = int(np.all(common.reshape(16, 32), axis=1).sum())
            static = int(np.all(common[perm].reshape(16, 32), axis=1).sum())
            oracle = int(common.sum() // 32)
            assert original <= oracle and static <= oracle
            counts['original'] += original
            counts['static'] += static
            counts['oracle'] += oracle
            counts['tiles'] += 1
            counts['slots'] += len(tile)
            if len(tr):
                counts['static_shared'] += static
                counts['oracle_shared'] += oracle
    return counts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture', type=Path, default=CAPTURE)
    p.add_argument('--traffic', type=Path, default=TRAFFIC)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    traffic = json.loads(args.traffic.read_text())
    hashes = {}
    layers = []
    for layer in range(40):
        tr_id, tr_z = capture(args.capture, 'train', layer, hashes)
        he_id, he_z = capture(args.capture, 'held', layer, hashes)
        tensors = traffic['layers'][str(layer)]['tensors']
        down = next(t for t in tensors if 'ffn_down_exps.weight' in t['name'])
        assert down['bytes'] % 256 == 0
        data = {'layer': layer, 'down_type': down['type'], 'down_expert_bytes': down['bytes'] // 256,
                'held_zero_floats': int(he_z.sum()), 'train_zero_floats': int(tr_z.sum()),
                'panels': {str(w): panel(tr_id, tr_z, he_id, he_z, w) for w in (1, 16, 64)}}
        layers.append(data)
    totals = {}
    for width in (1, 16, 64):
        rows = [layer['panels'][str(width)] for layer in layers]
        sums = {key: sum(r[key] for r in rows) for key in rows[0]}
        # One-read comparator weights each routed slot, NOT each expert tile.
        # Tile observations have a separate weight-read grammar, so no tile
        # fraction is silently promoted to whole-model one-read bytes.
        if width == 1:
            savings = {arm: sum(layer['down_expert_bytes'] * layer['panels']['1'][arm] / 16 / 64
                                for layer in layers) for arm in ('original', 'static', 'oracle')}
            sums['conditional_one_read_bytes_per_token'] = savings
            sums['conditional_complete_percent'] = {k: 100 * v / COMPLETE_BYTES
                                                     for k, v in savings.items()}
        totals[str(width)] = sums
    result = {'contract': 'Observed original-float post-SwiGLU zero blocks; train-only per-(layer, expert) column permutation; fixed expert-major contiguous J-slot tiles. Free changed weight layout, testing and selective 32-column reads; not bit-identical native Q5/Q6 arithmetic, DRAM or model quality.',
              'source_sha256': digest(Path(__file__)), 'traffic_sha256': digest(args.traffic),
              'capture_receipt_sha256': digest(args.capture / 'receipt.json'),
              'capture_sha256': hashes, 'complete_one_read_bytes_per_token': COMPLETE_BYTES,
              'layers': layers, 'totals': totals}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    for width, row in totals.items():
        print(width, {k: row[k] for k in ('tiles', 'slots', 'static', 'oracle', 'shared_conflicts', 'train_varying', 'held_varying')},
              row.get('conditional_complete_percent', ''))


if __name__ == '__main__':
    main()
