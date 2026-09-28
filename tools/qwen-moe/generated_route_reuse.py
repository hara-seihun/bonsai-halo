#!/usr/bin/env python3
"""Price next-token expert-image retention from serial generated route captures."""
import argparse
import hashlib
import json
import struct
from pathlib import Path

LAYERS = 40
EXPERTS = 256
ROUTED = 8


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capture(root, split):
    arrays, hashes = [], {}
    for layer in range(LAYERS):
        path = root / f'{split}.layer-{layer}.i32'
        raw = path.read_bytes()
        assert len(raw) % (4 * ROUTED) == 0
        ids = struct.unpack(f'<{len(raw) // 4}i', raw)
        rows = [ids[i:i + ROUTED] for i in range(0, len(ids), ROUTED)]
        assert all(len(set(row)) == ROUTED and all(0 <= x < EXPERTS for x in row) for row in rows)
        arrays.append(rows)
        hashes[path.name] = sha(path)
    assert len({len(rows) for rows in arrays}) == 1
    path = root / f'{split}.tokens'
    raw = path.read_bytes()
    steps = len(arrays[0])
    assert len(raw) == 8 * steps
    plain, observed = struct.unpack(f'<{steps * 2}i', raw)[:steps], struct.unpack(f'<{steps * 2}i', raw)[steps:]
    hashes[path.name] = sha(path)
    return arrays, hashes, sum(a != b for a, b in zip(plain, observed)), steps


def analyze(arrays, inventory):
    per_layer = []
    total_hits = total_request = total_bytes = total_hit_bytes = 0
    split_hits = [0, 0]
    for layer, rows in enumerate(arrays):
        image_bytes = inventory['layers'][str(layer)]['bank_bytes'] // EXPERTS
        assert image_bytes * EXPERTS == inventory['layers'][str(layer)]['bank_bytes']
        hits = [len(set(a) & set(b)) for a, b in zip(rows[:-1], rows[1:])]
        n = len(hits)
        per_layer.append({'layer': layer, 'overlap': sum(hits), 'transitions': n,
                          'image_bytes': image_bytes, 'half_overlaps': [sum(hits[:n//2]), sum(hits[n//2:])]})
        total_hits += sum(hits)
        total_request += ROUTED * n
        total_hit_bytes += sum(hits) * image_bytes
        total_bytes += ROUTED * n * image_bytes
        split_hits[0] += sum(hits[:n//2]); split_hits[1] += sum(hits[n//2:])
    return {'overlap': total_hits, 'assignments': total_request, 'hit_fraction': total_hits / total_request,
            'full_image_request_bytes': total_bytes, 'oracle_previous_image_hit_bytes': total_hit_bytes,
            'half_overlaps': split_hits, 'per_layer': per_layer}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture', type=Path)
    parser.add_argument('inventory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    inventory = json.loads(args.inventory.read_text())
    assert inventory['expert_count'] == EXPERTS and inventory['experts_per_token'] == ROUTED
    result = {}
    for split in ('train', 'held'):
        arrays, hashes, mismatches, steps = capture(args.capture, split)
        result[split] = {'steps': steps, 'plain_vs_observed_token_mismatches': mismatches,
                         'files_sha256': hashes, **analyze(arrays, inventory)}
    receipt = {
        'contract': 'Single-token autoregressive greedy continuation after first 64 prompt tokens; callback requests top-k views and cuts fusion on observed arm. Plain same-process control has callback disabled and greedy tokens compared. Previous eight same-layer expert images retained without a capacity or intervening-layer charge; logical full-image-byte oracle, not measured DRAM or native throughput.',
        'source_sha256': sha(Path(__file__)),
        'observer_source_sha256': sha(Path(__file__).with_name('capture_generated_routes.cpp')),
        'observer_binary_sha256': sha(args.capture / 'capture-generated-routes'),
        'inventory_sha256': sha(args.inventory),
        'model_sha256': json.loads((args.capture.parent / 'acquisition.json').read_text())['sha256'],
        'text_sha256': {s: sha(args.capture.parent / 'route-capture' / f'{s}.txt') for s in result},
        'runtime_sha256': {name: sha(args.capture.parent / 'runtime/current/bin' / name)
                           for name in ('libllama.so', 'libggml-hip.so')},
        'runtime_selection_sha256': sha(args.capture.parent / 'runtime/current/selection.json'),
        'logs_sha256': {s: sha(args.capture / f'{s}.log') for s in result},
        'results': result,
    }
    args.output.write_text(json.dumps(receipt, indent=2) + '\n')
    for split, value in result.items():
        print(split, value['overlap'], value['assignments'], value['hit_fraction'],
              value['oracle_previous_image_hit_bytes'], value['plain_vs_observed_token_mismatches'])


if __name__ == '__main__':
    main()
