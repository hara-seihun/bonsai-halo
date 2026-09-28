#!/usr/bin/env python3
"""Logical expert image reads for actual all-layer prompt routes; no GPU counters."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

LAYERS = 40
EXPERTS = 256
ROUTED = 8
WIDTHS = (1, 8, 16, 32, 64, 128, 256)


def sha(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_split(root, split):
    token_file = root / f'{split}.tokens'
    count = token_file.stat().st_size // 4
    assert count * 4 == token_file.stat().st_size and count > 0
    paths = [root / f'{split}.layer-{layer}.i32' for layer in range(LAYERS)]
    arrays = [np.fromfile(path, dtype=np.int32).reshape(count, ROUTED) for path in paths]
    for ids in arrays:
        assert np.all((ids >= 0) & (ids < EXPERTS))
        assert all(len(set(row)) == ROUTED for row in ids)
    return arrays, {'tokens': sha(token_file), 'layers': [sha(path) for path in paths]}


def panel(arrays, inventory):
    per_layer = []
    by_width = {str(width): 0 for width in WIDTHS}
    naive = 0
    for layer, ids in enumerate(arrays):
        bank = inventory['layers'][str(layer)]['bank_bytes']
        assert bank % EXPERTS == 0
        size = bank // EXPERTS
        counts = {}
        for width in WIDTHS:
            reads = sum(len(np.unique(ids[start:start + width])) for start in range(0, len(ids), width))
            counts[str(width)] = reads
            by_width[str(width)] += reads * size
        per_layer.append({'layer': layer, 'expert_image_bytes': size,
                          'unique_experts': counts['256'], 'logical_image_reads': counts,
                          'max_assignment_count': max(Counter(ids.flat).values())})
        naive += ids.size * size
    assert naive == by_width['1']
    return {'tokens': len(arrays[0]), 'assignments': sum(ids.size for ids in arrays),
            'naive_image_bytes': naive, 'image_bytes_by_tile_width': by_width,
            'layer': per_layer,
            'one_read_lower_bound_bytes': by_width['256']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture', type=Path)
    parser.add_argument('inventory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    inventory = json.loads(args.inventory.read_text())
    assert inventory['expert_count'] == EXPERTS and inventory['experts_per_token'] == ROUTED
    captures = {}
    panels = {}
    for split in ('train', 'held'):
        ids, captures[split] = read_split(args.capture, split)
        panels[split] = panel(ids, inventory)
        # Reconcile this capture with the independent, five-node layer-0 observer.
        prior = args.capture.parent / 'route-capture' / f'{split}.0.ffn_moe_topk-0.bin'
        assert sha(prior) == captures[split]['layers'][0], f'layer-0 observer mismatch: {split}'
        captures[split]['prior_layer0_sha256'] = sha(prior)
    model = args.capture.parent / 'Qwen3.6-35B-A3B-UD-Q4_K_M.gguf'
    binary = args.capture / 'capture-all-routes'
    result = {'contract': 'Captured callback-cut prompt routes; each logical image is a full selected packed expert; no DRAM counter or native dispatch claim',
              'grammar': 'For each layer and tile, reorder token assignments freely by expert; one full-image read per distinct expert. Tiles execute sequentially, no retention across tiles. Width 256 covers each entire captured prompt and is the compulsory one-read floor in this grammar.',
              'source_sha256': {'capture': sha(Path(__file__).with_name('capture_all_routes.cpp')),
                                'analysis': sha(Path(__file__))},
              'binary_sha256': sha(binary), 'inventory_sha256': sha(args.inventory),
              'model_sha256': json.loads((args.capture.parent / 'acquisition.json').read_text())['sha256'],
              'model_path': str(model), 'capture_sha256': captures,
              'logs_sha256': {split: sha(args.capture / f'{split}.log') for split in panels},
              'splits': panels}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    for split, data in panels.items():
        print(split, 'tokens', data['tokens'], 'unique per layer',
              min(x['unique_experts'] for x in data['layer']),
              np.median([x['unique_experts'] for x in data['layer']]),
              max(x['unique_experts'] for x in data['layer']))
        print(split, 'MB by tile', {width: round(n / 1e6, 3) for width, n in data['image_bytes_by_tile_width'].items()})
    print(args.output)


if __name__ == '__main__':
    main()
