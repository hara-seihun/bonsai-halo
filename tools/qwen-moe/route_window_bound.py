#!/usr/bin/env python3
"""Finite-image-cache bounds for actual all-layer Qwen prompt routes (CPU only)."""
import argparse
import json
from pathlib import Path

from route_locality import EXPERTS, LAYERS, ROUTED, read_split, sha

CAPACITIES = (0, 8, 16, 32, 64)
WIDTHS = (8, 16, 32, 64, 128)


def layer_panel(ids, image_bytes, width, capacity):
    tiles = [set(map(int, ids[start:start + width].flat)) for start in range(0, len(ids), width)]
    seen = set()
    compulsory = 0
    optimistic = 0
    constructive = 0
    cache = set()
    for index, tile in enumerate(tiles):
        compulsory += len(tile)
        # Every cache image must have appeared earlier. This grants a different
        # clairvoyant cache state for each boundary, so it is a lower bound.
        optimistic += len(tile) - min(capacity, len(tile & seen))
        constructive += len(tile - cache)
        seen |= tile
        # Reorder accesses within the tile, bypassing loads not retained. Every
        # cached hit remains available throughout the tile; no extra reads.
        available = cache | tile
        def next_use(expert):
            return next((j for j in range(index + 1, len(tiles)) if expert in tiles[j]), len(tiles))
        cache = set(sorted(available, key=lambda e: (next_use(e), e))[:capacity])
    assert compulsory == sum(map(len, tiles))
    assert optimistic <= constructive <= compulsory
    return {'tiles': len(tiles), 'compulsory_image_reads': compulsory,
            'clairvoyant_lower_image_reads': optimistic,
            'next_use_schedule_image_reads': constructive,
            'compulsory_bytes': compulsory * image_bytes,
            'clairvoyant_lower_bytes': optimistic * image_bytes,
            'next_use_schedule_bytes': constructive * image_bytes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture', type=Path)
    parser.add_argument('inventory', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    inventory = json.loads(args.inventory.read_text())
    assert inventory['expert_count'] == EXPERTS and inventory['experts_per_token'] == ROUTED
    splits = {}
    capture_hashes = {}
    for split in ('train', 'held'):
        arrays, capture_hashes[split] = read_split(args.capture, split)
        result = {}
        for width in WIDTHS:
            result[str(width)] = {}
            for capacity in CAPACITIES:
                totals = {'compulsory_image_reads': 0, 'clairvoyant_lower_image_reads': 0,
                          'next_use_schedule_image_reads': 0, 'compulsory_bytes': 0,
                          'clairvoyant_lower_bytes': 0, 'next_use_schedule_bytes': 0}
                per_layer = []
                for layer, ids in enumerate(arrays):
                    bank = inventory['layers'][str(layer)]['bank_bytes']
                    assert bank % EXPERTS == 0
                    case = layer_panel(ids, bank // EXPERTS, width, capacity)
                    per_layer.append(case)
                    for key in totals:
                        totals[key] += case[key]
                result[str(width)][str(capacity)] = {**totals, 'per_layer': per_layer,
                                                     'resident_image_capacity': capacity,
                                                     'tile_token_capacity': width}
        splits[split] = result
    output = {'contract': 'One packed full-expert image per selected (layer,expert), disjoint layer images, cold per layer. Within a contiguous prompt tile assignments may reorder freely by expert. Images not retained may bypass the bounded cache. Other GPU cache occupants, scattering and activation work omitted; no native speed or physical DRAM claim.',
              'lower_bound': 'For each tile at most C distinct images already seen in the layer can be cached before it. Grant an independent clairvoyant choice at each boundary (not necessarily jointly feasible).',
              'constructive': 'Offline next-use retention from the previous cache union current tile, with free within-tile reordering and cache bypass. Feasible within the stated grammar, not necessarily optimal.',
              'source_sha256': sha(Path(__file__)), 'inventory_sha256': sha(args.inventory),
              'capture_sha256': capture_hashes,
              'model_sha256': json.loads((args.capture.parent / 'acquisition.json').read_text())['sha256'],
              'splits': splits}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + '\n')
    for split, cases in splits.items():
        for width in (32, 64):
            print(split, width, {c: tuple(round(cases[str(width)][str(c)][key] / 1e9, 3)
                                       for key in ('compulsory_bytes', 'clairvoyant_lower_bytes', 'next_use_schedule_bytes'))
                                    for c in CAPACITIES})
    print(args.output)


if __name__ == '__main__':
    main()
