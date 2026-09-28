#!/usr/bin/env python3
"""Exact offline fractional-image cache optimum for actual tiled prompt routes."""
import argparse
import json
from pathlib import Path

from route_locality import EXPERTS, LAYERS, ROUTED, read_split, sha

CAPACITY = 32 * 1024 * 1024
WIDTHS = (8, 16, 32, 64)


def panel(ids, image_bytes, width):
    tiles = [set(map(int, ids[start:start + width].flat)) for start in range(0, len(ids), width)]
    # Every address of an image has the same demand schedule. Fractional
    # retention represents arbitrary, independently addressable bytes, with
    # an exact integer count (not a continuous relaxation).
    cache = {}
    demanded = saved = 0
    boundaries = []
    for i, tile in enumerate(tiles):
        demanded += len(tile) * image_bytes
        hit = sum(cache.get(expert, 0) for expert in tile)
        saved += hit
        available = set(cache) | tile
        order = sorted(available, key=lambda e: (
            next((j for j in range(i + 1, len(tiles)) if e in tiles[j]), len(tiles)), e))
        remaining = CAPACITY
        next_cache = {}
        for expert in order:
            if remaining == 0:
                break
            amount = min(image_bytes, remaining)
            next_cache[expert] = amount
            remaining -= amount
        cache = next_cache
        boundaries.append({'selected': len(tile), 'hits_bytes': hit,
                           'resident_bytes': sum(cache.values())})
    assert saved <= demanded
    return {'tiles': len(tiles), 'no_cache_bytes': demanded,
            'offline_min_read_bytes': demanded - saved, 'offline_max_saved_bytes': saved,
            'per_tile': boundaries}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('capture', type=Path)
    p.add_argument('inventory', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    inventory = json.loads(args.inventory.read_text())
    previous = json.loads((args.capture / 'window-bound.json').read_text())
    assert inventory['expert_count'] == EXPERTS and inventory['experts_per_token'] == ROUTED
    assert previous['inventory_sha256'] == sha(args.inventory)
    result = {'contract': 'Cold per layer; one read per selected expert image per contiguous token tile, with free within-tile reorder, bypass and clairvoyant fully associative 32 MiB arbitrary-byte cache. Each byte is demanded once per tile; other occupants and work omitted. Conditional logical reads, not measured DRAM or TPS.',
              'source_sha256': sha(Path(__file__)), 'inventory_sha256': sha(args.inventory),
              'previous_receipt_sha256': sha(args.capture / 'window-bound.json'),
              'model_sha256': previous['model_sha256'], 'cache_capacity_bytes': CAPACITY,
              'splits': {}}
    for split in ('train', 'held'):
        layers, hashes = read_split(args.capture, split)
        assert hashes == previous['capture_sha256'][split]
        result['splits'][split] = {}
        for width in WIDTHS:
            cases = []
            for layer, ids in enumerate(layers):
                bank = inventory['layers'][str(layer)]['bank_bytes']
                assert bank % EXPERTS == 0
                cases.append(panel(ids, bank // EXPERTS, width))
            totals = {k: sum(x[k] for x in cases) for k in
                      ('no_cache_bytes', 'offline_min_read_bytes', 'offline_max_saved_bytes')}
            full = previous['splits'][split][str(width)]['16']
            assert totals['no_cache_bytes'] == full['compulsory_bytes']
            assert totals['offline_min_read_bytes'] <= full['next_use_schedule_bytes']
            result['splits'][split][str(width)] = {**totals,
                'full_16_image_optimum_bytes': full['next_use_schedule_bytes'],
                'incremental_subimage_ceiling_bytes': full['next_use_schedule_bytes'] - totals['offline_min_read_bytes'],
                'per_layer': cases}
            print(split, width, totals['no_cache_bytes'], full['next_use_schedule_bytes'],
                  totals['offline_min_read_bytes'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
