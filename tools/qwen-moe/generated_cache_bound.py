#!/usr/bin/env python3
"""Offline full-image cache bound on serial generated forty-layer Qwen routes (CPU)."""
import argparse
import hashlib
import json
import struct
from collections import Counter
from pathlib import Path

LAYERS, EXPERTS, ROUTED = 40, 256, 8
CAPACITY_BYTES = 32 * 1024 * 1024


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(root, split):
    arrays, hashes = [], {}
    for layer in range(LAYERS):
        path = root / f'{split}.layer-{layer}.i32'
        raw = path.read_bytes()
        assert len(raw) % (4 * ROUTED) == 0
        ids = struct.unpack(f'<{len(raw) // 4}i', raw)
        rows = [ids[j:j + ROUTED] for j in range(0, len(ids), ROUTED)]
        assert all(len(set(row)) == ROUTED and all(0 <= x < EXPERTS for x in row) for row in rows)
        arrays.append(rows)
        hashes[path.name] = digest(path)
    assert len({len(rows) for rows in arrays}) == 1
    return arrays, hashes


def accesses(arrays):
    steps = len(arrays[0])
    return [(step, layer, expert) for step in range(steps)
            for layer in range(LAYERS) for expert in arrays[layer][step]]


def simulate(requests, slots, policy):
    next_at = [len(requests)] * len(requests)
    next_seen = {}
    for i in range(len(requests) - 1, -1, -1):
        key = requests[i][1:]
        next_at[i] = next_seen.get(key, len(requests))
        next_seen[key] = i
    cache = {}  # image -> next use, or last use for LRU
    hits = [0] * (len(requests) // (LAYERS * ROUTED))
    per_layer = [0] * LAYERS
    for i, (step, layer, expert) in enumerate(requests):
        key = (layer, expert)
        if key in cache:
            hits[step] += 1
            per_layer[layer] += 1
        if policy == 'clairvoyant':
            cache[key] = next_at[i]
            if len(cache) > slots:
                evict = max(cache, key=lambda k: (cache[k], k))
                del cache[evict]
        else:
            cache[key] = i
            if len(cache) > slots:
                evict = min(cache, key=lambda k: (cache[k], k))
                del cache[evict]
    return {'hits': sum(hits), 'hits_per_token': hits, 'hits_per_layer': per_layer}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('capture', type=Path)
    p.add_argument('inventory', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    inventory = json.loads(a.inventory.read_text())
    assert inventory['expert_count'] == EXPERTS and inventory['experts_per_token'] == ROUTED
    sizes = [inventory['layers'][str(i)]['bank_bytes'] // EXPERTS for i in range(LAYERS)]
    assert all(s * EXPERTS == inventory['layers'][str(i)]['bank_bytes'] for i, s in enumerate(sizes))
    assert 16 * max(sizes) <= CAPACITY_BYTES < 18 * min(sizes)
    result = {'contract': 'Cold full-image cache over chronological (generated token, layer, selected expert) accesses. Distinct expert banks have distinct identities. Free expert reordering within a layer, free bypass, zero cache metadata/replacement work, no activation or nonexpert occupants, no measured DRAM traffic. The top-k capture cuts fusion and is not the uninstrumented route map.',
              'source_sha256': digest(Path(__file__)), 'inventory_sha256': digest(a.inventory),
              'capture_observer_sha256': digest(Path(__file__).with_name('capture_generated_routes.cpp')),
              'capacity_bytes': CAPACITY_BYTES, 'image_size_bytes': dict(Counter(sizes)),
              'minimum_image_bytes': min(sizes), 'maximum_image_bytes': max(sizes),
              'model_sha256': json.loads((a.capture.parent / 'acquisition.json').read_text())['sha256'],
              'splits': {}}
    for split in ('train', 'held'):
        arrays, hashes = load(a.capture, split)
        requests = accesses(arrays)
        assert len(requests) == len(arrays[0]) * LAYERS * ROUTED
        oracle16 = simulate(requests, 16, 'clairvoyant')
        oracle17 = simulate(requests, 17, 'clairvoyant')
        lru16 = simulate(requests, 16, 'lru')
        assert lru16['hits'] <= oracle16['hits'] <= oracle17['hits']
        saving16 = sum(n * sizes[layer] for layer, n in enumerate(oracle16['hits_per_layer']))
        saving17 = sum(n * sizes[layer] for layer, n in enumerate(oracle17['hits_per_layer']))
        full_bytes = len(arrays[0]) * inventory['one_token_weight_stream_bytes']
        # The 17-slot optimum maximizes hits, not bytes with unequal images.
        # Give every such hit the largest image for a rigorous byte ceiling.
        byte_ceiling = oracle17['hits'] * max(sizes)
        result['splits'][split] = {'steps': len(arrays[0]), 'capture_sha256': hashes,
                                   'requests': len(requests), 'routed_request_bytes': len(arrays[0]) * inventory['active_routed_bytes_per_token'],
                                   'complete_one_read_bytes': full_bytes,
                                   'lru16': lru16, 'offline16': oracle16, 'offline17_hit_ceiling': oracle17,
                                   'offline16_saved_bytes': saving16, 'offline17_saved_bytes_uniform_hit_schedule': saving17,
                                   'rigorous_saved_byte_ceiling': byte_ceiling,
                                   'rigorous_complete_stream_fraction': byte_ceiling / full_bytes}
        print(split, 'steps', len(arrays[0]), 'LRU16', lru16['hits'], 'offline16', oracle16['hits'],
              'offline17', oracle17['hits'], 'ceiling fraction', byte_ceiling / full_bytes)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')


if __name__ == '__main__':
    main()
