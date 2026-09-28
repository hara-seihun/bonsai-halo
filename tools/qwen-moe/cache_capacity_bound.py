#!/usr/bin/env python3
"""Bound serial MoE expert-byte retention at arbitrary cache granularity (CPU)."""
import argparse
import hashlib
import json
import struct
from pathlib import Path

CAPACITY = 32 * 1024 * 1024
LAYERS = 40
EXPERTS = 256
ROUTED = 8


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_split(root, split):
    rows, hashes = [], {}
    for layer in range(LAYERS):
        path = root / f'{split}.layer-{layer}.i32'
        raw = path.read_bytes()
        assert len(raw) % (4 * ROUTED) == 0
        values = struct.unpack(f'<{len(raw) // 4}i', raw)
        layer_rows = [values[i:i + ROUTED] for i in range(0, len(values), ROUTED)]
        assert all(len(set(row)) == ROUTED and all(0 <= e < EXPERTS for e in row) for row in layer_rows)
        rows.append(layer_rows)
        hashes[path.name] = sha(path)
    assert len({len(layer) for layer in rows}) == 1
    return rows, hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture', type=Path)
    parser.add_argument('inventory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    inventory = json.loads(args.inventory.read_text())
    assert inventory['expert_count'] == EXPERTS and inventory['experts_per_token'] == ROUTED
    image_bytes = [inventory['layers'][str(i)]['bank_bytes'] // EXPERTS for i in range(LAYERS)]
    assert all(image_bytes[i] * EXPERTS == inventory['layers'][str(i)]['bank_bytes'] for i in range(LAYERS))
    assert ROUTED * sum(image_bytes) == inventory['active_routed_bytes_per_token']
    stream = inventory['one_token_weight_stream_bytes']
    previous = json.loads((args.capture / 'global-cache.json').read_text())
    assert previous['inventory_sha256'] == sha(args.inventory)
    assert previous['capacity_bytes'] == CAPACITY
    result = {
        'contract': 'Cold serial generated steps, unique packed expert-byte addresses per (layer, expert); each selected byte is demanded once per step. Fully associative 32 MiB cache, arbitrary byte/line/subimage placement, free bypass, free clairvoyance and all other traffic bypasses. Only bytes present before a step can avoid a unique demand read; prefetch bytes count as reads. No changed image, producer/consumer map, cross-step speculation or within-step rereads. Conditional one-read model, not physical DRAM bytes or native time.',
        'source_sha256': sha(Path(__file__)),
        'inventory_sha256': sha(args.inventory),
        'previous_full_image_receipt_sha256': sha(args.capture / 'global-cache.json'),
        'model_sha256': previous['model_sha256'],
        'cache_capacity_bytes': CAPACITY,
        'one_step_expert_bytes': inventory['active_routed_bytes_per_token'],
        'one_step_complete_bytes': stream,
        'steady_expert_fraction_ceiling': CAPACITY / inventory['active_routed_bytes_per_token'],
        'steady_complete_fraction_ceiling': CAPACITY / stream,
        'splits': {},
    }
    for split in ('train', 'held'):
        rows, hashes = load_split(args.capture, split)
        assert hashes == previous['splits'][split]['capture_sha256']
        steps = len(rows[0])
        assert steps == previous['splits'][split]['steps']
        # Distinct layer/expert images are disjoint byte ranges. This overlap is
        # not needed for the universal capacity proof, but prices the route data.
        adjacent_bytes = [sum(len(set(rows[layer][step - 1]) & set(rows[layer][step])) * image_bytes[layer]
                              for layer in range(LAYERS)) for step in range(1, steps)]
        ceiling = (steps - 1) * CAPACITY
        total = steps * stream
        prior_full_ceiling = previous['splits'][split]['rigorous_saved_byte_ceiling']
        assert previous['splits'][split]['complete_one_read_bytes'] == total
        assert prior_full_ceiling > ceiling
        result['splits'][split] = {
            'steps': steps, 'capture_sha256': hashes,
            'adjacent_full_image_overlap_bytes_per_transition': adjacent_bytes,
            'adjacent_min_bytes': min(adjacent_bytes),
            'adjacent_mean_bytes': sum(adjacent_bytes) / len(adjacent_bytes),
            'adjacent_transitions_above_capacity': sum(x >= CAPACITY for x in adjacent_bytes),
            'cold_max_saved_bytes_any_granularity': ceiling,
            'cold_complete_one_read_bytes': total,
            'cold_complete_fraction_ceiling': ceiling / total,
            'previous_full_image_loose_ceiling_bytes': prior_full_ceiling,
            'previous_feasible_16_image_oracle_saved_bytes': previous['splits'][split]['offline16_saved_bytes'],
        }
        print(split, 'steps', steps, 'max saved', ceiling,
              'whole-stream fraction', ceiling / total,
              'min adjacent overlap bytes', min(adjacent_bytes))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')


if __name__ == '__main__':
    main()
