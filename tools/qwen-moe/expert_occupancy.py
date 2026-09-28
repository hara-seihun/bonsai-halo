#!/usr/bin/env python3
"""Count actual Qwen routed assignments per expert/tile without GPU execution."""
import hashlib
import json
import struct
from collections import Counter
from pathlib import Path

DATA = Path('../../data/qwen-moe/all-layer-routes')
WIDTHS = (8, 16, 32, 64, 128, 256)
NATIVE = Path('../bonsai-hip/ggml/src/ggml-cuda')
NATIVE_FILES = ('ggml-cuda.cu', 'mmq.cu', 'mmq.cuh', 'mmq-config-rdna3-5.cuh')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def route_rows(path, n):
    raw = path.read_bytes()
    assert len(raw) == n * 8 * 4, (path, len(raw), n)
    rows = list(zip(*[iter(struct.unpack(f'<{n * 8}i', raw))] * 8))
    assert all(len(set(row)) == 8 and all(0 <= e < 256 for e in row) for row in rows)
    return rows


def summarize(rows_by_layer, width):
    sizes = Counter()
    tiles = assignments = active = groups = 0
    for rows in rows_by_layer:
        for start in range(0, len(rows), width):
            counts = Counter(e for row in rows[start:start + width] for e in row)
            sizes.update(counts.values())
            tiles += 1
            active += len(counts)
            assignments += sum(counts.values())
            groups += sum((v + 15) // 16 for v in counts.values())
    # The selected MMQ chooses J from the complete tensor row count, not the
    # observed number of assignments for an individual expert. RDNA3.5 Q4_K
    # and Q5_K support J=16,32,48,... .
    J = min(128, (width + 15) // 16 * 16)
    fixed_slots = sum(((count + J - 1) // J) * J * multiplicity for count, multiplicity in sizes.items())
    sixteen_slots = sum(((count + 15) // 16) * 16 * multiplicity for count, multiplicity in sizes.items())
    adaptive_slots = sixteen_slots
    fixed_groups = sum(((count + J - 1) // J) * multiplicity for count, multiplicity in sizes.items())
    adaptive_groups = fixed_groups  # One matching-size tile per active expert, no repeated image.
    return {
        'width': width, 'layer_tiles': tiles, 'assignments': assignments,
        'active_expert_tiles': active, 'group_size_histogram': dict(sorted(sizes.items())),
        'single_assignment_fraction': sizes[1] / active,
        'assignments_in_groups_size_le_4': sum(k*v for k, v in sizes.items() if k <= 4) / assignments,
        'assignments_in_groups_size_ge_16': sum(k*v for k, v in sizes.items() if k >= 16) / assignments,
        'max_group_size': max(sizes), 'native_global_J': J,
        'fixed_J_lane_slots': fixed_slots, 'fixed_J_lane_fill': assignments / fixed_slots,
        'fixed_J_groups': fixed_groups, 'adaptive_J16_multiple_lane_slots': adaptive_slots,
        'adaptive_J16_multiple_groups': adaptive_groups,
        'per_expert_J16_lane_slots': sixteen_slots, 'per_expert_J16_lane_fill': assignments / sixteen_slots,
        'nonempty_J16_groups': groups,
    }


def main():
    capture = json.loads((DATA / 'receipt.json').read_text())
    result = {'contract': 'Captured prompt top-eight IDs, expert-major assignment counts. Lane slots are logical capacity, not issued arithmetic, DRAM bytes, or kernel duration.',
              'source_sha256': sha(Path(__file__)), 'capture_receipt_sha256': sha(DATA / 'receipt.json'),
              'model_sha256': capture['model_sha256'],
              'native_source_revision': '1a07bfa5f4144274c8f1c9963821dd9d9a51854b',
              'native_source_sha256': {name: sha(NATIVE / name) for name in NATIVE_FILES},
              'splits': {}}
    for split in ('train', 'held'):
        token_path = DATA / f'{split}.tokens'
        n = len(token_path.read_bytes()) // 4
        hashes = {'tokens': sha(token_path), 'layers': []}
        layers = []
        for layer in range(40):
            path = DATA / f'{split}.layer-{layer}.i32'
            digest = sha(path)
            assert digest == capture['capture_sha256'][split]['layers'][layer]
            hashes['layers'].append(digest)
            layers.append(route_rows(path, n))
        assert hashes['tokens'] == capture['capture_sha256'][split]['tokens']
        result['splits'][split] = {'tokens': n, 'capture_sha256': hashes,
                                    'widths': [summarize(layers, w) for w in WIDTHS]}
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
