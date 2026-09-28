#!/usr/bin/env python3
"""Construct a compact, GPU-indirect J16/J64 routed MMQ tile queue.

CPU witness for a proposed native dispatcher, not a native implementation or timing.
The down consumer reuses each descriptor twice, with down row tile 2*r + parity.
"""
import argparse
import hashlib
import json
import struct
from collections import Counter
from pathlib import Path

from expert_occupancy import route_rows
from mmq_two_width import choose_group

DEFAULT = Path('../../data/qwen-moe/all-layer-routes')


def encode(expert, column, row, j):
    assert 0 <= expert < 256 and 0 <= column < 16 and 0 <= row < 16
    assert j in (16, 64)
    return expert | (column << 8) | (row << 12) | ((j == 64) << 16)


def decode(word):
    assert word >> 17 == 0
    return word & 255, (word >> 8) & 15, (word >> 12) & 15, 64 if word & (1 << 16) else 16


def construct(rows):
    assignments = [[] for _ in range(256)]
    for token, selected in enumerate(rows):
        for slot, expert in enumerate(selected):
            assignments[expert].append((token, slot))
    # The actual native sorted ids_dst and expert_bounds supply this same stable
    # per-expert order; task generation never sorts or re-quantizes activations.
    counts = {16: [0] * 256, 64: [0] * 256}
    groups = {}
    for expert, ids in enumerate(assignments):
        if not ids:
            continue
        selection = choose_group(len(ids), (16, 64), 64, True)
        assert selection is not None
        j = selection[3]
        i = 64 if j == 16 else 128
        ncol = (len(ids) + j - 1) // j
        counts[j][expert] = ncol * (1024 // i)
        assert counts[j][expert] <= 16
        groups[expert] = (j, ncol, i)
    prefixes = {j: [0] for j in counts}
    words = {j: [] for j in counts}
    for expert in range(256):
        if expert in groups:
            j, ncol, i = groups[expert]
            for column in range(ncol):
                for row in range(1024 // i):
                    words[j].append(encode(expert, column, row, j))
        for j in counts:
            prefixes[j].append(prefixes[j][-1] + counts[j][expert])
    for j in counts:
        assert len(words[j]) == prefixes[j][-1]
        for expert in range(256):
            segment = words[j][prefixes[j][expert]:prefixes[j][expert + 1]]
            assert len(segment) == counts[j][expert]
            for word in segment:
                e, column, row, body = decode(word)
                assert e == expert and body == j
                assert column * j < len(assignments[e])
                assert row * (64 if j == 16 else 128) < 1024
        assert len(set(words[j])) == len(words[j])
    assert set(words[16]).isdisjoint(words[64])
    return counts, prefixes, words, groups


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture-dir', type=Path, default=DEFAULT)
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    occupancy_raw = (a.capture_dir / 'occupancy.json').read_bytes()
    occupancy = json.loads(occupancy_raw)
    result = {
        'contract': 'Native sorted per-expert ids/bounds; one J16/J64 body per nonempty expert, no more image passes than J64. One 256-lane scan/emit CTA per layer-tile, then two indirect persistent body launches per gate/up and down. Down reuses descriptor with two row halves. Counts and descriptor bytes only; no device time.',
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'occupancy_sha256': hashlib.sha256(occupancy_raw).hexdigest(),
        'capture_receipt_sha256': occupancy['capture_receipt_sha256'],
        'model_sha256': occupancy['model_sha256'],
        'native_source_revision': occupancy['native_source_revision'],
        'native_source_sha256': occupancy['native_source_sha256'],
        'format': {'descriptor_bytes': 4, 'expert_bits': 8, 'column_bits': 4,
                   'gate_row_bits': 4, 'body_bits': 1,
                   'down_row': '2 * gate_row + parity, parity in {0,1}'},
        'splits': {},
    }
    for split, record in occupancy['splits'].items():
        width = 128
        layers = []
        total = Counter()
        for layer, digest in enumerate(record['capture_sha256']['layers']):
            path = a.capture_dir / f'{split}.layer-{layer}.i32'
            assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
            rows = route_rows(path, record['tokens'])
            for first in range(0, len(rows), width):
                tile = rows[first:first + width]
                counts, prefixes, words, groups = construct(tile)
                for j in (16, 64):
                    total[f'{j}_tasks'] += len(words[j])
                    total[f'{j}_groups'] += sum(c > 0 for c in counts[j])
                total['tiles'] += 1
                total['assignments'] += len(tile) * 8
                total['groups'] += len(groups)
                binary = b''.join(struct.pack('<I', v) for j in (16, 64) for v in words[j])
                layers.append({'layer': layer, 'first': first, 'tokens': len(tile),
                               'groups': len(groups), 'counts': {str(j): prefixes[j][-1] for j in (16, 64)},
                               'descriptor_bytes': len(binary),
                               'descriptor_sha256': hashlib.sha256(binary).hexdigest(),
                               'max_descriptors_per_expert': max(max(counts[j]) for j in (16, 64))})
        gate = total['16_tasks'] + total['64_tasks']
        assert total['groups'] <= total['assignments']
        result['splits'][split] = {
            'prompt_tiles': total['tiles'], 'assignments': total['assignments'],
            'expert_groups': total['groups'],
            'body_groups': {str(j): total[f'{j}_groups'] for j in (16, 64)},
            'gate_up_tasks': {str(j): total[f'{j}_tasks'] for j in (16, 64)},
            'down_tasks': 2 * gate, 'gate_up_task_total': gate,
            'descriptor_bytes': 4 * gate,
            'max_descriptors_per_expert': max(x['max_descriptors_per_expert'] for x in layers),
            'max_descriptors_per_layer': max(x['descriptor_bytes'] // 4 for x in layers),
            'max_fixed_capacity_descriptors_per_layer': 256 * 16,
            'full_grid_gate_up_entries': total['tiles'] * 256 * (8 * 16 + 2 * 8),
            'layers': layers,
        }
    text = json.dumps(result, indent=2) + '\n'
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(text)
    else:
        print(text)


if __name__ == '__main__':
    main()
