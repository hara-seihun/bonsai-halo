#!/usr/bin/env python3
"""Replay builder-free per-expert MMQ dispatch against the compact tile queue.

One CTA is launched for each expert in each J body. It reads the existing
expert_bounds, selects its body, then loops over the column and output-row
tiles. The down projection repeats each row tile twice. This is a scheduling
construction, not a GPU timing or a replacement for the native MMQ kernel.
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from expert_occupancy import route_rows
from mmq_indirect import construct, encode
from mmq_two_width import choose_group

DEFAULT = Path('../../data/qwen-moe/all-layer-routes')


def digest(words):
    import struct
    return hashlib.sha256(b''.join(struct.pack('<I', x) for x in words)).hexdigest()


def replay(rows):
    counts, prefixes, queued, _ = construct(rows)
    assignments = Counter(expert for selected in rows for expert in selected)
    per_expert = {16: [0] * 256, 64: [0] * 256}
    direct = {16: [], 64: []}
    for expert in range(256):
        n = assignments[expert]
        if not n:
            continue
        selection = choose_group(n, (16, 64), 64, True)
        assert selection is not None
        j = selection[3]
        i = 64 if j == 16 else 128
        columns = (n + j - 1) // j
        # This is the program's loop nest, with bounds supplied by the native
        # expert_bounds. Neither a descriptor nor a global task claim is read.
        for column in range(columns):
            for row in range(1024 // i):
                direct[j].append(encode(expert, column, row, j))
                per_expert[j][expert] += 1
    assert all(direct[j] == queued[j] for j in (16, 64))
    assert all(per_expert[j][expert] == prefixes[j][expert + 1] - prefixes[j][expert]
               for j in (16, 64) for expert in range(256))
    return per_expert, {str(j): digest(direct[j]) for j in (16, 64)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-dir', type=Path, default=DEFAULT)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    raw = (args.capture_dir / 'occupancy.json').read_bytes()
    occupancy = json.loads(raw)
    result = {
        'contract': 'One CTA per expert per J16/J64 body, launched 256 per body; selected CTA loops its existing sorted expert group column and row tiles sequentially. Down loops two halves of each gate/up row tile. Same individual tile body, distinct scheduling. No queue, builder, per-tile atomic or extra sort. CPU tile-list equality only; no GPU time or bit-exact FP32 acceptance.',
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'occupancy_sha256': hashlib.sha256(raw).hexdigest(),
        'model_sha256': occupancy['model_sha256'],
        'native_source_revision': occupancy['native_source_revision'],
        'native_source_sha256': occupancy['native_source_sha256'],
        'capture_receipt_sha256': occupancy['capture_receipt_sha256'],
        'splits': {},
    }
    for split, record in occupancy['splits'].items():
        total = Counter()
        layers = []
        for layer, expected_hash in enumerate(record['capture_sha256']['layers']):
            path = args.capture_dir / f'{split}.layer-{layer}.i32'
            assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_hash
            rows = route_rows(path, record['tokens'])
            for first in range(0, len(rows), 128):
                tile = rows[first:first + 128]
                c, hashes = replay(tile)
                active = {j: sum(bool(n) for n in c[j]) for j in (16, 64)}
                steps = {j: sum(c[j]) for j in (16, 64)}
                depth = {j: max(c[j]) for j in (16, 64)}
                total.update({'gate_up_tasks': sum(steps.values()),
                              'down_tasks': 2 * sum(steps.values()),
                              'launched_gate_up_ctas': 512,
                              'active_gate_up_ctas': sum(active.values()),
                              'empty_or_other_body_ctas': 512 - sum(active.values()),
                              'builder_ctas': 0, 'descriptor_bytes': 0,
                              'queue_claims': 0, 'prompt_tiles': 1})
                for j in (16, 64):
                    total[f'{j}_tasks'] += steps[j]
                    total[f'{j}_active_ctas'] += active[j]
                    total[f'{j}_max_serial_depth'] = max(total[f'{j}_max_serial_depth'], depth[j])
                layers.append({'layer': layer, 'first': first, 'tokens': len(tile),
                               'active_ctas': {str(j): active[j] for j in (16, 64)},
                               'serial_tile_steps': {str(j): steps[j] for j in (16, 64)},
                               'longest_cta': {str(j): depth[j] for j in (16, 64)},
                               'tile_hash': hashes})
        result['splits'][split] = {'totals': dict(total), 'layers': layers}
    text = json.dumps(result, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    else:
        print(text)


if __name__ == '__main__':
    main()
