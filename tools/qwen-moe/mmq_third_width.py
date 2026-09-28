#!/usr/bin/env python3
"""Price a third native MMQ J body against the captured two-body frontier.

This is a static work/launch count, not a native timing or DRAM estimate.
"""
import argparse
import hashlib
import itertools
import json
from collections import Counter
from pathlib import Path

from expert_occupancy import route_rows
from mmq_hybrid_bound import DEFAULT_OCCUPANCY, CONFIG
from mmq_two_width import panel


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def groups_per_tile(record, split, directory, width):
    hists = []
    for layer, expected in enumerate(record['capture_sha256']['layers']):
        path = directory / f'{split}.layer-{layer}.i32'
        assert digest(path) == expected, path
        rows = route_rows(path, record['tokens'])
        for start in range(0, len(rows), width):
            hists.append(Counter(Counter(e for row in rows[start:start + width] for e in row).values()))
    return hists


def evaluate(hists, baseline):
    # Exhaustive over native one-, two- and three-body choices. Each group picks
    # minimum output positions, with image passes and row/column blocks as ties.
    choices = [p for n in (1, 2, 3)
               for js in itertools.combinations(CONFIG, n)
               if (p := panel(hists, js, baseline, True)) is not None]
    def best(count):
        return min((p for p in choices if len(p['widths']) <= count),
                   key=lambda p: (p['output_positions_gate_up'], p['image_passes'],
                                  p['gate_up_launches'], p['row_column_blocks_gate_up']))
    two, three = best(2), best(3)
    dominated_three = not any(
        p['output_positions_gate_up'] < two['output_positions_gate_up'] and
        p['image_passes'] <= two['image_passes'] and
        p['row_column_blocks_gate_up'] <= two['row_column_blocks_gate_up'] and
        p['gate_up_launches'] <= two['gate_up_launches'] for p in choices
        if len(p['widths']) == 3)
    return {'installed': panel(hists, (baseline,), baseline, True),
            'best_two': two, 'best_three': three,
            'three_has_no_strict_dominance_over_two': dominated_three,
            'additional_saved_positions': two['output_positions_gate_up'] - three['output_positions_gate_up'],
            'additional_launches': three['gate_up_launches'] - two['gate_up_launches'],
            'additional_row_column_blocks': three['row_column_blocks_gate_up'] - two['row_column_blocks_gate_up']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--occupancy', type=Path, default=DEFAULT_OCCUPANCY)
    parser.add_argument('--capture-dir', type=Path, default=DEFAULT_OCCUPANCY.parent)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    occupancy = json.loads(args.occupancy.read_bytes())
    result = {'contract': 'One native J per expert group; no more group image passes than installed J. Free partition. Actual Qwen gate/up 1024 output rows; down 2048 doubles positions and row/column blocks. No timing/physical bytes/FP32 identity claim.',
              'source_sha256': digest(Path(__file__)),
              'occupancy_sha256': digest(args.occupancy),
              'model_sha256': occupancy['model_sha256'],
              'native_source_revision': occupancy['native_source_revision'],
              'native_source_sha256': occupancy['native_source_sha256'],
              'capture_receipt_sha256': occupancy['capture_receipt_sha256'],
              'splits': {}}
    for split, record in occupancy['splits'].items():
        result['splits'][split] = {}
        for tile in record['widths']:
            width = tile['width']
            if width <= 8:
                continue
            hists = groups_per_tile(record, split, args.capture_dir, width)
            combined = Counter()
            for hist in hists:
                combined.update(hist)
            assert combined == Counter({int(k): v for k, v in tile['group_size_histogram'].items()})
            baseline = 64 if 65 <= width <= 256 else tile['native_global_J']
            result['splits'][split][str(width)] = evaluate(hists, baseline)
    text = json.dumps(result, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    else:
        print(text)


if __name__ == '__main__':
    main()
