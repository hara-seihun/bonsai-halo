#!/usr/bin/env python3
"""Enumerate one/two native MMQ widths on pinned Qwen prompt route captures.

A group uses one J body; its complete K/output tiles stay in that body. This
counts issued full-tile positions and logical image passes, not device time.
"""
import argparse
import hashlib
import itertools
import json
from collections import Counter
from pathlib import Path

from expert_occupancy import route_rows
from mmq_hybrid_bound import DEFAULT_OCCUPANCY, CONFIG, WIDTHS, cost


def choose_group(size, widths, baseline, no_extra_pass):
    choices = []
    for j in widths:
        positions, passes = cost(size, j, 1024)
        if not no_extra_pass or passes <= cost(size, baseline, 1024)[1]:
            i, _ = CONFIG[j]
            choices.append((positions, passes, passes * (1024 // i), j))
    if not choices:
        return None
    return min(choices)


def panel(histograms, widths, baseline, no_extra_pass):
    groups_by_j = Counter()
    positions = passes = blocks = staging = launches = 0
    for hist in histograms:
        used = set()
        for size, count in hist.items():
            choice = choose_group(size, widths, baseline, no_extra_pass)
            if choice is None:
                return None
            p, n, b, j = choice
            groups_by_j[j] += count
            positions += p * count
            passes += n * count
            blocks += b * count
            staging += b * j * count
            used.add(j)
        launches += len(used)
    return {
        'widths': list(widths), 'output_positions_gate_up': positions,
        'image_passes': passes, 'row_column_blocks_gate_up': blocks,
        'staging_columns_gate_up': staging, 'gate_up_launches': launches,
        'groups_by_J': {str(j): groups_by_j[j] for j in widths if groups_by_j[j]},
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--occupancy', type=Path, default=DEFAULT_OCCUPANCY)
    p.add_argument('--capture-dir', type=Path, default=DEFAULT_OCCUPANCY.parent)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    raw = args.occupancy.read_bytes()
    occupancy = json.loads(raw)
    result = {
        'contract': 'One native J per grouped expert. Exhaustive one/two-J sets and exact per-group lexicographic positions/passes/blocks choice. Optional no-extra-image-pass constraint relative to installed J. Free group partition and launch; static tile counts only.',
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'occupancy_sha256': hashlib.sha256(raw).hexdigest(),
        'model_sha256': occupancy['model_sha256'],
        'native_source_revision': occupancy['native_source_revision'],
        'native_source_sha256': occupancy['native_source_sha256'],
        'capture_receipt_sha256': occupancy['capture_receipt_sha256'],
        'splits': {},
    }
    sets = [(j,) for j in WIDTHS] + list(itertools.combinations(WIDTHS, 2))
    for split, record in occupancy['splits'].items():
        result['splits'][split] = {}
        rows_by_layer = []
        for layer, digest in enumerate(record['capture_sha256']['layers']):
            path = args.capture_dir / f'{split}.layer-{layer}.i32'
            assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
            rows_by_layer.append(route_rows(path, record['tokens']))
        for tile in record['widths']:
            width = tile['width']
            if width < 32:
                continue
            hists = []
            combined = Counter()
            for rows in rows_by_layer:
                for first in range(0, len(rows), width):
                    groups = Counter(e for row in rows[first:first + width] for e in row)
                    hist = Counter(groups.values())
                    hists.append(hist)
                    combined.update(hist)
            assert combined == Counter({int(k): v for k, v in tile['group_size_histogram'].items()})
            baseline = 64 if 65 <= width <= 256 else tile['native_global_J']
            installed = panel(hists, (baseline,), baseline, True)
            candidates = [x for js in sets if (x := panel(hists, js, baseline, True)) is not None]
            frontier = [x for x in candidates if not any(
                y is not x and y['output_positions_gate_up'] <= x['output_positions_gate_up']
                and y['image_passes'] <= x['image_passes']
                and y['row_column_blocks_gate_up'] <= x['row_column_blocks_gate_up']
                and y['gate_up_launches'] <= x['gate_up_launches']
                and any(y[k] < x[k] for k in ('output_positions_gate_up', 'image_passes',
                                                'row_column_blocks_gate_up', 'gate_up_launches'))
                for y in candidates)]
            best = min(candidates, key=lambda x: (x['output_positions_gate_up'], x['image_passes'],
                                                  x['gate_up_launches'], x['row_column_blocks_gate_up']))
            unconstrained = min((x for js in sets if (x := panel(hists, js, baseline, False)) is not None),
                                key=lambda x: (x['output_positions_gate_up'], x['image_passes'],
                                               x['gate_up_launches']))
            result['splits'][split][str(width)] = {
                'tokens': record['tokens'], 'groups': sum(combined.values()),
                'installed': installed, 'best_two_width_no_extra_pass': best,
                'best_two_width_unconstrained': unconstrained,
                'candidate_count': len(candidates),
                'nondominated_no_extra_pass': sorted(frontier, key=lambda x: (
                    x['gate_up_launches'], x['output_positions_gate_up'], x['image_passes'])),
                'suggested_16_64': panel(hists, (16, 64), baseline, True),
                'suggested_32_64': panel(hists, (32, 64), baseline, True),
            }
    text = json.dumps(result, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    else:
        print(text)


if __name__ == '__main__':
    main()
