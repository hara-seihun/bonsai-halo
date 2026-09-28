#!/usr/bin/env python3
"""Exact mixed-width MMQ tile bound for captured Qwen routes, not a timing model.

Every expert group selects one native J body for all its K/output tiles. The
free-dispatch grammar may choose J independently per group; it never changes
assignment order, packed weights, or the FP32 dot/reduction order. Compare
issued full I x J positions and logical packed-image passes, not DRAM bytes.
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from mmq_work import CONFIG, ROWS, account
from expert_occupancy import route_rows

DEFAULT_OCCUPANCY = Path('../../data/qwen-moe/all-layer-routes/occupancy.json')
DEFAULT_CAPTURE = DEFAULT_OCCUPANCY.parent
WIDTHS = tuple(sorted(CONFIG))


def cost(size, j, rows):
    i, _ = CONFIG[j]
    passes = (size + j - 1) // j
    return passes * ((rows + i - 1) // i) * i * j, passes


def choose(size, rows, baseline):
    limit = cost(size, baseline, rows)[1]
    # A group cannot change its reduction or order; only select one native body.
    # The primary objective is issued output slots; among ties, fewer image passes.
    return min((*cost(size, j, rows), j)
               for j in WIDTHS if cost(size, j, rows)[1] <= limit)


def analyze(hist, rows, baseline):
    uniform = account(hist, baseline, rows)
    slots = passes = tiles = staging = 0
    assignment_by_j = Counter()
    group_by_j = Counter()
    rows_per_tile = Counter()
    for size, count in hist.items():
        position, n_passes, j = choose(size, rows, baseline)
        i, _ = CONFIG[j]
        slots += position * count
        passes += n_passes * count
        tiles += n_passes * ((rows + i - 1) // i) * count
        staging += n_passes * ((rows + i - 1) // i) * j * count
        assignment_by_j[j] += size * count
        group_by_j[j] += count
        rows_per_tile[j] += n_passes * count
    assert max(hist) <= 128
    floor_slots = sum(((size + 15) // 16) * 16 * count * rows
                      for size, count in hist.items())
    assert slots == floor_slots
    assert passes == sum(hist.values())
    assert passes <= uniform['packed_image_passes']
    assert slots <= uniform['full_tile_output_slots']
    return {
        'selected_uniform_J': baseline,
        'uniform_image_passes': uniform['packed_image_passes'],
        'uniform_full_tile_output_slots': uniform['full_tile_output_slots'],
        'mixed_image_passes': passes,
        'proven_minimum_image_passes': sum(hist.values()),
        'mixed_full_tile_output_slots': slots,
        'proven_minimum_output_slots': floor_slots,
        'mixed_row_column_tiles': tiles,
        'mixed_staging_columns': staging,
        'mixed_groups_by_J': {str(k): group_by_j[k] for k in WIDTHS if group_by_j[k]},
        'mixed_assignments_by_J': {str(k): assignment_by_j[k] for k in WIDTHS if group_by_j[k]},
        'mixed_column_tiles_by_J': {str(k): rows_per_tile[k] for k in WIDTHS if group_by_j[k]},
        'slot_fraction': slots / uniform['full_tile_output_slots'],
        'pass_fraction': passes / uniform['packed_image_passes'],
    }


def launch_account(rows_by_layer, width, baseline):
    launch_hist = Counter()
    aggregate_sizes = Counter()
    for rows in rows_by_layer:
        for first in range(0, len(rows), width):
            groups = Counter(e for row in rows[first:first + width] for e in row)
            aggregate_sizes.update(groups.values())
            bodies = {choose(size, ROWS['gate_up'], baseline)[2] for size in groups.values()}
            launch_hist[len(bodies)] += 1
    return {
        'uniform_kernel_launches': sum(launch_hist.values()),
        'mixed_kernel_launches': sum(k * v for k, v in launch_hist.items()),
        'widths_per_layer_tile_histogram': dict(sorted(launch_hist.items())),
    }, aggregate_sizes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--occupancy', type=Path, default=DEFAULT_OCCUPANCY)
    parser.add_argument('--capture-dir', type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    raw = args.occupancy.read_bytes()
    occupancy = json.loads(raw)
    source = Path(__file__).read_bytes()
    result = {
        'contract': 'For groups of size <=128, one native J per expert group from 16..128 in steps of 16. Selecting J=16*ceil(size/16) attains both global minimum full-tile output positions and minimum one image pass per group. Free mixed dispatch, sorting, launches and boundaries; static issued positions, not time or DRAM.',
        'source_sha256': hashlib.sha256(source).hexdigest(),
        'occupancy_sha256': hashlib.sha256(raw).hexdigest(),
        'model_sha256': occupancy['model_sha256'],
        'capture_receipt_sha256': occupancy['capture_receipt_sha256'],
        'native_source_revision': occupancy['native_source_revision'],
        'native_source_sha256': occupancy['native_source_sha256'],
        'splits': {},
    }
    for split, record in occupancy['splits'].items():
        result['splits'][split] = {}
        rows_by_layer = []
        for layer, digest in enumerate(record['capture_sha256']['layers']):
            path = args.capture_dir / f'{split}.layer-{layer}.i32'
            assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
            rows_by_layer.append(route_rows(path, record['tokens']))
        for tile in record['widths']:
            if tile['width'] <= 8:
                continue  # MMVQ, not this grammar.
            hist = Counter({int(size): int(count) for size, count in tile['group_size_histogram'].items()})
            baseline = 64 if 65 <= tile['width'] <= 256 else tile['native_global_J']
            panel = {name: analyze(hist, rows, baseline) for name, rows in ROWS.items()}
            launches, observed_hist = launch_account(rows_by_layer, tile['width'], baseline)
            assert hist == observed_hist
            # Equal row alignment at Qwen's 1024/2048 output widths.
            assert panel['gate_up']['slot_fraction'] == panel['down']['slot_fraction']
            result['splits'][split][str(tile['width'])] = {
                'tokens': record['tokens'], 'groups': sum(hist.values()),
                'assignments': sum(k * v for k, v in hist.items()),
                'launches_if_separate_bodies': launches,
                'arms': panel,
            }
    text = json.dumps(result, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    else:
        print(text)


if __name__ == '__main__':
    main()
