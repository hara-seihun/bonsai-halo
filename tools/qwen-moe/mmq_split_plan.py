#!/usr/bin/env python3
"""Price the grid that a J16/J64 split actually launches on Qwen routes.

The native non-stream-K MMQ grid is (ceil(R/I), ceil(T/J), 256), and
inactive experts/columns return inside the kernel. Count entries, not cycles.
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from expert_occupancy import route_rows
from mmq_hybrid_bound import DEFAULT_OCCUPANCY
from mmq_two_width import choose_group
from mmq_work import CONFIG


def account(rows, width, selected):
    launches = []
    for first in range(0, len(rows), width):
        tile = rows[first:first + width]
        counts = Counter(e for row in tile for e in row)
        assert len(tile) * 8 == sum(counts.values())
        buckets = {j: [] for j in selected}
        for expert, size in counts.items():
            choice = choose_group(size, selected, 64, True)
            assert choice is not None
            buckets[choice[-1]].append((expert, size))
        for j, groups in buckets.items():
            if not groups:
                continue
            i, _ = CONFIG[j]
            row_tiles = 1024 // i
            grid_columns = (len(tile) + j - 1) // j
            grid_expert_columns = 256 * grid_columns
            group_columns = sum((n + j - 1) // j for _, n in groups)
            # A kernel body may filter by size before loading the weight image;
            # the existing single body cannot distinguish which J owns a group.
            # This counts grid entries that reach a group versus entries which
            # only execute an early return.
            launches.append({
                'tokens': len(tile), 'J': j, 'groups': len(groups),
                'grid_entries': row_tiles * grid_expert_columns,
                'active_entries': row_tiles * group_columns,
                'empty_expert_entries': row_tiles * (256 - len(counts)) * grid_columns,
                'wrong_body_entries': row_tiles * (len(counts) - len(groups)) * grid_columns,
                'column_tail_entries': row_tiles * sum(grid_columns - (n + j - 1) // j for _, n in groups),
                'compact_group_entries': row_tiles * group_columns,
                'group_tasks': group_columns,
            })
    return launches


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--occupancy', type=Path, default=DEFAULT_OCCUPANCY)
    p.add_argument('--capture-dir', type=Path, default=DEFAULT_OCCUPANCY.parent)
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    raw = a.occupancy.read_bytes()
    occupancy = json.loads(raw)
    source = Path(__file__).read_bytes()
    result = {
        'contract': 'Non-stream-K native grid entries. Assigned groups use the same packed sorted activations and ids_dst; width filtering occurs before operand reads. A compact group/column worklist is a cost comparator requiring construction and another dispatch, not an installed kernel. Entries are not cycles.',
        'source_sha256': hashlib.sha256(source).hexdigest(),
        'occupancy_sha256': hashlib.sha256(raw).hexdigest(),
        'capture_receipt_sha256': occupancy['capture_receipt_sha256'],
        'model_sha256': occupancy['model_sha256'],
        'native_source_revision': occupancy['native_source_revision'],
        'native_source_sha256': occupancy['native_source_sha256'],
        'splits': {},
    }
    for split, record in occupancy['splits'].items():
        rows = []
        for layer, digest in enumerate(record['capture_sha256']['layers']):
            path = a.capture_dir / f'{split}.layer-{layer}.i32'
            data = path.read_bytes()
            assert hashlib.sha256(data).hexdigest() == digest
            rows.append(route_rows(path, record['tokens']))
        per_width = {}
        for width in (record['tokens'],):
            arms = {}
            for label, widths in (('installed_J64', (64,)), ('split_J16_J64', (16, 64))):
                launches = [launch for layer in rows for launch in account(layer, width, widths)]
                totals = {k: sum(launch[k] for launch in launches)
                          for k in ('grid_entries', 'active_entries', 'empty_expert_entries',
                                    'wrong_body_entries', 'column_tail_entries', 'compact_group_entries',
                                    'group_tasks')}
                assert totals['grid_entries'] == sum(totals[k] for k in
                    ('active_entries', 'empty_expert_entries', 'wrong_body_entries', 'column_tail_entries'))
                arms[label] = {'launches': len(launches), **totals,
                               'entries_per_launch_max': max(x['grid_entries'] for x in launches),
                               'active_fraction': totals['active_entries'] / totals['grid_entries']}
            per_width[str(width)] = arms
        result['splits'][split] = per_width
    text = json.dumps(result, indent=2) + '\n'
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(text)
    else:
        print(text)


if __name__ == '__main__':
    main()
