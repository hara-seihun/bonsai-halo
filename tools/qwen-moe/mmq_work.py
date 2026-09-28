#!/usr/bin/env python3
"""Count native MMQ work issued for real grouped Qwen expert routes.

Counts do not stand in for device time or physical DRAM traffic. The native
mul_mat_q_process_tile stages J activation columns and calls vec_dot over the
full I x J tile; only write_back masks inactive output columns.
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

OCCUPANCY = Path('../../data/qwen-moe/all-layer-routes/occupancy.json')
CONFIG = {16: (64, 128), 32: (64, 128), 48: (128, 256),
          64: (128, 256), 80: (128, 256), 96: (128, 256),
          112: (128, 256), 128: (128, 256)}
ROWS = {'gate_up': 1024, 'down': 2048}


def account(hist, j, rows):
    i, threads = CONFIG[j]
    groups = sum(hist.values())
    tiles = sum(((size + j - 1) // j) * count for size, count in hist.items())
    row_tiles = (rows + i - 1) // i
    assignments = sum(size * count for size, count in hist.items())
    return {
        'I': i, 'J': j, 'threads': threads, 'expert_groups': groups,
        'column_tiles': tiles, 'launched_row_column_tiles': tiles * row_tiles,
        'full_tile_output_slots': tiles * row_tiles * i * j,
        'useful_output_elements': assignments * rows,
        'packed_image_passes': tiles,
        'input_staging_columns_per_row_tile': tiles * row_tiles * j,
        'assignment_column_fill': assignments / (tiles * j),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--occupancy', type=Path, default=OCCUPANCY)
    p.add_argument('--output', type=Path)
    p.add_argument('--native-source', type=Path, default=Path('../bonsai-hip/ggml/src/ggml-cuda'))
    a = p.parse_args()
    raw = a.occupancy.read_bytes()
    occupancy = json.loads(raw)
    source_hashes = {}
    for name in ('mmq.cuh', 'mmq-vec-dot.cuh', 'mmq-config-rdna3-5.cuh'):
        source_hashes[name] = hashlib.sha256((a.native_source / name).read_bytes()).hexdigest()
    for name in ('mmq.cuh', 'mmq-config-rdna3-5.cuh'):
        assert source_hashes[name] == occupancy['native_source_sha256'][name], name
    result = {
        'contract': 'Non-stream-K gfx1151 MMQ full-tile issued work and logical image-pass counts, not timing/DRAM. The 8-row MMVQ path is excluded.',
        'occupancy_sha256': hashlib.sha256(raw).hexdigest(),
        'analysis_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'capture_receipt_sha256': occupancy['capture_receipt_sha256'],
        'model_sha256': occupancy['model_sha256'],
        'native_source_revision': occupancy['native_source_revision'],
        'native_source_sha256': source_hashes,
        'configuration': {str(j): {'I': i, 'threads': t} for j, (i, t) in CONFIG.items()},
        'splits': {},
    }
    for split, record in occupancy['splits'].items():
        out = {}
        for tile in record['widths']:
            width = tile['width']
            if width <= 8:
                continue
            hist = Counter({int(k): int(v) for k, v in tile['group_size_histogram'].items()})
            chosen = tile['native_global_J']
            if chosen not in CONFIG:
                raise ValueError(f'unsupported native J={chosen}')
            assert sum(k * v for k, v in hist.items()) == tile['assignments']
            assert sum(hist.values()) == tile['active_expert_tiles']
            panel = {
                'width': width, 'prompt_tokens': record['tokens'],
                'selected_J': chosen, 'assignments': tile['assignments'],
                'expert_groups': tile['active_expert_tiles'],
                'arms': {},
            }
            for name, rows in ROWS.items():
                panel['arms'][name] = {str(j): account(hist, j, rows) for j in CONFIG}
            out[str(width)] = panel
        result['splits'][split] = out
    data = json.dumps(result, indent=2) + '\n'
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(data)
    else:
        print(data)


if __name__ == '__main__':
    main()
