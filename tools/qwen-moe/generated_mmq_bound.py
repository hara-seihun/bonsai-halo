#!/usr/bin/env python3
"""Exact static native-MMQ width frontier on observed 32-stream generated routes.

The callback changes graph fusion. This proves counts for its captured routes,
not native timing, DRAM traffic, or unchanged unobserved route IDs.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import struct

from mmq_work import CONFIG, ROWS, account


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=Path('../../data/qwen-moe/generated-routes/32-stream'))
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    data = args.data
    plain = (data / 'plain.tokens').read_bytes()
    observed = (data / 'observed.tokens').read_bytes()
    assert len(plain) == 3 * 32 * 4 and observed == plain
    capture_hashes = {}
    hist = Counter()
    snapshots = []
    for step in range(2):
        step_hist = Counter()
        layer_stats = []
        for layer in range(40):
            path = data / f'observed.layer-{layer}.i32'
            raw = path.read_bytes()
            assert len(raw) == 2 * 32 * 8 * 4
            capture_hashes[str(layer)] = sha(path)
            all_ids = struct.unpack('<512i', raw)
            ids = all_ids[step * 256:(step + 1) * 256]
            assert all(0 <= expert < 256 for expert in ids)
            assert all(len(set(ids[row * 8:(row + 1) * 8])) == 8 for row in range(32))
            groups = Counter(ids)
            sizes = Counter(groups.values())
            assert sum(k * n for k, n in sizes.items()) == 256
            step_hist.update(sizes)
            layer_stats.append({'layer': layer, 'groups': len(groups),
                                'max_group': max(groups.values()),
                                'groups_over_16': sum(n for size, n in sizes.items() if size > 16)})
        hist.update(step_hist)
        snapshots.append({'step': step + 1, 'groups': sum(step_hist.values()),
                          'histogram': dict(sorted(step_hist.items())), 'layers': layer_stats})
    assert sum(k * n for k, n in hist.items()) == 2 * 40 * 32 * 8
    # Native 16/32 share I=64 and 128 threads. Each group with 17..32
    # assignments costs two J16 passes or one J32 pass; both issue 32 columns.
    assert all(size <= 32 for size in hist)
    panels = {}
    for projection, rows in ROWS.items():
        j16 = account(hist, 16, rows)
        j32 = account(hist, 32, rows)
        hybrid = dict(j16)
        hybrid['packed_image_passes'] = sum(hist.values())
        hybrid['column_tiles'] = hybrid['packed_image_passes']
        hybrid['launched_row_column_tiles'] = hybrid['column_tiles'] * (rows // 64)
        hybrid['input_staging_columns_per_row_tile'] = j16['input_staging_columns_per_row_tile']
        hybrid['full_tile_output_slots'] = j16['full_tile_output_slots']
        panels[projection] = {'selected_J16': j16, 'original_J32': j32,
                              'free_mixed_J16_J32': hybrid}
        assert hybrid['packed_image_passes'] <= j16['packed_image_passes']
        assert hybrid['full_tile_output_slots'] == j16['full_tile_output_slots']
        assert j32['packed_image_passes'] == hybrid['packed_image_passes']
        assert j16['full_tile_output_slots'] == sum(((k + 15) // 16) * 16 * n * rows
                                                      for k, n in hist.items())
    inventory_path = Path('../../data/qwen-moe/traffic.json')
    inventory = json.loads(inventory_path.read_text())
    acquisition_path = inventory_path.with_name('acquisition.json')
    acquisition = json.loads(acquisition_path.read_text())
    assert acquisition['verified'] and acquisition['size'] == Path(acquisition['path']).stat().st_size
    per_expert_bytes = [inventory['layers'][str(layer)]['bank_bytes'] // 256 for layer in range(40)]
    extra = sum(stat['groups_over_16'] * per_expert_bytes[stat['layer']]
                for snapshot in snapshots for stat in snapshot['layers'])
    grouped_expert_bytes = sum(stat['groups'] * per_expert_bytes[stat['layer']]
                               for snapshot in snapshots for stat in snapshot['layers'])
    grouped_complete_bytes = (2 * inventory['nonexpert_nonembedding_bytes'] +
                              2 * 32 * inventory['embedding_lookup_bytes'] +
                              grouped_expert_bytes + extra)
    result = {'contract': 'Two callback-observed generated 32-stream steps after eight-token distinct-prefix setup. Every row computes a full vocabulary head and feeds back greedy decisions. Plain and observed greedy tokens identical. Width grammar: grouped expert, one native J in {16,32} per group, unchanged K/output tiling and column order, free dispatch. Issued I*J positions and complete-image passes; not actual DRAM, arithmetic FP32 identity of a mixed kernel, or inference speed. Native J16/J32 both I64, 128 threads. Callback cuts graph fusion.',
              'source_sha256': sha(__file__), 'capture_source_sha256': sha(Path(__file__).with_name('generated_mmq_capture.cpp')),
              'probe_sha256': sha(data.parent / 'probe32'),
              'plain_tokens_sha256': sha(data / 'plain.tokens'),
              'observed_tokens_sha256': sha(data / 'observed.tokens'),
              'capture_sha256': capture_hashes,
              'inventory_sha256': sha(inventory_path),
              'acquisition_sha256': sha(acquisition_path),
              'model_sha256': acquisition['sha256'],
              'installed_hip_sha256': sha('../../data/qwen-moe/runtime/current/bin/libggml-hip.so.0.21.0'),
              'installed_revision': Path('../../data/qwen-moe/runtime/current').resolve().name,
              'failed_first_capture_wrapper_sha256': sha(data.parent / '32-wrapper.log'),
              'successful_wrapper_sha256': sha(data.parent / '32-observed-wrapper.log'),
              'group_size_histogram': dict(sorted(hist.items())),
              'steps': snapshots, 'panels': panels,
              'free_image_pass_bytes_saved_by_mixing': extra,
              'modeled_selected_J16_grouped_one_read_bytes': grouped_complete_bytes,
              'free_mixed_extra_saving_fraction': extra / grouped_complete_bytes} 
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'groups': sum(hist.values()), 'groups_over_16': sum(n for k, n in hist.items() if k > 16),
                      'max_size': max(hist), 'panels': panels,
                      'extra_bytes': extra, 'grouped_complete_bytes': grouped_complete_bytes,
                      'extra_fraction': extra / grouped_complete_bytes}, indent=2))


if __name__ == '__main__':
    main()
