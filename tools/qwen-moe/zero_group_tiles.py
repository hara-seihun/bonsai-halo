#!/usr/bin/env python3
"""Price zero down-K blocks at the expert MMQ tile observation boundary."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from static_zero_permutation import CAPTURE, DOWN_BYTES_PER_LAYER_TOKEN, LAYERS, ONE_READ_MODEL_BYTES, layout, load


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def count_tile(rows, perm):
    return int(np.all(rows[:, perm].reshape(len(rows), 16, 32) == 0, axis=(0, 2)).sum())


def summarize(ids, hidden, perms, width):
    flat_ids = ids.reshape(-1)
    flat_hidden = hidden.reshape(-1, 512)
    counts = {'tiles': 0, 'slots': len(flat_ids), 'original': 0,
              'train_layout': 0, 'group_hindsight': 0, 'slot_hindsight': 0,
              'active_tiles': 0, 'unseen_tiles': 0, 'size_histogram': {},
              'per_expert': []}
    for expert in range(256):
        rows = flat_hidden[flat_ids == expert]
        if not len(rows):
            continue
        record = {'expert': expert, 'rows': len(rows), 'tiles': 0,
                  'original': 0, 'train_layout': 0, 'group_hindsight': 0,
                  'slot_hindsight': 0}
        for start in range(0, len(rows), width):
            tile = rows[start:start + width]
            record['tiles'] += 1
            record['original'] += count_tile(tile, np.arange(512))
            record['train_layout'] += count_tile(tile, perms[expert])
            # Every all-zero block must draw 32 coordinates in the intersection
            # of the rows' original zero sets. Hindsight can pack that intersection.
            record['group_hindsight'] += int(np.count_nonzero(np.all(tile == 0, axis=0)) // 32)
            # An impossible stronger bound: each slot may skip its own blocks,
            # even when sharing a tile of weights with the other rows.
            record['slot_hindsight'] += int(np.min(np.count_nonzero(tile == 0, axis=1) // 32))
            size = str(len(tile))
            counts['size_histogram'][size] = counts['size_histogram'].get(size, 0) + 1
        for key in ('tiles', 'original', 'train_layout', 'group_hindsight', 'slot_hindsight'):
            counts[key] += record[key]
        counts['per_expert'].append(record)
    if not (counts['original'] <= counts['group_hindsight'] and
            counts['train_layout'] <= counts['group_hindsight'] <= counts['slot_hindsight']):
        raise AssertionError('zero block bound violated')
    for key in ('original', 'train_layout', 'group_hindsight', 'slot_hindsight'):
        counts[key + '_tile_block_fraction'] = counts[key] / (16 * counts['tiles'])
        # Generous: treat tile-level zero-block fraction as removable even from a
        # one-read image for every token, across every layer; no detection cost.
        counts[key + '_conditional_complete_fraction'] = (
            counts[key + '_tile_block_fraction'] * LAYERS *
            DOWN_BYTES_PER_LAYER_TOKEN / ONE_READ_MODEL_BYTES)
    return counts


def mask_stability(ids, hidden):
    ids = ids.reshape(-1)
    masks = hidden.reshape(-1, 512) == 0
    first = {}
    varying = []
    for expert in np.unique(ids):
        selections = masks[ids == expert]
        first[int(expert)] = selections[0]
        if np.any(selections != selections[0]):
            varying.append(int(expert))
    return first, varying


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture', type=Path, default=CAPTURE)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    train_id, train_hidden, tr_hash = load(args.capture, 'train')
    held_id, held_hidden, he_hash = load(args.capture, 'held')
    train_masks, train_varying = mask_stability(train_id, train_hidden)
    held_masks, held_varying = mask_stability(held_id, held_hidden)
    shared = set(train_masks) & set(held_masks)
    conflicting = sorted(e for e in shared if not np.array_equal(train_masks[e], held_masks[e]))
    tr_id = train_id.reshape(-1)
    tr_hidden = train_hidden.reshape(-1, 512)
    perms = [layout(tr_hidden[tr_id == e]) for e in range(256)]
    result = {'source_sha256': sha(__file__),
              'layout_source_sha256': sha(Path(__file__).with_name('static_zero_permutation.py')),
              'capture_receipt_sha256': sha(args.capture / 'receipt.json'),
              'input_sha256': {'train': tr_hash, 'held': he_hash},
              'grammar': 'expert-major stable capture row order; per expert contiguous J-row tiles, 32-coordinate K blocks. Static permutation trained solely on train masks; group hindsight uses intersection of original zeros, repacked freely per tile; slot hindsight also grants independent per-slot layout.',
              'model': {'down_bytes_per_layer_token': DOWN_BYTES_PER_LAYER_TOKEN,
                        'layers': LAYERS, 'complete_one_read_bytes_per_token': ONE_READ_MODEL_BYTES},
              'mask_stability': {'train_experts': len(train_masks), 'held_experts': len(held_masks),
                                 'shared_experts': len(shared), 'train_varying': train_varying,
                                 'held_varying': held_varying, 'cross_split_conflicting': conflicting,
                                 'held_slots_on_unseen_experts': int(sum(np.count_nonzero(held_id == e)
                                                                          for e in held_masks if e not in train_masks))},
              'panels': {split: {str(w): summarize(ids, hidden, perms, w) for w in (1, 16, 32, 64, 128)}
                         for split, ids, hidden in (('train', train_id, train_hidden),
                                                    ('held', held_id, held_hidden))}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    for split, widths in result['panels'].items():
        for width, row in widths.items():
            print(split, width, row['tiles'], *(f'{key}={row[key]}' for key in
                  ('original', 'train_layout', 'group_hindsight', 'slot_hindsight')))


if __name__ == '__main__':
    main()
