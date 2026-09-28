#!/usr/bin/env python3
"""Bound expert-fixed zero-block layouts on actual routed down producers."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

CAPTURE = Path('../../data/qwen-moe/route-capture')
DOWN_BYTES_PER_LAYER_TOKEN = 5_767_168
LAYERS = 40
ONE_READ_MODEL_BYTES = 2_626_187_904


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(root, split):
    prefix = root / split
    paths = {key: Path(f'{prefix}.0.{name}-0.bin') for key, name in
             [('ids', 'ffn_moe_topk'), ('hidden', 'ffn_moe_swiglu')]}
    ids = np.fromfile(paths['ids'], dtype='<i4').reshape(-1, 8)
    hidden = np.fromfile(paths['hidden'], dtype='<f4').reshape(-1, 8, 512)
    if len(ids) != len(hidden) or not np.isfinite(hidden).all() or np.any((ids < 0) | (ids >= 256)):
        raise ValueError('invalid routed producer capture')
    return ids, hidden, {k: sha(v) for k, v in paths.items()}


def layout(rows):
    if not len(rows):
        return np.arange(512)
    active = rows != 0
    # Lowest train activation frequency first. Within a frequency, place
    # identical activity signatures together; both rules use train only.
    signatures = [tuple(bool(x) for x in active[:, col]) for col in range(512)]
    return np.array(sorted(range(512), key=lambda c: (sum(signatures[c]), signatures[c], c)))


def count(rows, permutation):
    return np.all(rows[:, permutation].reshape(-1, 16, 32) == 0, axis=2).sum(axis=1)


def evaluate(ids, hidden, perms):
    flat_ids, flat_rows = ids.reshape(-1), hidden.reshape(-1, 512)
    identity = count(flat_rows, np.arange(512))
    learned = np.empty(len(flat_ids), dtype=np.int64)
    for expert in np.unique(flat_ids):
        slots = flat_ids == expert
        learned[slots] = count(flat_rows[slots], perms[int(expert)])
    zero = np.count_nonzero(flat_rows == 0, axis=1)
    oracle = zero // 32
    if np.any(learned > oracle):
        raise AssertionError('layout beat unrestricted per-slot oracle')
    def stats(v):
        return {'blocks': int(v.sum()), 'slots_with_skip': int(np.count_nonzero(v)),
                'mean_blocks_per_slot': float(v.mean()), 'max_blocks_per_slot': int(v.max()),
                'per_token_blocks': v.reshape(-1, 8).sum(axis=1).tolist(),
                'conditional_down_image_fraction': float(v.sum() / (len(v) * 16)),
                'optimistic_complete_image_fraction_if_all_layers_match':
                    float(LAYERS * DOWN_BYTES_PER_LAYER_TOKEN / ONE_READ_MODEL_BYTES * v.sum() / (len(v) * 16))}
    return {'slots': len(flat_ids), 'zero_floats': int(zero.sum()),
            'original': stats(identity), 'train_layout': stats(learned),
            'per_slot_hindsight': stats(oracle),
            'expert_slot_counts': np.bincount(flat_ids, minlength=256).tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, default=CAPTURE)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    tr_ids, tr_hidden, tr_hash = load(args.capture, 'train')
    he_ids, he_hidden, he_hash = load(args.capture, 'held')
    tr_flat_ids = tr_ids.reshape(-1)
    tr_flat_hidden = tr_hidden.reshape(-1, 512)
    perms = [layout(tr_flat_hidden[tr_flat_ids == e]) for e in range(256)]
    trained = evaluate(tr_ids, tr_hidden, perms)
    held = evaluate(he_ids, he_hidden, perms)
    seen = np.flatnonzero(np.bincount(tr_flat_ids, minlength=256))
    unseen_held = sum(int(np.count_nonzero(he_ids == e)) for e in range(256) if e not in seen)
    result = {'source_sha256': sha(Path(__file__)),
              'capture_receipt_sha256': sha(args.capture / 'receipt.json'),
              'input_sha256': {'train': tr_hash, 'held': he_hash},
              'grammar': 'static expert-specific 512-column permutation learned from train zero masks; group 32, free zero detection and selective image reads; per-slot oracle rearranges each hidden freely',
              'one_read_bytes': {'down_per_layer_token': DOWN_BYTES_PER_LAYER_TOKEN,
                                 'layers': LAYERS,
                                 'complete_per_token': ONE_READ_MODEL_BYTES},
              'train_experts': len(seen), 'held_slots_unseen_in_train': unseen_held,
              'train': trained, 'held': held}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    for key in ('train', 'held'):
        print(key, 'slots', result[key]['slots'], 'zero', result[key]['zero_floats'])
        for arm in ('original', 'train_layout', 'per_slot_hindsight'):
            s = result[key][arm]
            print(' ', arm, s['blocks'], 'zero blocks', s['slots_with_skip'], 'slots',
                  'whole-image optimistic %', round(100 * s['optimistic_complete_image_fraction_if_all_layers_match'], 4))
    print('held slots without train expert:', unseen_held)


if __name__ == '__main__':
    main()
