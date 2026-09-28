#!/usr/bin/env python3
"""Screen common down-output coordinates on actual Qwen routed producer captures.

The projection is deliberately an oracle: coefficients are read from captured
outputs, not computed from a new packed expert image. This is a necessary local
quality test for any fixed linear output basis, never a model-quality result.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

WIDTH = 2048
SLOTS = 8
ROWS = 64
RANKS = (16, 32, 64, 128, 256, 384, 512)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def capture(root, split, layer, hashes):
    paths = [root / f'{split}.layer-{layer}.ffn_moe_{name}.f32' for name in ('weights_norm', 'down')]
    for path in paths:
        hashes[path.name] = sha(path)
    scores = np.fromfile(paths[0], dtype='<f4').reshape(ROWS, SLOTS).astype(np.float64)
    outputs = np.fromfile(paths[1], dtype='<f4').reshape(ROWS, SLOTS, WIDTH).astype(np.float64)
    assert np.isfinite(scores).all() and np.isfinite(outputs).all()
    assert np.all(scores >= 0)
    return (scores[:, :, None] * outputs).reshape(ROWS * SLOTS, WIDTH)


def basis(data):
    gram = data @ data.T
    eig, vec = np.linalg.eigh(gram)
    order = np.argsort(eig)[::-1]
    eig, vec = eig[order], vec[:, order]
    assert eig[-1] > eig[0] * 1e-12
    u = (data.T @ vec) / np.sqrt(eig)[None, :]
    assert np.max(np.abs(u.T @ u - np.eye(data.shape[0]))) < 1e-7
    return u


def errors(data, u):
    summed = data.reshape(ROWS, SLOTS, WIDTH).sum(axis=1)
    denom = np.square(summed).sum()
    assert denom > 0
    # Project the eight outputs into the SAME basis, then sum. This equals
    # projecting the weighted sum, without materializing individual dots.
    coeff = summed @ u
    cumul = np.cumsum(np.square(coeff), axis=1)
    total = np.square(summed).sum()
    return {str(rank): float(np.sqrt(max(0., (total - cumul[:, rank - 1].sum()) / denom)))
            for rank in RANKS if rank <= u.shape[1]}


def analyze(root, layers):
    hashes = {}
    result = []
    for layer in layers:
        train = capture(root, 'train', layer, hashes)
        held = capture(root, 'held', layer, hashes)
        train_basis = basis(train)
        held_basis = basis(held)
        sum_basis = basis(train.reshape(ROWS, SLOTS, WIDTH).sum(axis=1))
        train_error = errors(train, train_basis)
        held_error = errors(held, train_basis)
        held_oracle = errors(held, held_basis)
        id_files = [root / f'{split}.layer-{layer}.ffn_moe_topk.i32' for split in ('train', 'held')]
        for path in id_files:
            hashes[path.name] = sha(path)
        train_ids, held_ids = (np.fromfile(path, dtype='<i4').reshape(ROWS * SLOTS) for path in id_files)
        assert np.all((train_ids >= 0) & (train_ids < 256)) and np.all((held_ids >= 0) & (held_ids < 256))
        covered = np.isin(held_ids, train_ids)
        slot_residual = held - (held @ train_basis[:, :256]) @ train_basis[:, :256].T
        slot_error = {part: float(np.sqrt(np.square(slot_residual[mask]).sum() / np.square(held[mask]).sum()))
                      for part, mask in (('seen', covered), ('unseen', ~covered)) if mask.any()}
        result.append({'layer': layer, 'train': train_error, 'held': held_error,
                       'held_slot_svd_oracle': held_oracle,
                       'held_train_sum_basis': errors(held, sum_basis),
                       'held_slots_seen_train': int(covered.sum()),
                       'held_slot_r256_error_by_expert_coverage': slot_error,
                       'held_routed_sum_norm2': float(np.square(held.reshape(ROWS, SLOTS, WIDTH).sum(axis=1)).sum())})
        print(f'layer {layer}: r64 train {train_error["64"]:.4f}, held {held_error["64"]:.4f}; '
              f'r256 held {held_error["256"]:.4f}, oracle {held_oracle["256"]:.4f}', flush=True)
    return result, hashes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('capture', type=Path)
    ap.add_argument('--first', type=int, required=True)
    ap.add_argument('--count', type=int, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    assert 0 <= args.first < 40 and 0 < args.count <= 40 - args.first
    layers, hashes = analyze(args.capture, range(args.first, args.first + args.count))
    args.output.write_text(json.dumps({'contract': 'FP64 local weighted sum; per-layer unquantized orthogonal projection onto train-fitted 512-slot basis; free output-derived coefficients; not logits or native arithmetic',
                                        'source_sha256': sha(Path(__file__)), 'files_sha256': hashes,
                                        'rank': RANKS, 'layers': layers}, indent=2) + '\n')


if __name__ == '__main__':
    main()
