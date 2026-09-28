#!/usr/bin/env python3
"""Actual Qwen routed outputs: best stored same-expert ray, with a free held oracle.

The oracle sees the target output, selects one train vector from that layer/expert,
then fits its unrestricted signed real gain. Unseen experts retain their exact
output. This is a local finite-dictionary diagnostic, not a runnable codec.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def capture(root, split, layer):
    prefix = f'{split}.layer-{layer}.'
    names = [prefix + n for n in ('ffn_moe_topk.i32', 'ffn_moe_weights_norm.f32',
                                   'ffn_moe_down.f32', 'attn_post_norm.f32')]
    paths = [root / n for n in names]
    ids = np.fromfile(paths[0], dtype='<i4').reshape(64, 8)
    scores = np.fromfile(paths[1], dtype='<f4').reshape(64, 8).astype(np.float64)
    outputs = np.fromfile(paths[2], dtype='<f4').reshape(64, 8, 2048).astype(np.float64)
    inputs = np.fromfile(paths[3], dtype='<f4').reshape(64, 2048).astype(np.float64)
    assert np.all((ids >= 0) & (ids < 256)) and np.all(np.diff(np.sort(ids, axis=1), axis=1) > 0)
    assert np.isfinite(scores).all() and np.all(scores >= 0) and np.isfinite(outputs).all() and np.isfinite(inputs).all()
    return ids, scores, outputs, inputs, {p.name: sha(p) for p in paths}


def layer_scan(root, layer):
    ti, _, ty, tx, th = capture(root, 'train', layer)
    hi, hs, hy, hx, hh = capture(root, 'held', layer)
    ref = (hs[:, :, None] * hy).sum(axis=1)
    denom = float(np.square(ref).sum())
    assert denom > 0
    ray = hy.copy()
    input_nn = hy.copy()
    ray_directions = np.zeros_like(hy)
    nn_directions = np.zeros_like(hy)
    seen_mask = np.zeros((64, 8), dtype=bool)
    chosen_gain = []
    seen = 0
    oracle_slot_num = oracle_slot_den = 0.
    nn_slot_num = 0.
    for expert in np.unique(hi):
        tr, ts = np.where(ti == expert)
        hr, hslt = np.where(hi == expert)
        if not len(tr):
            continue
        seen += len(hr)
        seen_mask[hr, hslt] = True
        bank = ty[tr, ts]
        target = hy[hr, hslt]
        norm2 = np.square(bank).sum(axis=1)
        assert np.all(norm2 > 0)
        dot = target @ bank.T
        # Per-target optimal signed scalar and train direction, granted true
        # held output. This minimizes the individual slot's squared error,
        # not necessarily the complete sum's error (cancellation may matter).
        chosen = np.argmax(np.square(dot) / norm2[None, :], axis=1)
        gain = dot[np.arange(len(chosen)), chosen] / norm2[chosen]
        predicted = gain[:, None] * bank[chosen]
        ray[hr, hslt] = predicted
        ray_directions[hr, hslt] = bank[chosen]
        chosen_gain.extend(gain.tolist())
        oracle_slot_num += float(np.square(target - predicted).sum())
        oracle_slot_den += float(np.square(target).sum())
        # Executable selector control: nearest captured producer from this
        # expert, then the same inaccessible target-dependent optimal gain.
        dist = np.square(hx[hr, None, :] - tx[None, tr, :]).sum(axis=2)
        nearest = np.argmin(dist, axis=1)
        nn_gain = dot[np.arange(len(nearest)), nearest] / norm2[nearest]
        nn_pred = nn_gain[:, None] * bank[nearest]
        input_nn[hr, hslt] = nn_pred
        nn_directions[hr, hslt] = bank[nearest]
        nn_slot_num += float(np.square(target - nn_pred).sum())
    # Stronger hindsight control: choose the per-slot ray as above, but grant
    # joint least-squares coefficients to cancel errors in the observed sum.
    # This still does not optimize the combinatorial direction selection.
    def joint_numerator(directions):
        value = 0.
        for token in range(64):
            mask = seen_mask[token]
            if not mask.any():
                continue
            target = (hs[token, mask, None] * hy[token, mask]).sum(axis=0)
            basis = directions[token, mask].T
            projected = basis @ np.linalg.lstsq(basis, target, rcond=None)[0]
            value += float(np.square(target - projected).sum())
        return value
    def numerator(pred):
        return float(np.square((hs[:, :, None] * (hy - pred)).sum(axis=1)).sum())
    return {'layer': layer, 'seen_assignments': seen, 'total_assignments': hi.size,
            'denominator': denom, 'oracle_ray_numerator': numerator(ray),
            'nearest_input_ray_numerator': numerator(input_nn),
            'joint_ray_numerator': joint_numerator(ray_directions),
            'joint_input_numerator': joint_numerator(nn_directions),
            'oracle_slot_num': oracle_slot_num, 'nearest_input_slot_num': nn_slot_num,
            'seen_slot_den': oracle_slot_den,
            'oracle_ray_local_rms': float(np.sqrt(numerator(ray) / denom)),
            'nearest_input_ray_local_rms': float(np.sqrt(numerator(input_nn) / denom)),
            'joint_ray_local_rms': float(np.sqrt(joint_numerator(ray_directions) / denom)),
            'gain_quantiles': np.quantile(chosen_gain, [0, .05, .5, .95, 1]).tolist(),
            'capture_sha256': th | hh}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('capture', type=Path)
    p.add_argument('--first', type=int, required=True)
    p.add_argument('--count', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert 0 <= a.first < 40 and 0 < a.count <= 40 - a.first
    provenance = a.capture / 'receipt.json'
    source = json.loads(provenance.read_text())
    layers = []
    for layer in range(a.first, a.first + a.count):
        result = layer_scan(a.capture, layer)
        for filename, digest in result['capture_sha256'].items():
            assert source['splits'][filename.split('.')[0]]['files_sha256'][filename] == digest, filename
        layers.append(result)
        print(layer, result['seen_assignments'], result['oracle_ray_local_rms'], flush=True)
    out = {'contract': 'free true-output per-slot best signed-gain one-train-vector ray; exact unseen experts; FP64 score-weighted local sum, not native bits/model quality/serving gain',
           'source_sha256': sha(Path(__file__)), 'capture_receipt_sha256': sha(provenance), 'layers': layers}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=2) + '\n')


if __name__ == '__main__':
    main()
