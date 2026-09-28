#!/usr/bin/env python3
"""Finite support and free-hindsight output-prototype oracle on real Qwen routes."""
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


def arrays(root, split, layer):
    prefix = root / f'{split}.layer-{layer}.'
    ids = np.fromfile(str(prefix) + 'ffn_moe_topk.i32', dtype='<i4').reshape(64, 8)
    scores = np.fromfile(str(prefix) + 'ffn_moe_weights_norm.f32', dtype='<f4').reshape(64, 8)
    down = np.fromfile(str(prefix) + 'ffn_moe_down.f32', dtype='<f4').reshape(64, 8, 2048)
    assert np.all((ids >= 0) & (ids < 256)) and np.isfinite(scores).all() and np.isfinite(down).all()
    assert np.all(np.sort(ids, axis=1)[:, 1:] != np.sort(ids, axis=1)[:, :-1])
    return ids, scores.astype(np.float64), down.astype(np.float64)


def layer_scan(root, layer):
    tid, _, td = arrays(root, 'train', layer)
    hid, hs, hd = arrays(root, 'held', layer)
    tc = np.bincount(tid.ravel(), minlength=256)
    hc = np.bincount(hid.ravel(), minlength=256)
    support = tc[hid]
    rows = []
    for cutoff in (0, 1, 2, 4, 8, 16):
        rows.append({'train_occurrences_at_most': cutoff,
                     'held_assignments': int(np.sum(support <= cutoff)),
                     'held_score_mass': float(np.sum(hs[support <= cutoff]))})
    flat_train_id, flat_train_down = tid.ravel(), td.reshape(-1, 2048)
    train_indices = [np.flatnonzero(flat_train_id == expert) for expert in range(256)]
    pred = np.zeros_like(hd)
    oracle = np.zeros_like(hd)
    for t in range(64):
        for j in range(8):
            e = int(hid[t, j]); ix = train_indices[e]
            if ix.size == 0:
                continue
            bank = flat_train_down[ix]
            pred[t, j] = bank.mean(axis=0)
            d2 = np.square(bank - hd[t, j]).sum(axis=1)
            oracle[t, j] = bank[np.argmin(d2)]
    weighted = hs[:, :, None] * hd
    denom = float(np.square(weighted.sum(axis=1)).sum())
    def error(candidate):
        return float(np.sqrt(np.square((weighted - hs[:, :, None] * candidate).sum(axis=1)).sum() / denom))
    unseen_exact = np.where((support == 0)[:, :, None], 0, hd)
    seen_exact = np.where((support > 0)[:, :, None], 0, hd)
    oracle_unseen_exact = np.where((support == 0)[:, :, None], hd, oracle)
    return {'layer': layer, 'train_distinct': int(np.sum(tc > 0)), 'held_distinct': int(np.sum(hc > 0)),
            'held_unseen_distinct': int(np.sum((tc == 0) & (hc > 0))),
            'train_support_histogram': np.bincount(tc, minlength=65).tolist(),
            'held_support': rows, 'local_rms_omit_all': error(np.zeros_like(hd)),
            'local_rms_train_mean': error(pred), 'local_rms_hindsight_nearest': error(oracle),
            'local_rms_unseen_omitted_seen_exact': error(unseen_exact),
            'local_rms_seen_omitted_unseen_exact': error(seen_exact),
            'local_rms_nearest_seen_unseen_exact': error(oracle_unseen_exact),
            'denominator': denom,
            'unseen_numerator': float(np.square((weighted - hs[:, :, None] * unseen_exact).sum(axis=1)).sum()),
            'nearest_seen_numerator': float(np.square((weighted - hs[:, :, None] * oracle_unseen_exact).sum(axis=1)).sum()),
            'mean_numerator': float(np.square((weighted - hs[:, :, None] * pred).sum(axis=1)).sum()),
            'oracle_numerator': float(np.square((weighted - hs[:, :, None] * oracle).sum(axis=1)).sum()),
            'zero_numerator': float(np.square(weighted.sum(axis=1)).sum())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('root', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    source_receipt = args.root / 'receipt.json'
    original = json.loads(source_receipt.read_text())
    for split in ('train', 'held'):
        for layer in range(40):
            for name, ext in (('ffn_moe_topk', 'i32'), ('ffn_moe_weights_norm', 'f32'), ('ffn_moe_down', 'f32')):
                filename = f'{split}.layer-{layer}.{name}.{ext}'
                assert sha(args.root / filename) == original['splits'][split]['files_sha256'][filename], filename
    layers = [layer_scan(args.root, layer) for layer in range(40)]
    total = 40 * 64 * 8
    result = {'contract': 'actual GGUF callback routed captures, finite train-support counts and free hindsight nearest same-expert stored output; local FP64 score-weighted sum, not model NLL or native speed',
              'source_receipt_sha256': sha(source_receipt), 'analysis_sha256': sha(Path(__file__)),
              'layers': layers,
              'summary': {'assignments': total,
                          'train_distinct_layer_experts': sum(r['train_distinct'] for r in layers),
                          'held_unseen_layer_experts': sum(r['held_unseen_distinct'] for r in layers),
                          'held_support': [{'train_occurrences_at_most': c, 'held_assignments': sum(r['held_support'][i]['held_assignments'] for r in layers), 'held_score_mass': sum(r['held_support'][i]['held_score_mass'] for r in layers)} for i, c in enumerate((0, 1, 2, 4, 8, 16))],
                          'local_rms_unseen_omitted_seen_exact': float(np.sqrt(sum(r['unseen_numerator'] for r in layers) / sum(r['denominator'] for r in layers))),
                          'local_rms_nearest_seen_unseen_exact': float(np.sqrt(sum(r['nearest_seen_numerator'] for r in layers) / sum(r['denominator'] for r in layers))),
                          'local_rms_train_mean': float(np.sqrt(sum(r['mean_numerator'] for r in layers) / sum(r['denominator'] for r in layers))),
                          'local_rms_hindsight_nearest': float(np.sqrt(sum(r['oracle_numerator'] for r in layers) / sum(r['denominator'] for r in layers)))}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['summary'], indent=2))


if __name__ == '__main__':
    main()
