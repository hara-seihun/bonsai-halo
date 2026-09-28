#!/usr/bin/env python3
"""Finite real-map temporal router certificates on captured consecutive tokens."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path('../../data/qwen-moe')
MODEL = ROOT / 'Qwen3.6-35B-A3B-UD-Q4_K_M.gguf'
INVENTORY = ROOT / 'traffic.json'
CAPTURE = ROOT / 'all-producers'
OUT = ROOT / 'router-temporal-certificate'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def panel(w, x, ids):
    scores = x.astype(np.float64) @ w.T
    actual = np.argsort(-scores, axis=1)[:, :8]
    if any(set(a) != set(b) for a, b in zip(actual, ids)):
        raise ValueError('stored real router scores disagree with native route')
    prev = ids[:-1]
    outsider = np.ones((len(prev), 256), dtype=bool)
    np.put_along_axis(outsider, prev, False, axis=1)
    delta = np.linalg.norm(np.diff(x.astype(np.float64), axis=0), axis=1)
    prior = scores[:-1]
    current = scores[1:]
    prev_score = np.take_along_axis(prior, prev, axis=1)
    curr_score = np.take_along_axis(current, prev, axis=1)
    prior_gap = prev_score[:, :, None] - prior[:, None, :]
    # Pairwise triangle bound directly encloses the changed logit difference.
    # This grants previous complete scores and precomputed 8x256 row distances.
    norms = np.linalg.norm(w, axis=1)
    dist2 = np.maximum(0., np.sum(w*w, axis=1)[None, :] +
                       np.sum(w*w, axis=1)[:, None] - 2 * (w @ w.T))
    distance = np.sqrt(dist2)
    pair_cert = np.all(np.where(outsider[:, None, :],
                                prior_gap > delta[:, None, None] * distance[prev], True), axis=(1, 2))
    # Stronger certificate pays eight current dots and compares them against
    # one previous outsider score plus its own Lipschitz change bound.
    upper = prior + delta[:, None] * norms[None, :]
    upper[~outsider] = -np.inf
    selected_cert = curr_score.min(axis=1) > upper.max(axis=1)
    overlap = np.array([len(set(a) & set(b)) for a, b in zip(ids[:-1], ids[1:])])
    same = overlap == 8
    if np.any((pair_cert | selected_cert) & ~same):
        raise AssertionError('sound certificate disagrees with actual route')
    # Actual eighth/ninth scores are a hindsight control, not a certificate.
    margin = np.sort(current, axis=1)[:, -8] - np.sort(current, axis=1)[:, -9]
    old_min = prev_score.min(axis=1)
    old_max = np.where(outsider, prior, -np.inf).max(axis=1)
    return dict(pairs=len(prev), unchanged=int(same.sum()), pair_certified=int(pair_cert.sum()),
                selected_certified=int(selected_cert.sum()), overlap_histogram=np.bincount(overlap, minlength=9).tolist(),
                median_delta_norm=float(np.median(delta)),
                median_previous_gap=float(np.median(old_min - old_max)),
                median_current_margin=float(np.median(margin)),
                sample=[dict(previous=int(i), same=bool(same[i]),
                             pair_certified=bool(pair_cert[i]), selected_certified=bool(selected_cert[i]))
                        for i in range(len(prev))])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--first', type=int, default=0)
    ap.add_argument('--count', type=int, default=40)
    ap.add_argument('--summarize', action='store_true')
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    if args.summarize:
        source = sha(Path(__file__))
        layers = [OUT / f'layer-{i:02d}.json' for i in range(40)]
        records = [json.loads(p.read_text()) for p in layers]
        assert all(r['layer'] == i and r['source_sha256'] == source and
                   r['inventory_sha256'] == sha(INVENTORY) for i, r in enumerate(records))
        summary = dict(source_sha256=source, inventory_sha256=sha(INVENTORY),
                       acquisition_sha256=sha(ROOT / 'acquisition.json'),
                       layer_receipts=[dict(path=p.name, sha256=sha(p), weight_sha256=r['weight_sha256'])
                                       for p, r in zip(layers, records)], splits={})
        for split in ('train', 'held'):
            rows = [r['splits'][split] for r in records]
            summary['splits'][split] = dict(**{k: sum(r[k] for r in rows) for k in
                ('pairs', 'unchanged', 'pair_certified', 'selected_certified')},
                overlap_histogram=np.sum([r['overlap_histogram'] for r in rows], axis=0).tolist())
        (OUT / 'receipt.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(json.dumps(summary['splits'], indent=2))
        return
    inv = json.loads(INVENTORY.read_text())
    acquisition = json.loads((ROOT / 'acquisition.json').read_text())
    assert acquisition['verified'] and acquisition['size'] == MODEL.stat().st_size
    assert inv['metadata']['general.architecture'] == 'qwen35moe'
    with MODEL.open('rb') as f:
        assert hashlib.sha256(f.read(inv['header_bytes'])).hexdigest() == inv['header_sha256']
    alignment = int(inv['metadata'].get('general.alignment', 32))
    base = (inv['header_bytes'] + alignment - 1) // alignment * alignment
    indexed = {t['name']: t for t in inv['tensors']}
    for layer in range(args.first, min(40, args.first + args.count)):
        tensor = indexed[f'blk.{layer}.ffn_gate_inp.weight']
        assert tensor['type'] == 'F32' and tensor['shape'] == [2048, 256]
        w = np.memmap(MODEL, dtype='<f4', offset=base + tensor['offset'], shape=(256, 2048)).astype(np.float64)
        record = dict(layer=layer, model_sha256=acquisition['sha256'],
                      source_sha256=sha(Path(__file__)), inventory_sha256=sha(INVENTORY),
                      weight_sha256=hashlib.sha256(w.astype('<f4').tobytes()).hexdigest(), splits={})
        for split in ('train', 'held'):
            prefix = CAPTURE / f'{split}.layer-{layer}'
            xp = Path(str(prefix) + '.attn_post_norm.f32')
            ip = Path(str(prefix) + '.ffn_moe_topk.i32')
            x = np.fromfile(xp, dtype='<f4').reshape(-1, 2048)
            ids = np.fromfile(ip, dtype='<i4').reshape(-1, 8)
            assert x.shape == (64, 2048) and ids.shape == (64, 8) and np.isfinite(x).all()
            record['splits'][split] = dict(input_sha256=sha(xp), ids_sha256=sha(ip), **panel(w, x, ids))
        path = OUT / f'layer-{layer:02d}.json'
        path.write_text(json.dumps(record, indent=2) + '\n')
        print(layer, *(f'{s}: {record["splits"][s]["unchanged"]}/{record["splits"][s]["pairs"]}, cert {record["splits"][s]["pair_certified"]}/{record["splits"][s]["selected_certified"]}' for s in ('train', 'held')), flush=True)


if __name__ == '__main__':
    main()
