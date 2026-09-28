#!/usr/bin/env python3
"""Replay a raw-top-eight/seven-exp Qwen router on captured forty-layer producers.

The score map is explicitly host NumPy FP32 after an FP64 stored-weight dot;
it is not the installed HIP router's FP32 execution order.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

BASE = Path('../../data/qwen-moe')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def layer_panel(layer, split, model, offset, cap, provenance):
    stems = {'x': ('attn_post_norm', 'f32', (64, 2048)),
             'ids': ('ffn_moe_topk', 'i32', (64, 8)),
             'scores': ('ffn_moe_weights_norm', 'f32', (64, 8)),
             'down': ('ffn_moe_down', 'f32', (64, 8, 2048))}
    inputs, hashes = {}, {}
    for key, (stem, kind, shape) in stems.items():
        path = cap / f'{split}.layer-{layer}.{stem}.{kind}'
        assert path.stat().st_size == np.prod(shape) * 4
        hashes[path.name] = sha(path)
        assert hashes[path.name] == provenance['splits'][split]['files_sha256'][path.name]
        inputs[key] = np.memmap(path, mode='r', dtype='<i4' if kind == 'i32' else '<f4', shape=shape)
    w = np.memmap(model, mode='r', dtype='<f4', offset=offset, shape=(256, 2048))
    assert np.isfinite(w).all() and np.isfinite(inputs['x']).all()
    assert np.isfinite(inputs['scores']).all() and np.isfinite(inputs['down']).all()
    ids = np.asarray(inputs['ids'])
    assert np.all((0 <= ids) & (ids < 256)) and all(len(set(row)) == 8 for row in ids)
    logits = np.asarray(inputs['x'], dtype=np.float64) @ np.asarray(w, dtype=np.float64).T
    raw_ids = np.argsort(-logits, axis=1, kind='stable')[:, :8]
    matches = np.array([set(a) == set(b) for a, b in zip(ids, raw_ids)])
    selected = np.take_along_axis(logits, ids, axis=1).astype(np.float32)
    exp = np.exp(selected - selected.max(axis=1, keepdims=True))
    proposed = exp / exp.sum(axis=1, keepdims=True, dtype=np.float32)
    native = np.asarray(inputs['scores'])
    assert np.allclose(native.sum(axis=1), 1, rtol=0, atol=2e-6)
    down = np.asarray(inputs['down'], dtype=np.float64)
    reference = np.einsum('ne,ned->nd', native.astype(np.float64), down, optimize=True)
    candidate = np.einsum('ne,ned->nd', proposed.astype(np.float64), down, optimize=True)
    delta = candidate - reference
    relative = np.linalg.norm(delta, axis=1) / np.linalg.norm(reference, axis=1)
    margin = np.sort(logits, axis=1)[:, -8] - np.sort(logits, axis=1)[:, -9]
    return dict(input_sha256=hashes, weight_sha256=hashlib.sha256(w.tobytes()).hexdigest(),
                route_set_matches=int(matches.sum()), route_mismatches=np.flatnonzero(~matches).tolist(),
                min_route_margin=float(margin.min()), median_route_margin=float(np.median(margin)),
                max_score_delta=float(np.max(np.abs(proposed - native))),
                sum_error_sq=float(np.sum(delta * delta)), sum_reference_sq=float(np.sum(reference * reference)),
                max_sum_relative=float(relative.max()), median_sum_relative=float(np.median(relative)),
                worst_token=int(np.argmax(relative)))


def aggregate(out, source, provenance):
    paths = [out / f'layer-{i:02d}.json' for i in range(40)]
    parts = [json.loads(p.read_text()) for p in paths]
    assert all(row['layer'] == i and row['source_sha256'] == source and
               row['capture_receipt_sha256'] == sha(BASE / 'all-producers/receipt.json') for i, row in enumerate(parts))
    summary = dict(contract='Observed forty-layer route and routed-sum error against installed captured scores and unchanged expert outputs. FP64 stored-F32 router dot, then host FP32 seven-exp normalization. Not native FP32 bit identity, complete-model quality or TPS.',
                   source_sha256=source, model_sha256=parts[0]['model_sha256'],
                   capture_receipt_sha256=parts[0]['capture_receipt_sha256'],
                   layer_receipts_sha256={p.name: sha(p) for p in paths}, splits={})
    for split in ('train', 'held'):
        rows = [part['splits'][split] for part in parts]
        assert all(part['model_sha256'] == summary['model_sha256'] for part in parts)
        worst = max(enumerate(rows), key=lambda x: x[1]['max_sum_relative'])
        summary['splits'][split] = dict(decisions=2560, route_set_matches=sum(r['route_set_matches'] for r in rows),
                                       min_route_margin=min(r['min_route_margin'] for r in rows),
                                       max_score_delta=max(r['max_score_delta'] for r in rows),
                                       weighted_sum_rms=(sum(r['sum_error_sq'] for r in rows) / sum(r['sum_reference_sq'] for r in rows)) ** .5,
                                       max_token_relative=worst[1]['max_sum_relative'],
                                       worst_layer=worst[0], worst_token=worst[1]['worst_token'])
    (out / 'receipt.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary['splits'], indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--first', type=int, default=0)
    p.add_argument('--last', type=int, default=40)
    p.add_argument('--aggregate', action='store_true')
    p.add_argument('--out', type=Path, default=BASE / 'router-seven-exp')
    a = p.parse_args()
    assert 0 <= a.first < a.last <= 40
    a.out.mkdir(parents=True, exist_ok=True)
    source = sha(__file__)
    provenance_path = BASE / 'all-producers/receipt.json'
    provenance = json.loads(provenance_path.read_text())
    if a.aggregate:
        aggregate(a.out, source, provenance)
        return
    traffic = json.loads((BASE / 'traffic.json').read_text())
    acquire = json.loads((BASE / 'acquisition.json').read_text())
    model = BASE / 'Qwen3.6-35B-A3B-UD-Q4_K_M.gguf'
    assert acquire['verified'] and acquire['sha256'] == provenance['model_sha256']
    assert acquire['size'] == model.stat().st_size
    with model.open('rb') as f:
        assert hashlib.sha256(f.read(traffic['header_bytes'])).hexdigest() == traffic['header_sha256']
    align = int(traffic['metadata'].get('general.alignment', 32))
    start = (traffic['header_bytes'] + align - 1) // align * align
    tensors = {t['name']: t for t in traffic['tensors']}
    cap = BASE / 'all-producers'
    for layer in range(a.first, a.last):
        t = tensors[f'blk.{layer}.ffn_gate_inp.weight']
        assert t['type'] == 'F32' and t['shape'] == [2048, 256] and t['bytes'] == 2048 * 256 * 4
        row = dict(layer=layer, source_sha256=source, model_sha256=acquire['sha256'],
                   acquisition_sha256=sha(BASE / 'acquisition.json'),
                   traffic_sha256=sha(BASE / 'traffic.json'),
                   capture_receipt_sha256=sha(provenance_path), splits={})
        for split in ('train', 'held'):
            row['splits'][split] = layer_panel(layer, split, model, start + t['offset'], cap, provenance)
        (a.out / f'layer-{layer:02d}.json').write_text(json.dumps(row, indent=2) + '\n')
        print(layer, row['splits']['held']['route_set_matches'], row['splits']['held']['max_sum_relative'], flush=True)


if __name__ == '__main__':
    main()
