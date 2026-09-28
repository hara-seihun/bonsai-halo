#!/usr/bin/env python3
"""CPU oracle for a Q8-first, exact-score Qwen router; no native bit-identity claim."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

BASE = Path('../../data/qwen-moe')


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def q8(a):
    b = a.reshape(*a.shape[:-1], 64, 32)
    scale = (np.max(np.abs(b), axis=-1) / 127).astype(np.float16).astype(np.float64)
    code = np.clip(np.rint(np.divide(b, scale[..., None], out=np.zeros_like(b), where=scale[..., None] != 0)), -127, 127).astype(np.int8)
    return code, scale


def layer_observation(layer, model, meta, offset, capture):
    entry = meta[f'blk.{layer}.ffn_gate_inp.weight']
    assert entry['type'] == 'F32' and entry['bytes'] == 256 * 2048 * 4
    w = np.memmap(model, dtype='<f4', offset=offset + entry['offset'], shape=(256, 2048)).astype(np.float64)
    wc, ws = q8(w)
    qw = (wc.astype(np.float64) * ws[..., None]).reshape(256, 2048)
    wd = (w - qw).reshape(256, 64, 32)
    residual_norm = np.linalg.norm(wd, axis=-1)
    quant_norm = np.linalg.norm(qw.reshape(256, 64, 32), axis=-1)
    result = {}
    for split in ('train', 'held'):
        x_path = capture / f'{split}.layer-{layer}.attn_post_norm.f32'
        id_path = capture / f'{split}.layer-{layer}.ffn_moe_topk.i32'
        x = np.fromfile(x_path, '<f4').reshape(64, 2048).astype(np.float64)
        ids = np.fromfile(id_path, '<i4').reshape(64, 8)
        xc, xs = q8(x)
        qx = (xc.astype(np.float64) * xs[..., None]).reshape(64, 2048)
        true = x @ w.T
        prediction = qx @ qw.T
        assert all(set(np.argsort(-row, kind='stable')[:8]) == set(route) for row, route in zip(true, ids))
        delta = (x - qx).reshape(64, 64, 32)
        xn = np.linalg.norm(x.reshape(64, 64, 32), axis=-1)
        dn = np.linalg.norm(delta, axis=-1)
        # W.x - QW.Qx = (W-QW).x + QW.(x-Qx). All factors are
        # known before the 256 exact F32 dots; no exact logit is consulted.
        weight_bound = xn @ residual_norm.T
        activation_bound = dn @ quant_norm.T
        bound_group = weight_bound + activation_bound
        bound_global = np.linalg.norm(x, axis=1)[:, None] * np.linalg.norm(w - qw, axis=1)[None, :] + np.linalg.norm(delta.reshape(64, 2048), axis=1)[:, None] * np.linalg.norm(qw, axis=1)[None, :]
        assert np.all(np.abs(prediction - true) <= bound_group * (1 + 1e-11) + 1e-12)
        counters = {}
        for name, center, bounds in (('global', prediction, bound_global), ('group', prediction, bound_group),
                                     ('weight_only', x @ qw.T, weight_bound),
                                     ('activation_only', qx @ w.T, dn @ np.linalg.norm(w.reshape(256, 64, 32), axis=-1).T)):
            certificate = []
            stable = 0
            fallback = []
            gap_ratio = []
            for t in range(64):
                rank = np.argsort(-center[t], kind='stable')
                chosen, other = rank[:8], rank[8:]
                same = set(chosen) == set(ids[t])
                stable += int(same)
                # The eight predicted winners' lower intervals must all exceed
                # the 248 outsiders' upper intervals. Strict inequality also
                # resolves ties without relying on tie-breaking convention.
                safe = np.min(center[t, chosen] - bounds[t, chosen]) > np.max(center[t, other] + bounds[t, other])
                certificate.append(bool(safe))
                if safe:
                    assert same
                else:
                    fallback.append(t)
                gap_ratio.append(float((np.min(center[t, chosen])-np.max(center[t, other])) / (np.max(bounds[t, chosen]) + np.max(bounds[t, other]) + 1e-30)))
            counters[name] = {'certified': sum(certificate), 'stable': stable, 'fallback': fallback,
                              'margin_over_interval_median': float(np.median(gap_ratio)),
                              'margin_over_interval_max': float(np.max(gap_ratio))}
        result[split] = {'input_sha256': digest(x_path), 'route_sha256': digest(id_path), 'bounds': counters,
                         'max_actual_q8_logit_error': float(np.max(np.abs(prediction - true))),
                         'max_group_bound': float(np.max(bound_group)),
                         'max_global_bound': float(np.max(bound_global))}
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--first', type=int, default=0)
    p.add_argument('--last', type=int, default=40)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    assert 0 <= args.first < args.last <= 40
    traffic = BASE / 'traffic.json'
    model = BASE / 'Qwen3.6-35B-A3B-UD-Q4_K_M.gguf'
    capture = BASE / 'all-producers'
    t = json.loads(traffic.read_text())
    offset = (t['header_bytes'] + 31) // 32 * 32
    meta = {entry['name']: entry for entry in t['tensors']}
    results = {}
    for layer in range(args.first, args.last):
        results[str(layer)] = layer_observation(layer, model, meta, offset, capture)
        print(layer, 'held group certified', results[str(layer)]['held']['bounds']['group']['certified'], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'contract': 'Ideal-real CPU F32-installed-router reference; Q8 block32 symmetric signed activation and router weights with FP16 scales. Certificate from original input and offline precomputed router residual/quantized norms. No native FP32 or full-model timing claim.',
                                       'source_sha256': digest(__file__), 'traffic_sha256': digest(traffic),
                                       'model_sha256': json.loads((BASE / 'acquisition.json').read_text())['sha256'],
                                       'capture_receipt_sha256': digest(capture / 'receipt.json'), 'layers': results}, indent=2) + '\n')


if __name__ == '__main__':
    main()
