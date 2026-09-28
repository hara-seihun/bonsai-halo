#!/usr/bin/env python3
"""Actual-producer shared router-input code and fixed-route output observation.

This is an offline FP64 map, not the native HIP router's FP32 instruction order.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

BASE = Path('../../data/qwen-moe')


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for part in iter(lambda: f.read(1 << 20), b''):
            h.update(part)
    return h.hexdigest()


def code(x, bits):
    bound = (1 << (bits - 1)) - 1
    groups = x.reshape(-1, 32)
    scales = (np.max(np.abs(groups), axis=1) / bound).astype(np.float16).astype(np.float64)
    assert np.all(np.isfinite(scales)) and np.all(scales > 0)
    integers = np.clip(np.rint(groups / scales[:, None]), -bound, bound)
    return (integers * scales[:, None]).reshape(-1)


def weights(logits, ids):
    selected = logits[ids]
    exp = np.exp(selected - np.max(selected))
    return exp / np.sum(exp)


def measure(layer, split, router, capture, norms):
    paths = {name: capture / f'{split}.layer-{layer}.{stem}.{dtype}' for name, stem, dtype in (
        ('x', 'attn_post_norm', 'f32'), ('ids', 'ffn_moe_topk', 'i32'),
        ('scores', 'ffn_moe_weights_norm', 'f32'), ('down', 'ffn_moe_down', 'f32'))}
    inputs = np.fromfile(paths['x'], '<f4').reshape(64, 2048).astype(np.float64)
    ids = np.fromfile(paths['ids'], '<i4').reshape(64, 8)
    scores = np.fromfile(paths['scores'], '<f4').reshape(64, 8).astype(np.float64)
    down = np.memmap(paths['down'], dtype='<f4', shape=(64, 8, 2048))
    logits = inputs @ router.T
    assert all(set(np.argsort(-row, kind='stable')[:8]) == set(route) for row, route in zip(logits, ids)), (layer, split)
    measures = {str(bits): {'stable': 0, 'certified': 0, 'score_sse': 0., 'sum_sse': 0., 'sum_reference': 0.,
                            'score_max': 0., 'max_delta': 0., 'changed': [], 'uncertified_stable': 0} for bits in (4, 8)}
    margins = []
    radius = []
    for t in range(64):
        selected = ids[t]
        outside = np.ones(256, dtype=bool)
        outside[selected] = False
        orig = logits[t]
        gap = float(np.min(orig[selected]) - np.max(orig[outside]))
        margins.append(gap)
        # For every ||d||_inf <= r, each logit's perturbation is at most
        # r * ||w_e||_1. A stronger pairwise certificate can only help.
        radius.append(gap / (np.max(norms[selected]) + np.max(norms[outside])))
        original_scores = weights(orig, selected)
        for bits in (4, 8):
            rec = measures[str(bits)]
            quantized = code(inputs[t], bits)
            delta = quantized - inputs[t]
            rec['max_delta'] = max(rec['max_delta'], float(np.max(np.abs(delta))))
            predicted = quantized @ router.T
            choice = np.argsort(-predicted, kind='stable')[:8]
            stable = set(choice) == set(selected)
            rec['stable'] += int(stable)
            # Dot uncertainty using the *known input error box*, without
            # examining the quantized logits. This is a sufficient certificate
            # for exactly the chosen rounded input (in ideal real arithmetic).
            uncertainty = np.abs(router) @ np.abs(delta)
            certified = np.min(orig[selected] - uncertainty[selected]) > np.max(orig[outside] + uncertainty[outside])
            rec['certified'] += int(certified)
            assert not certified or stable
            rec['uncertified_stable'] += int(stable and not certified)
            if not stable:
                rec['changed'].append({'token': t, 'new_ids': choice.tolist(), 'lost_ids': [int(i) for i in sorted(set(selected) - set(choice))],
                                       'gap': gap, 'max_input_error': float(np.max(np.abs(delta)))})
                continue
            p = weights(predicted, selected)
            rec['score_sse'] += float(np.sum((p - original_scores)**2))
            rec['score_max'] = max(rec['score_max'], float(np.max(np.abs(p - original_scores))))
            output = np.einsum('e,ed->d', original_scores, down[t].astype(np.float64), optimize=True)
            changed = np.einsum('e,ed->d', p, down[t].astype(np.float64), optimize=True)
            rec['sum_sse'] += float(np.sum((changed - output)**2))
            rec['sum_reference'] += float(np.sum(output**2))
    for rec in measures.values():
        rec['rms_fixed_route'] = (rec['sum_sse'] / rec['sum_reference'])**.5 if rec['stable'] else None
    return {'paths': {k: sha(v) for k, v in paths.items()}, 'score_native_max_delta': float(np.max(np.abs(scores - np.array([weights(r, e) for r, e in zip(logits, ids)])))),
            'gap_min': float(min(margins)), 'gap_median': float(np.median(margins)),
            'linf_radius_min': float(min(radius)), 'linf_radius_median': float(np.median(radius)), 'codes': measures}


def aggregate(directory, output):
    parts = {first: directory / f'part-{first}.json' for first in (0, 10, 20, 30)}
    shards = [json.loads(path.read_text()) for path in parts.values()]
    for key in ('source_sha256', 'traffic_sha256', 'capture_receipt_sha256', 'model_sha256', 'contract', 'bits_and_paid_input_bytes_per_token_layer'):
        assert all(shard[key] == shards[0][key] for shard in shards)
    layers = {}
    for shard in shards:
        assert not (set(layers) & set(shard['layers']))
        layers.update(shard['layers'])
    assert set(layers) == set(map(str, range(40)))
    summary = {'source_sha256': shards[0]['source_sha256'], 'model_sha256': shards[0]['model_sha256'],
               'capture_receipt_sha256': shards[0]['capture_receipt_sha256'],
               'parts_sha256': {str(first): sha(path) for first, path in parts.items()}, 'splits': {}}
    for split in ('train', 'held'):
        rows = [layers[str(i)][split] for i in range(40)]
        codes = {}
        for bits in ('4', '8'):
            entries = [row['codes'][bits] for row in rows]
            flips = [0] * 64
            for entry in entries:
                for change in entry['changed']:
                    flips[change['token']] += 1
            codes[bits] = {'stable': sum(e['stable'] for e in entries),
                           'certified': sum(e['certified'] for e in entries),
                           'tokens_with_any_route_change': sum(x > 0 for x in flips),
                           'max_changes_per_token': max(flips),
                           'fixed_route_weighted_sum_rms': (sum(e['sum_sse'] for e in entries) / sum(e['sum_reference'] for e in entries))**.5,
                           'max_fixed_route_score_change': max(e['score_max'] for e in entries),
                           'max_input_error': max(e['max_delta'] for e in entries)}
        summary['splits'][split] = {'rows': 40 * 64, 'min_top8_gap': min(row['gap_min'] for row in rows),
                                    'min_linf_robust_radius': min(row['linf_radius_min'] for row in rows),
                                    'max_native_score_difference': max(row['score_native_max_delta'] for row in rows),
                                    'codes': codes}
    output.write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary['splits'], indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--first', type=int, default=0)
    p.add_argument('--last', type=int, default=40)
    p.add_argument('--aggregate', action='store_true')
    p.add_argument('--base', type=Path, default=BASE)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.aggregate:
        aggregate(args.output.parent, args.output)
        return
    traffic_path = args.base / 'traffic.json'
    model = args.base / 'Qwen3.6-35B-A3B-UD-Q4_K_M.gguf'
    traffic = json.loads(traffic_path.read_text())
    meta = {t['name']: t for t in traffic['tensors']}
    start = (traffic['header_bytes'] + 31) // 32 * 32
    capture = args.base / 'all-producers'
    result = {'contract': 'FP64 F32-image router dot and selected-softmax, per-token signed symmetric group-32 input rounding with paid FP16 scales; actual installed producer inputs, scores and unchanged selected down outputs. Not native FP32 bit identity or complete model.',
              'source_sha256': sha(__file__), 'traffic_sha256': sha(traffic_path),
              'capture_receipt_sha256': sha(capture / 'receipt.json'),
              'model_sha256': json.loads((args.base / 'acquisition.json').read_text())['sha256'],
              'bits_and_paid_input_bytes_per_token_layer': {'4': 1152, '8': 2176, 'f32': 8192}, 'layers': {}}
    for layer in range(args.first, args.last):
        meta_row = meta[f'blk.{layer}.ffn_gate_inp.weight']
        assert meta_row['type'] == 'F32' and meta_row['bytes'] == 256 * 2048 * 4
        router = np.memmap(model, '<f4', mode='r', offset=start + meta_row['offset'], shape=(256, 2048)).astype(np.float64)
        norms = np.sum(np.abs(router), axis=1)
        result['layers'][str(layer)] = {split: measure(layer, split, router, capture, norms) for split in ('train', 'held')}
        print(f'layer {layer} ' + ' '.join(f'{split} Q4={result["layers"][str(layer)][split]["codes"]["4"]["stable"]}/64 Q8={result["layers"][str(layer)][split]["codes"]["8"]["stable"]}/64' for split in ('train', 'held')), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
