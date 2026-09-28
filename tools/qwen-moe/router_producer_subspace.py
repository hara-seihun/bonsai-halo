#!/usr/bin/env python3
"""Observed-domain shared low-rank router input, with an honest train/held split.

A centered train PCA carrier is reused by all 256 router rows. Its score and
route errors are compared with the *same FP64 F32-image dot*, not native bits.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

BASE = Path('../../data/qwen-moe')
WIDTHS = (8, 16, 32, 63)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def inputs(root, layer, split, provenance):
    stems = [('attn_post_norm', 'f32', (64, 2048)),
             ('ffn_moe_topk', 'i32', (64, 8)),
             ('ffn_moe_weights_norm', 'f32', (64, 8)),
             ('ffn_moe_down', 'f32', (64, 8, 2048))]
    paths = {}
    arrays = []
    for stem, dtype, shape in stems:
        path = root / f'{split}.layer-{layer}.{stem}.{dtype}'
        assert path.stat().st_size == np.prod(shape) * 4
        digest = sha(path)
        assert provenance['splits'][split]['files_sha256'][path.name] == digest
        paths[path.name] = digest
        arrays.append(np.memmap(path, mode='r', dtype='<' + dtype[0] + '4', shape=shape))
    return arrays, paths


def layer_result(base, layer, metadata, offset, provenance):
    capture = base / 'all-producers'
    (tx, _, _, _), th = inputs(capture, layer, 'train', provenance)
    (hx, ids, native_scores, down), hh = inputs(capture, layer, 'held', provenance)
    matrix = np.memmap(base / 'Qwen3.6-35B-A3B-UD-Q4_K_M.gguf', dtype='<f4', mode='r',
                       offset=offset + metadata['offset'], shape=(256, 2048))
    w = np.asarray(matrix, dtype=np.float64)
    x = np.asarray(tx, dtype=np.float64)
    y = np.asarray(hx, dtype=np.float64)
    center = x.mean(axis=0)
    _, s, vt = np.linalg.svd(x - center, full_matrices=False)
    assert s[62] > 1e-8
    reference = y @ w.T
    original_sets = np.argsort(-reference, axis=1, kind='stable')[:, :8]
    assert all(set(a) == set(b) for a, b in zip(original_sets, ids)), layer
    cases = {}
    for rank in WIDTHS:
        basis = vt[:rank]
        # The actual program evaluates c=(x-mu)B^T, then (WB^T)c+Wmu.
        # Mathematically equal to W times the projected input, but its finite
        # FP32 behavior would have to be separately qualified for native use.
        reconstructed = center + ((y - center) @ basis.T) @ basis
        predicted = reconstructed @ w.T
        choices = np.argsort(-predicted, axis=1, kind='stable')[:, :8]
        stable = np.array([set(a) == set(b) for a, b in zip(choices, ids)])
        logit_delta = predicted - reference
        score_sse = sum_sse = sum_ref = 0.
        max_score_delta = 0.
        for t in np.flatnonzero(stable):
            selected = np.asarray(ids[t], dtype=np.int64)
            def softmax(logits):
                z = logits[selected]
                e = np.exp(z - z.max())
                return e / e.sum()
            p0, p1 = softmax(reference[t]), softmax(predicted[t])
            delta = p1 - p0
            max_score_delta = max(max_score_delta, float(np.max(np.abs(delta))))
            score_sse += float(delta @ delta)
            out = np.asarray(down[t], dtype=np.float64)
            sum_sse += float(np.square(delta @ out).sum())
            sum_ref += float(np.square(p0 @ out).sum())
        cases[str(rank)] = {
            'stable_route_sets': int(stable.sum()), 'route_changes': int((~stable).sum()),
            'relative_input_rms': float(np.linalg.norm(y - reconstructed) / np.linalg.norm(y)),
            'relative_router_logit_rms': float(np.linalg.norm(logit_delta) / np.linalg.norm(reference)),
            'max_logit_change': float(np.max(np.abs(logit_delta))),
            'stable_score_sse': score_sse, 'stable_sum_sse': sum_sse,
            'stable_sum_ref': sum_ref, 'max_stable_score_change': max_score_delta,
            'held_route_changed_tokens': np.flatnonzero(~stable).tolist(),
            'stored_f32_bytes_per_layer': 4 * (rank * (2048 + 256) + 2048 + 256),
            'online_multiply_count': rank * (2048 + 256),
        }
    return {'cases': cases, 'input_sha256': th | hh,
            'router_tensor_sha256': hashlib.sha256(matrix.tobytes()).hexdigest(),
            'native_score_max_difference': float(np.max(np.abs(np.asarray(native_scores) -
                np.array([np.exp(reference[t, ids[t]] - reference[t, ids[t]].max()) /
                    np.exp(reference[t, ids[t]] - reference[t, ids[t]].max()).sum() for t in range(64)])))),
            'singular_first_last': [float(s[0]), float(s[62])]}


def aggregate(output):
    shards = [output.parent / f'part-{start}.json' for start in (0, 10, 20, 30)]
    data = [json.loads(path.read_text()) for path in shards]
    for key in ('contract', 'source_sha256', 'model_sha256', 'traffic_sha256', 'capture_receipt_sha256'):
        assert len({entry[key] for entry in data}) == 1
    assert data[0]['source_sha256'] == sha(__file__)
    rows = {layer: entry for shard in data for layer, entry in shard['layers'].items()}
    assert set(rows) == set(map(str, range(40)))
    summary = {}
    for rank in WIDTHS:
        cases = [row['cases'][str(rank)] for row in rows.values()]
        summary[str(rank)] = {
            'held_route_sets': len(cases) * 64,
            'stable_route_sets': sum(c['stable_route_sets'] for c in cases),
            'held_tokens_with_any_layer_route_change': int(np.count_nonzero(
                np.bincount([t for c in cases for t in c['held_route_changed_tokens']], minlength=64))),
            'relative_held_input_rms_median_layer': float(np.median([c['relative_input_rms'] for c in cases])),
            'relative_held_router_logit_rms_median_layer': float(np.median([c['relative_router_logit_rms'] for c in cases])),
            'stable_fixed_route_weighted_sum_rms': (sum(c['stable_sum_sse'] for c in cases) /
                                                    sum(c['stable_sum_ref'] for c in cases)) ** .5 if sum(c['stable_sum_ref'] for c in cases) else None,
            'stored_f32_bytes_all_layers': sum(c['stored_f32_bytes_per_layer'] for c in cases),
            'online_multiplies_all_layers_per_token': sum(c['online_multiply_count'] for c in cases),
        }
    result = {'contract': data[0]['contract'], 'source_sha256': sha(__file__),
              'model_sha256': data[0]['model_sha256'], 'capture_receipt_sha256': data[0]['capture_receipt_sha256'],
              'parts_sha256': {path.name: sha(path) for path in shards}, 'summary': summary,
              'baseline_router_f32_bytes_all_layers': 40 * 256 * 2048 * 4,
              'baseline_router_multiplies_all_layers_per_token': 40 * 256 * 2048}
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, default=BASE)
    p.add_argument('--first', type=int)
    p.add_argument('--count', type=int)
    p.add_argument('--aggregate', action='store_true')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.aggregate:
        aggregate(a.output)
        return
    assert a.first is not None and a.count is not None
    assert 0 <= a.first < 40 and 0 < a.count <= 40 - a.first
    traffic_path = a.base / 'traffic.json'
    traffic = json.loads(traffic_path.read_text())
    metadata = {t['name']: t for t in traffic['tensors']}
    offset = (traffic['header_bytes'] + 31) // 32 * 32
    provenance_path = a.base / 'all-producers/receipt.json'
    provenance = json.loads(provenance_path.read_text())
    rows = {}
    for layer in range(a.first, a.first + a.count):
        entry = metadata[f'blk.{layer}.ffn_gate_inp.weight']
        assert entry['type'] == 'F32' and entry['bytes'] == 256 * 2048 * 4
        rows[str(layer)] = layer_result(a.base, layer, entry, offset, provenance)
        print(layer, {r: rows[str(layer)]['cases'][str(r)]['stable_route_sets'] for r in WIDTHS}, flush=True)
    receipt = {'contract': 'Centered train-64 PCA shared input factor; held-64 F32 installed router evaluated in FP64. All 256 logits and actual eight selected IDs. Stable-route-only score and fixed-output sum error; changed routes are counted, not imputed. No native FP32 identity, paid quantization, complete language loss or speed claim.',
               'source_sha256': sha(__file__), 'model_sha256': provenance['model_sha256'],
               'traffic_sha256': sha(traffic_path), 'capture_receipt_sha256': sha(provenance_path),
               'layers': rows}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(receipt, indent=2) + '\n')


if __name__ == '__main__':
    main()
