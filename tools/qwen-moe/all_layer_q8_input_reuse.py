#!/usr/bin/env python3
"""Finite exact-code memoization ceiling on forty real Qwen routed producers."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from route_activation_reuse import digest, counts

BASE = Path('../../data/qwen-moe/all-producers')
QUANT = Path('../bonsai-hip/ggml/src/ggml-cuda/quantize.cu')
RUNTIME = Path('../../data/qwen-moe/runtime/current/runtime.json')


def coded_layer(base, split, layer):
    producer = base / f'{split}.layer-{layer}.attn_post_norm.f32'
    route = base / f'{split}.layer-{layer}.ffn_moe_topk.i32'
    inputs = np.fromfile(producer, dtype='<f4').reshape(-1, 64, 32)
    ids = np.fromfile(route, dtype='<i4').reshape(-1, 8)
    if len(inputs) != len(ids) or not np.isfinite(inputs).all() or np.any((ids < 0) | (ids >= 256)):
        raise ValueError(f'invalid producer or routes: {split}, {layer}')
    if any(len(set(row)) != 8 for row in ids):
        raise ValueError(f'non-distinct routed experts: {split}, {layer}')
    maxima = np.max(np.abs(inputs), axis=-1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        scaled = inputs / (maxima / np.float32(127))
        codes = np.copysign(np.floor(np.abs(scaled) + np.float32(0.5)), scaled).astype(np.int8)
    codes[maxima[..., 0] == 0] = 0
    return codes, ids, {producer.name: digest(producer), route.name: digest(route)}


def cross_split_witness(base, layer, train, held):
    a, b = train[0], held[0]
    left = {(k, row[k].tobytes()): i for i, row in enumerate(a) for k in range(64)}
    witness = []
    if not any((k, row[k].tobytes()) in left for row in b for k in range(64)):
        return witness
    x = np.fromfile(base / f'train.layer-{layer}.attn_post_norm.f32', dtype='<f4').reshape(-1, 64, 32)
    y = np.fromfile(base / f'held.layer-{layer}.attn_post_norm.f32', dtype='<f4').reshape(-1, 64, 32)
    for j, row in enumerate(b):
        for k in range(64):
            i = left.get((k, row[k].tobytes()))
            if i is None:
                continue
            xi, yj = x[i, k], y[j, k]
            sums = [float(np.float16(np.sum(v, dtype=np.float32))) for v in (xi, yj)]
            scales = [float(np.float16(np.max(np.abs(v)) / np.float32(127))) for v in (xi, yj)]
            witness.append({'train_token': i, 'held_token': j, 'k': k,
                            'common_experts': len(set(train[1][i]) & set(held[1][j])),
                            'fp16_sum': sums, 'fp16_scale': scales})
    return witness


def run(base, output):
    per_layer, files, collisions = [], {}, []
    totals = {s: Counter() for s in ('train', 'held', 'combined')}
    for layer in range(40):
        split = {s: coded_layer(base, s, layer) for s in ('train', 'held')}
        for _, _, hashes in split.values():
            files.update(hashes)
        collisions.extend({'layer': layer, **w} for w in cross_split_witness(base, layer, split['train'], split['held']))
        split['combined'] = (np.concatenate([split[s][0] for s in ('train', 'held')]),
                             np.concatenate([split[s][1] for s in ('train', 'held')]), {})
        summary = {'layer': layer}
        for s, (code, ids, _) in split.items():
            record = counts(code, ids)
            summary[s] = record
            totals[s].update({k: v for k, v in record.items() if isinstance(v, int)})
        per_layer.append(summary)
    result = {
        'contract': 'Q8_1 32-code equality is necessary, not sufficient, for exact prepared-operand equality. Same K and expert required for across-token gate/up dot memoization. Native grouped dispatch already shares the producer among experts in one token. CPU round-away-from-zero emulates roundf but is not a native GPU bit witness.',
        'domain': 'two disjoint actual GGUF callback prompt captures, 64 tokens each, all forty layers; no generated-stream claim',
        'capture_receipt_sha256': digest(base / 'receipt.json'),
        'model_sha256': json.loads((base / 'receipt.json').read_text())['model_sha256'],
        'native_quantizer_source_sha256': digest(QUANT),
        'runtime_receipt_sha256': digest(RUNTIME),
        'source_sha256': digest(Path(__file__)),
        'input_sha256': files,
        'layers': per_layer,
        'cross_split_code_collisions': collisions,
        'totals': {s: dict(totals[s]) for s in totals},
        'conditional_one_read_bytes_per_token': 2626187904,
        'all_40_layer_q8_input_bytes_per_token': 40 * 64 * 36,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'totals': result['totals'],
                      'cross_split_collision_locations': [(r['layer'], r['train_token'], r['held_token'], r['k']) for r in collisions],
                      'source_sha256': result['source_sha256']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=BASE)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.base, args.output)
