#!/usr/bin/env python3
"""Ideal norm-informed prefix certificate on actual Qwen routed outputs."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

FULL_BYTES = 2_626_187_904
TOLERANCES = (0.01, 0.05, 0.10)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def measure(weights, down, order, tolerance):
    v = np.asarray(down, dtype=np.float64) * np.asarray(weights, dtype=np.float64)[..., None]
    rows = np.arange(len(v))[:, None]
    v = v[rows, order]
    norms = np.linalg.norm(v, axis=2)
    target = v.sum(axis=1)
    total_norm = np.linalg.norm(target, axis=1)
    assert np.all(total_norm > 0)
    prefix = np.concatenate((np.zeros((len(v), 1, v.shape[2])), np.cumsum(v, axis=1)), axis=1)
    remaining = np.concatenate((norms.sum(axis=1, keepdims=True),
                                norms.sum(axis=1, keepdims=True) - np.cumsum(norms, axis=1)), axis=1)
    prefix_norm = np.linalg.norm(prefix, axis=2)
    bound = np.divide(remaining, prefix_norm - remaining,
                      out=np.full_like(remaining, np.inf), where=prefix_norm > remaining)
    error = np.linalg.norm(target[:, None, :] - prefix, axis=2) / total_norm[:, None]
    assert np.all(error[bound <= tolerance] <= tolerance + 1e-12)
    skips = 8 - np.arange(9)
    certified = np.max(np.where(bound <= tolerance, skips, -1), axis=1)
    actual = np.max(np.where(error <= tolerance, skips, -1), axis=1)
    assert np.all(certified <= actual)
    return certified, actual, error[:, -2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('capture', type=Path)
    p.add_argument('--out', required=True, type=Path)
    a = p.parse_args()
    receipt = a.capture / 'receipt.json'
    provenance = json.loads(receipt.read_text())
    traffic = a.capture.parent / 'traffic.json'
    inventory = json.loads(traffic.read_text())
    layer_cost = [inventory['layers'][str(i)]['bank_bytes'] // 256 for i in range(40)]
    assert sum(layer_cost) * 8 == inventory['active_routed_bytes_per_token']
    result = {}
    for split in ('train', 'held'):
        n = provenance['splits'][split]['tokens']
        assert n == 64
        aggregate = {str(t): {order: {'certified': [], 'actual': [], 'certified_bytes': [],
                                     'actual_bytes': [], 'rank8_error': []}
                                for order in ('router', 'norm_oracle')} for t in TOLERANCES}
        files = {}
        for layer in range(40):
            loaded = {}
            for name, shape in (('ffn_moe_weights_norm', (n, 8)), ('ffn_moe_down', (n, 8, 2048))):
                path = a.capture / f'{split}.layer-{layer}.{name}.f32'
                assert path.stat().st_size == np.prod(shape) * 4
                digest = sha(path)
                assert digest == provenance['splits'][split]['files_sha256'][path.name]
                files[path.name] = digest
                loaded[name] = np.memmap(path, dtype='<f4', mode='r', shape=shape)
                assert np.isfinite(loaded[name]).all()
            weights = loaded['ffn_moe_weights_norm']
            down = loaded['ffn_moe_down']
            assert np.all(weights > 0) and np.allclose(weights.sum(axis=1), 1, atol=1e-5)
            contributions = np.asarray(down, dtype=np.float64) * np.asarray(weights, dtype=np.float64)[..., None]
            norms = np.linalg.norm(contributions, axis=2)
            orders = {'router': np.broadcast_to(np.arange(8), (n, 8)),
                      'norm_oracle': np.argsort(-norms, axis=1, kind='stable')}
            for name, order in orders.items():
                for t in TOLERANCES:
                    cert, actual, rank8 = measure(weights, down, order, t)
                    dst = aggregate[str(t)][name]
                    dst['certified'].extend(cert.tolist())
                    dst['actual'].extend(actual.tolist())
                    dst['certified_bytes'].extend((cert * layer_cost[layer]).tolist())
                    dst['actual_bytes'].extend((actual * layer_cost[layer]).tolist())
                    dst['rank8_error'].extend(rank8.tolist())
        panels = {}
        for t, order_data in aggregate.items():
            panels[t] = {}
            for name, item in order_data.items():
                cert = np.asarray(item['certified'], dtype=np.int64)
                actual = np.asarray(item['actual'], dtype=np.int64)
                panels[t][name] = {
                    'certified_skips': int(cert.sum()), 'actual_prefix_skips': int(actual.sum()),
                    'certified_cases': int(np.count_nonzero(cert)), 'actual_prefix_cases': int(np.count_nonzero(actual)),
                    'certified_histogram_0_to_8': np.bincount(cert, minlength=9).tolist(),
                    'certified_conditional_one_read_percent': float(100 * sum(item['certified_bytes']) / (n * FULL_BYTES)),
                    'actual_prefix_conditional_one_read_percent': float(100 * sum(item['actual_bytes']) / (n * FULL_BYTES)),
                    'rank8_actual_error_median': float(np.median(item['rank8_error'])),
                }
        result[split] = {'cases': n * 40, 'files_sha256': files, 'panels': panels}
    output = {'contract': 'FP64 recombination of captured installed-GGUF FP32 scores/down outputs; prefix stop in descending router score or free true per-token weighted-output norm, free exact omitted norms and exact partial norm; triangle bound B/(||prefix||-B) certifies relative local Euclidean error. Exact norms require omitted expert evaluation unless independently bounded. No residual/shared observer, FP32 map identity, paid image, full-model quality or physical TPS. Conditional byte comparator is complete one-read selected weights.',
              'capture_receipt_sha256': sha(receipt), 'traffic_sha256': sha(traffic),
              'source_sha256': sha(Path(__file__)), 'complete_one_read_bytes_per_token': FULL_BYTES,
              'panels': result}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(output, indent=2) + '\n')
    for split, data in result.items():
        print(split, {t: {k: (v['certified_skips'], v['actual_prefix_skips'], v['certified_conditional_one_read_percent']) for k, v in d.items()} for t, d in data['panels'].items()})


if __name__ == '__main__':
    main()
