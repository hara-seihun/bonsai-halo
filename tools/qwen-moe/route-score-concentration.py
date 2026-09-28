#!/usr/bin/env python3
"""Test score-only lowest-expert skip policies against actual routed output vectors."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

FULL_BYTES = 2_626_187_904


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def observations(root, receipt, split, costs):
    score, error, cost, denom = [], [], [], []
    for layer in range(40):
        prefix = f'{split}.layer-{layer}.'
        n = receipt['splits'][split]['tokens']
        paths = [root / (prefix + name) for name in ('ffn_moe_weights_norm.f32', 'ffn_moe_down.f32')]
        arrays = []
        for path, shape in zip(paths, ((n, 8), (n, 8, 2048))):
            assert path.stat().st_size == np.prod(shape) * 4
            assert sha(path) == receipt['splits'][split]['files_sha256'][path.name]
            arrays.append(np.memmap(path, dtype='<f4', mode='r', shape=shape))
        s, down = arrays
        assert np.isfinite(s).all() and np.isfinite(down).all()
        assert np.all(s[:, :-1] >= s[:, 1:]) and np.all(s > 0)
        contrib = down.astype(np.float64) * s.astype(np.float64)[:, :, None]
        full = contrib.sum(axis=1)
        norm = np.sum(full * full, axis=1)
        assert np.all(norm > 0)
        # Only rank-8 can be selected by this score-only decision rule.
        sq = np.sum(contrib[:, -1, :] ** 2, axis=1)
        score.extend((layer, float(x)) for x in s[:, -1])
        error.extend(np.sqrt(sq / norm).tolist())
        cost.extend([costs[layer]] * n)
        denom.extend(norm.tolist())
    return {'layer': np.array([x[0] for x in score]), 'score': np.array([x[1] for x in score]),
            'error': np.array(error), 'cost': np.array(cost), 'norm2': np.array(denom)}


def evaluate(obs, thresholds, tolerance):
    eligible = obs['score'] < thresholds[obs['layer']]
    bad = eligible & (obs['error'] > tolerance)
    return {'skips': int(eligible.sum()), 'violations': int(bad.sum()),
            'worst_selected_error': float(obs['error'][eligible].max(initial=0)),
            'conditional_complete_one_read_bytes_percent': float(100 * obs['cost'][eligible].sum() / (len(obs['error']) / 40 * FULL_BYTES)),
            'routed_sum_relative_rms': float(np.sqrt(np.sum(obs['norm2'][eligible] * obs['error'][eligible] ** 2) / obs['norm2'].sum()))}


def thresholds_for(train, tolerance, by_layer):
    result = np.full(40, np.inf)
    if by_layer:
        for l in range(40):
            bad = (train['layer'] == l) & (train['error'] > tolerance)
            if np.any(bad):
                result[l] = train['score'][bad].min()
    else:
        bad = train['error'] > tolerance
        if np.any(bad):
            result[:] = train['score'][bad].min()
    # Strict comparison permits exactly zero training violations even at score ties.
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('capture', type=Path)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    receipt_path = a.capture / 'receipt.json'
    traffic_path = a.capture.parent / 'traffic.json'
    receipt = json.loads(receipt_path.read_text())
    traffic = json.loads(traffic_path.read_text())
    costs = [traffic['layers'][str(i)]['bank_bytes'] // 256 for i in range(40)]
    train = observations(a.capture, receipt, 'train', costs)
    held = observations(a.capture, receipt, 'held', costs)
    result = {}
    for tol in (0.01, 0.05, 0.10, 0.20):
        arms = {}
        for name, by_layer in (('global', False), ('per_layer', True)):
            t = thresholds_for(train, tol, by_layer)
            arms[name] = {'thresholds': [None if np.isinf(x) else float(x) for x in t],
                          'train': evaluate(train, t, tol), 'held': evaluate(held, t, tol)}
        arms['free_hindsight_rank8'] = {}
        for split, obs in (('train', train), ('held', held)):
            ok = obs['error'] <= tol
            arms['free_hindsight_rank8'][split] = {
                'skips': int(ok.sum()),
                'conditional_complete_one_read_bytes_percent': float(100 * obs['cost'][ok].sum() / (len(ok) / 40 * FULL_BYTES))}
        result[str(tol)] = arms
    output = {'contract': 'Finite score-only policy: omit rank-8 expert if normalized selected router score is below a train-chosen threshold. Threshold is the minimum train score of an unsafe rank-8 omission, global or per layer; strict inequality guarantees zero train local-error violations. Offline FP64 local Euclidean error against captured weighted routed sum; no replacement, renormalization, downstream quality, native speed or bitwise FP32 claim. Byte fraction assumes free decision and one full selected expert image read per call.',
              'capture_receipt_sha256': sha(receipt_path), 'traffic_sha256': sha(traffic_path),
              'source_sha256': sha(Path(__file__)), 'samples_per_split': len(train['error']),
              'policy': result}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps({tol: {name: {split: value[split] for split in ('train', 'held')} for name, value in arms.items()} for tol, arms in result.items()}, indent=2))


if __name__ == '__main__':
    main()
