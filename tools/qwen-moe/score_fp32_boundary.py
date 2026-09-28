#!/usr/bin/env python3
"""Measure the finite-FP32 boundary of seven-score Qwen routed-sum coordinates."""
import hashlib
import json
from pathlib import Path

import numpy as np

BASE = Path('../../data/qwen-moe/all-producers')
OUT = Path('../../data/qwen-moe/score-fp32-boundary/receipt.json')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def reconstruct(scores):
    # Reconstruct from seven saved FP32 values with a fixed left-to-right FP32 fold.
    acc = np.zeros(scores.shape[0], dtype=np.float32)
    for j in range(1, 8):
        acc = np.add(acc, scores[:, j], dtype=np.float32)
    return np.subtract(np.float32(1), acc, dtype=np.float32)


def main():
    provenance = json.loads((BASE / 'receipt.json').read_text())
    result = {'contract': 'Captured normalized FP32 scores and down outputs; one omitted score reconstructed by left-folded FP32 1-sum(scores[1:]). Exact positive-FP32 bit correction is tested only on the finite two-split capture. Weighted sums use FP64 as an arithmetic sensitivity observation, NOT native FP32 bits or complete-model quality.',
              'source_sha256': sha(Path(__file__)), 'capture_receipt_sha256': sha(BASE / 'receipt.json'),
              'model_sha256': provenance['model_sha256'], 'splits': {}}
    for split in ('train', 'held'):
        stats = {'rows': 0, 'nonunit_sum_rows': 0, 'changed_score_rows': 0,
                 'changed_sum_rows': 0, 'changed_sum_components': 0,
                 'sum_delta_sq': 0., 'reference_sq': 0., 'max_relative_sum_error': 0.,
                 'max_abs_score_delta': 0., 'min_ulp_correction': 0, 'max_ulp_correction': 0,
                 'max_abs_ulp_correction': 0, 'files_sha256': {}}
        for layer in range(40):
            names = [f'{split}.layer-{layer}.ffn_moe_{stem}.f32' for stem in ('weights_norm', 'down')]
            for name in names:
                path = BASE / name
                assert sha(path) == provenance['splits'][split]['files_sha256'][name]
                stats['files_sha256'][name] = provenance['splits'][split]['files_sha256'][name]
            s = np.memmap(BASE / names[0], dtype='<f4', mode='r', shape=(64, 8))
            d = np.memmap(BASE / names[1], dtype='<f4', mode='r', shape=(64, 8, 2048))
            assert np.isfinite(s).all() and np.isfinite(d).all() and np.all(s > 0)
            a0 = reconstruct(s)
            assert np.all(a0 > 0)
            raw = s[:, 0].view('<u4').astype(np.int64)
            pred = a0.view('<u4').astype(np.int64)
            ulp = raw - pred
            recovered = (pred + ulp).astype('<u4').view('<f4')
            assert np.array_equal(recovered.view('<u4'), s[:, 0].view('<u4'))
            scores_sum = np.zeros(64, dtype=np.float32)
            for j in range(8):
                scores_sum = np.add(scores_sum, s[:, j], dtype=np.float32)
            diff = (s[:, 0].astype(np.float64) - a0.astype(np.float64))[:, None] * d[:, 0].astype(np.float64)
            reference = np.einsum('ne,ned->nd', s.astype(np.float64), d.astype(np.float64), optimize=True)
            changed = ulp != 0
            stats['rows'] += 64
            stats['nonunit_sum_rows'] += int(np.count_nonzero(scores_sum != np.float32(1)))
            stats['changed_score_rows'] += int(np.count_nonzero(changed))
            stats['changed_sum_rows'] += int(np.count_nonzero(np.any(diff != 0, axis=1)))
            stats['changed_sum_components'] += int(np.count_nonzero(diff))
            stats['sum_delta_sq'] += float(np.sum(diff * diff))
            stats['reference_sq'] += float(np.sum(reference * reference))
            stats['max_relative_sum_error'] = max(stats['max_relative_sum_error'], float(np.max(np.linalg.norm(diff, axis=1) / np.linalg.norm(reference, axis=1))))
            stats['max_abs_score_delta'] = max(stats['max_abs_score_delta'], float(np.max(np.abs(s[:, 0].astype(np.float64) - a0.astype(np.float64)))))
            stats['min_ulp_correction'] = min(stats['min_ulp_correction'], int(ulp.min()))
            stats['max_ulp_correction'] = max(stats['max_ulp_correction'], int(ulp.max()))
            stats['max_abs_ulp_correction'] = max(stats['max_abs_ulp_correction'], int(np.max(np.abs(ulp))))
        stats['relative_rms_sum_error'] = (stats['sum_delta_sq'] / stats['reference_sq']) ** .5
        result['splits'][split] = stats
        print(split, {k: v for k, v in stats.items() if k != 'files_sha256'}, flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + '\n')
    print(OUT)


if __name__ == '__main__':
    main()
