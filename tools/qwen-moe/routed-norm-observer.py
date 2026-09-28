#!/usr/bin/env python3
"""Finite projective observer for the actual Qwen routed sum (not its residual)."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

FULL_BYTES = 2_626_187_904
MASK = np.arange(256, dtype=np.uint16)
OMIT = ((MASK[:, None] >> np.arange(8)) & 1).astype(np.float64)
COUNT = OMIT.sum(axis=1)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('capture', type=Path)
    p.add_argument('--out', required=True, type=Path)
    a = p.parse_args()
    receipt = a.capture / 'receipt.json'
    provenance = json.loads(receipt.read_text())
    traffic = a.capture.parent / 'traffic.json'
    traffic_data = json.loads(traffic.read_text())
    layer_cost = [traffic_data['layers'][str(i)]['bank_bytes'] // 256 for i in range(40)]
    assert sum(layer_cost) * 8 == traffic_data['active_routed_bytes_per_token']
    result = {}
    for split in ('train', 'held'):
        n = provenance['splits'][split]['tokens']
        assert n == 64
        errors = []
        target_norms = []
        costs = []
        for layer in range(40):
            arrays = []
            for name, shape in [('ffn_moe_weights_norm', (n, 8)), ('ffn_moe_down', (n, 8, 2048))]:
                file = a.capture / f'{split}.layer-{layer}.{name}.f32'
                assert file.stat().st_size == np.prod(shape) * 4
                assert sha(file) == provenance['splits'][split]['files_sha256'][file.name]
                arrays.append(np.memmap(file, dtype='<f4', mode='r', shape=shape))
            weights, down = arrays
            assert np.isfinite(weights).all() and np.isfinite(down).all()
            assert np.all(weights > 0) and np.allclose(weights.sum(axis=1), 1, atol=1e-5)
            c = np.asarray(down, dtype=np.float64) * np.asarray(weights, dtype=np.float64)[:, :, None]
            gram = c @ c.transpose(0, 2, 1)
            target = gram.sum(axis=(1, 2))
            removed = np.einsum('mi,nij->nmj', OMIT, gram, optimize=True).sum(axis=2)
            removed_sq = np.einsum('mi,nij,mj->nm', OMIT, gram, OMIT, optimize=True)
            kept_sq = target[:, None] - 2 * removed + removed_sq
            dot = target[:, None] - removed
            assert np.all(target > 0) and np.all(kept_sq[:, :-1] > 0)
            cosine = np.clip(dot[:, :-1] / np.sqrt(target[:, None] * kept_sq[:, :-1]), -1, 1)
            # RMS of the difference between unit-length complete and retained sums.
            angle_error = np.sqrt(np.maximum(0, 2 - 2 * cosine))
            angle_error = np.concatenate((angle_error, np.full((n, 1), np.inf)), axis=1)
            errors.append(angle_error)
            target_norms.append(target)
            costs.extend([layer_cost[layer]] * n)
        e = np.concatenate(errors)
        norms = np.concatenate(target_norms)
        costs = np.asarray(costs)
        assert np.all(e[:, 0] < 1e-6)
        panels = {}
        for tolerance in (.01, .05, .10, .20):
            eligible = e <= tolerance
            best = np.max(np.where(eligible, COUNT[None, :], -1), axis=1).astype(int)
            prefix_masks = np.asarray([((1 << k) - 1) << (8-k) for k in range(9)])
            prefix = np.max(np.where(eligible[:, prefix_masks], np.arange(9)[None, :], -1), axis=1)
            assert np.all(best >= prefix) and np.all(prefix >= 0)
            panels[str(tolerance)] = {
                'oracle_omitted_assignments': int(best.sum()),
                'score_prefix_omitted_assignments': int(prefix.sum()),
                'oracle_histogram_0_to_8': np.bincount(best, minlength=9).tolist(),
                'oracle_conditional_complete_one_read_bytes_percent': float(100 * np.dot(best, costs) / (n * FULL_BYTES)),
                'score_conditional_complete_one_read_bytes_percent': float(100 * np.dot(prefix, costs) / (n * FULL_BYTES)),
            }
        # Best one-skip, with positive radial gain granted free. The zero-skip
        # projective mismatch is mathematically zero; FP64 Gram cancellation may
        # report a few ULPs, which are below the stated thresholds.
        one = np.min(e[:, COUNT == 1], axis=1)
        result[split] = {'decisions': len(e), 'frontier': panels,
                         'best_one_skip_median_unit_vector_error': float(np.median(one)),
                         'best_one_skip_p95_unit_vector_error': float(np.quantile(one, .95)),
                         'best_one_skip_max_unit_vector_error': float(np.max(one)),
                         'positive_target_norm_min': float(np.sqrt(np.min(norms)))}
    output = {'contract': 'FP64 Gram, all 256 unchanged-routed-output subsets at each of 40x64 actual GGUF captured cases; compare unit-length routed sums only, granting free positive radial rescaling, free hindsight subset, zero residual/shared branch and ideal zero-epsilon norm. Does NOT bound the actual residual-plus-shared next normalization, full-model quality, native FP32 identity, physical traffic or TPS. Conditional image bytes assume one uncached image per selected expert and free skip decision.',
              'capture_receipt_sha256': sha(receipt), 'traffic_sha256': sha(traffic),
              'source_sha256': sha(Path(__file__)), 'layer_image_bytes': layer_cost,
              'complete_one_read_bytes_per_token': FULL_BYTES, 'panels': result}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
