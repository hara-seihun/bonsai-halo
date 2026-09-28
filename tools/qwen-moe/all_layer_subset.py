#!/usr/bin/env python3
"""Exact finite-subset omission oracle on the retained 40-layer Qwen capture."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

MASKS = np.arange(256, dtype=np.uint16)
OMIT = ((MASKS[:, None] >> np.arange(8)) & 1).astype(np.float64)
COUNT = OMIT.sum(axis=1).astype(np.int64)
TOLERANCES = (0.01, 0.05, 0.10, 0.20)
FULL_BYTES = 2_626_187_904


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def layer(root, provenance, split, number, n):
    data = {}
    for name, shape in (('ffn_moe_weights_norm', (n, 8)), ('ffn_moe_down', (n, 8, 2048))):
        path = root / f'{split}.layer-{number}.{name}.f32'
        assert path.stat().st_size == np.prod(shape) * 4
        assert digest(path) == provenance['splits'][split]['files_sha256'][path.name]
        data[name] = np.memmap(path, dtype='<f4', shape=shape, mode='r')
    scores = data['ffn_moe_weights_norm'].astype(np.float64)
    down = data['ffn_moe_down']
    assert np.isfinite(scores).all() and np.isfinite(down).all()
    assert np.all(scores[:, :-1] >= scores[:, 1:]) and np.all(scores > 0)
    assert np.allclose(scores.sum(axis=1), 1, atol=1e-5)
    # Build the full vector sum and the independent eight-dimensional Gram.
    # An omitted subset m has squared error m^T G m (cancellations included).
    contribution = down.astype(np.float64) * scores[:, :, None]
    gram = contribution @ contribution.transpose(0, 2, 1)
    norm2 = gram.sum(axis=(1, 2))
    assert np.all(norm2 > 0)
    square = np.einsum('mi,nij,mj->nm', OMIT, gram, OMIT, optimize=True)
    assert np.min(square) >= -1e-12 * np.max(norm2)
    square = np.maximum(square, 0)
    best_by_count = np.stack([square[:, COUNT == k].min(axis=1) for k in range(9)], axis=1)
    score_by_count = np.stack([square[:, ((1 << k) - 1) << (8 - k)] for k in range(9)], axis=1)
    pair_beats_single = int(np.count_nonzero(best_by_count[:, 2] < best_by_count[:, 1]))
    return norm2, best_by_count, score_by_count, pair_beats_single


def frontier(norm2, best, score, expert_bytes):
    total = len(norm2)
    result = {}
    for tolerance in TOLERANCES:
        threshold = tolerance * tolerance * norm2[:, None]
        best_count = np.max(np.where(best <= threshold, np.arange(9), -1), axis=1)
        score_count = np.max(np.where(score <= threshold, np.arange(9), -1), axis=1)
        assert np.all(best_count >= score_count) and np.all(score_count >= 0)
        best_err = best[np.arange(total), best_count]
        score_err = score[np.arange(total), score_count]
        best_assignments = int(best_count.sum())
        result[str(tolerance)] = {
            'oracle_skips': best_assignments,
            'score_prefix_skips': int(score_count.sum()),
            'oracle_histogram_0_to_8': np.bincount(best_count, minlength=9).tolist(),
            'oracle_local_relative_rms': float(np.sqrt(best_err.sum() / norm2.sum())),
            'score_local_relative_rms': float(np.sqrt(score_err.sum() / norm2.sum())),
            'oracle_conditional_complete_one_read_bytes_percent':
                100 * np.dot(best_count, expert_bytes) / (total / 40 * FULL_BYTES),
        }
    result['fixed_retained_four'] = {
        'oracle_relative_rms': float(np.sqrt(best[:, 4].sum() / norm2.sum())),
        'score_prefix_relative_rms': float(np.sqrt(score[:, 4].sum() / norm2.sum())),
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    provenance_path = args.capture / 'receipt.json'
    provenance = json.loads(provenance_path.read_text())
    traffic_path = args.capture.parent / 'traffic.json'
    traffic = json.loads(traffic_path.read_text())
    expert_bytes_by_layer = [traffic['layers'][str(i)]['bank_bytes'] // 256 for i in range(40)]
    assert sum(expert_bytes_by_layer) * 8 == traffic['active_routed_bytes_per_token']
    panels = {}
    for split in ('train', 'held'):
        n = provenance['splits'][split]['tokens']
        assert n == 64
        per_layer = []
        norms, bests, scores, bytes_per_decision = [], [], [], []
        for number in range(40):
            norm2, best, score, pairs = layer(args.capture, provenance, split, number, n)
            norms.append(norm2)
            bests.append(best)
            scores.append(score)
            cost = expert_bytes_by_layer[number]
            bytes_per_decision.extend([cost] * n)
            per_layer.append({'layer': number, 'expert_image_bytes': cost,
                              'pair_beats_single': pairs,
                              'frontier': frontier(norm2, best, score, np.full(n, cost))})
        panels[split] = {'tokens_per_layer': n, 'layers': per_layer,
                         'frontier': frontier(np.concatenate(norms), np.concatenate(bests), np.concatenate(scores), np.array(bytes_per_decision)),
                         'pair_beats_single': sum(row['pair_beats_single'] for row in per_layer)}
        print(split, json.dumps({k: v for k, v in panels[split].items() if k != 'layers'}, indent=2))
    output = {'contract': 'FP64 Gram exhaustive 256-subset oracle over actual callback Qwen GGUF routed down vectors and normalized FP32 scores at all 40 layers. Local Euclidean loss, not composed-model NLL, native FP32 identity, measured physical bytes or executable skip. Conditional bytes assume one uncached selected expert image read per assignment and free hindsight selection.',
              'capture_receipt_sha256': digest(provenance_path),
              'traffic_sha256': digest(traffic_path),
              'source_sha256': digest(Path(__file__)),
              'expert_image_bytes_by_layer': expert_bytes_by_layer,
              'complete_one_read_bytes_per_token': FULL_BYTES,
              'panels': panels}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + '\n')


if __name__ == '__main__':
    main()
