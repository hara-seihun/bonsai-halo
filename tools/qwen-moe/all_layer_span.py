#!/usr/bin/env python3
"""Hindsight scalar-span bound for unchanged actual Qwen routed expert directions."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

FULL_BYTES = 2_626_187_904


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def residual(gram, subset):
    cross = gram[:, subset, :].sum(axis=-1)
    subgram = gram[:, subset, :][:, :, subset]
    coefficients = np.linalg.solve(subgram, cross[..., None])[..., 0]
    return np.maximum(0, gram.sum(axis=(1, 2)) - np.einsum('ni,ni->n', cross, coefficients))


def layer(root, provenance, split, index, n):
    arrays = []
    for name, shape in (('ffn_moe_weights_norm', (n, 8)), ('ffn_moe_down', (n, 8, 2048))):
        path = root / f'{split}.layer-{index}.{name}.f32'
        assert path.stat().st_size == int(np.prod(shape)) * 4
        assert digest(path) == provenance['splits'][split]['files_sha256'][path.name]
        arrays.append(np.memmap(path, mode='r', dtype='<f4', shape=shape))
    scores, down = arrays
    assert np.isfinite(scores).all() and np.isfinite(down).all()
    assert np.allclose(scores.sum(axis=1), 1, atol=1e-5)
    vectors = down.astype(np.float64) * scores.astype(np.float64)[:, :, None]
    gram = vectors @ vectors.transpose(0, 2, 1)
    norm2 = gram.sum(axis=(1, 2))
    assert np.all(norm2 > 0)
    result = {}
    for retained in range(1, 8):
        by_subset = np.stack([residual(gram, np.array(c)) for c in itertools.combinations(range(8), retained)], axis=1)
        best = by_subset.min(axis=1)
        highest = residual(gram, np.arange(retained))
        assert np.all(best <= highest + 1e-10 * norm2)
        result[retained] = {'best': best, 'highest': highest}
    assert all(np.all(result[k + 1]['best'] <= result[k]['best'] + 1e-10 * norm2) for k in range(1, 7))
    # Independent 2048-dimensional projection checks for the two extreme subsets.
    for token in (0, n - 1):
        y = vectors[token].sum(axis=0)
        for subset in (np.arange(4), np.arange(7)):
            basis = vectors[token, subset].T
            coefficients = np.linalg.lstsq(basis, y, rcond=None)[0]
            direct = np.linalg.norm(y - basis @ coefficients) ** 2
            analytic = residual(gram[token:token + 1], subset)[0]
            assert abs(direct - analytic) <= 1e-8 * norm2[token]
    return norm2, result


def summarize(norms, errors, costs):
    norm2 = np.concatenate(norms)
    result = {'observations': len(norm2), 'reference_norm2': float(norm2.sum())}
    best_by_retained = np.stack([np.concatenate([entry[k]['best'] for entry in errors])
                                 for k in range(1, 8)], axis=1)
    costs = np.asarray(costs)
    result['adaptive_oracle'] = {}
    for tol in (.01, .05, .10):
        feasible = best_by_retained <= (tol * tol * norm2[:, None])
        retained = np.where(feasible.any(axis=1), feasible.argmax(axis=1) + 1, 8)
        omitted = 8 - retained
        result['adaptive_oracle'][str(tol)] = {
            'omitted_assignments': int(omitted.sum()),
            'histogram_omitted_0_to_7': np.bincount(omitted, minlength=8).tolist(),
            'conditional_complete_one_read_bytes_percent': float(
                100 * np.dot(omitted.astype(np.float64), costs) / (len(norm2) / 40 * FULL_BYTES)),
        }
    for retained in (4, 7):
        best = np.concatenate([entry[retained]['best'] for entry in errors])
        highest = np.concatenate([entry[retained]['highest'] for entry in errors])
        relative = np.sqrt(best / norm2)
        result[str(retained)] = {
            'oracle_relative_rms': float(np.sqrt(best.sum() / norm2.sum())),
            'highest_score_relative_rms': float(np.sqrt(highest.sum() / norm2.sum())),
            'oracle_median_token_relative_error': float(np.median(relative)),
            'oracle_max_token_relative_error': float(relative.max()),
            'oracle_count_below': {str(tol): int(np.count_nonzero(relative <= tol)) for tol in (.01, .05, .10, .20)},
        }
        if retained == 7:
            for tol in (.01, .05):
                possible = relative <= tol
                # Only one expert is suppressed in each qualifying observation.
                result[str(retained)][f'conditional_complete_one_read_bytes_percent_at_{tol}'] = float(
                    100 * np.dot(possible.astype(np.float64), costs) / (len(norm2) / 40 * FULL_BYTES))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture', type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    provenance_path = args.capture / 'receipt.json'
    traffic_path = args.capture.parent / 'traffic.json'
    provenance = json.loads(provenance_path.read_text())
    traffic = json.loads(traffic_path.read_text())
    expert_bytes = [traffic['layers'][str(i)]['bank_bytes'] // 256 for i in range(40)]
    assert sum(expert_bytes) * 8 == traffic['active_routed_bytes_per_token']
    panels = {}
    for split in ('train', 'held'):
        n = provenance['splits'][split]['tokens']
        assert n == 64
        norms, errors, costs = [], [], []
        for index in range(40):
            norm2, result = layer(args.capture, provenance, split, index, n)
            norms.append(norm2)
            errors.append(result)
            costs.extend([expert_bytes[index]] * n)
        panels[split] = summarize(norms, errors, np.array(costs))
        print(split, json.dumps(panels[split]))
    output = {
        'contract': 'Actual 40-layer GGUF callback down vectors and routed scores. FP64 Euclidean oracle selects subset and unconstrained independent real scalar coefficients per token using the complete target. Local routed sum, not complete language loss, native FP32 bit identity, executable cost or physical traffic. Scalar coefficients/choice are free hindsight; unchanged selected expert vectors and conditional one-read byte model.',
        'source_sha256': digest(Path(__file__)),
        'capture_receipt_sha256': digest(provenance_path),
        'traffic_sha256': digest(traffic_path),
        'model_sha256': provenance['model_sha256'],
        'expert_image_bytes_by_layer': expert_bytes,
        'panels': panels,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + '\n')


if __name__ == '__main__':
    main()
