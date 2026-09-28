#!/usr/bin/env python3
"""Free hindsight bound for reweighting a subset of real routed expert outputs."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(root, split):
    prefix = root / split
    tokens = np.fromfile(str(prefix) + '.tokens', dtype=np.int32)
    files = {
        'ids': (Path(str(prefix) + '.0.ffn_moe_topk-0.bin'), np.int32, (len(tokens), 8)),
        'scores': (Path(str(prefix) + '.0.ffn_moe_weights_norm-0.bin'), np.float32, (len(tokens), 8)),
        'down': (Path(str(prefix) + '.0.ffn_moe_down-0.bin'), np.float32, (len(tokens), 8, 2048)),
    }
    arrays = {name: np.memmap(path, dtype=dtype, mode='r', shape=shape)
              for name, (path, dtype, shape) in files.items()}
    if not np.allclose(arrays['scores'].sum(axis=1), 1, atol=1e-5):
        raise ValueError('router scores do not sum to one')
    if not np.isfinite(arrays['down']).all():
        raise ValueError('nonfinite expert output')
    hashes = {name: sha(path) for name, (path, _, _) in files.items()}
    hashes['tokens'] = sha(Path(str(prefix) + '.tokens'))
    hashes['text'] = sha(Path(str(prefix) + '.txt'))
    return arrays, hashes


def cases(data):
    scores = np.asarray(data['scores'], np.float64)
    outputs = np.asarray(data['down'], np.float64)
    order = np.argsort(-scores, axis=1, kind='stable')
    ordered = np.take_along_axis(outputs, order[:, :, None], axis=1)
    weights = np.take_along_axis(scores, order, axis=1)
    target = np.einsum('te,ted->td', weights, ordered)
    return target, ordered, weights


def evaluate(train, held):
    result = {}
    for k in (1, 2, 4, 6, 7):
        train_target, train_out, train_w = train
        pred_train = np.einsum('te,ted->td', train_w[:, :k], train_out[:, :k])
        global_scale = float(np.sum(train_target * pred_train) / np.sum(pred_train ** 2))
        result[str(k)] = {'train_selected_scalar': global_scale}
        for name, (target, out, weights) in [('train', train), ('held', held)]:
            retained = out[:, :k, :] * weights[:, :k, None]
            default = retained.sum(axis=1)
            denom = np.sum(target ** 2)
            def rms(y):
                return float(np.sqrt(np.sum((target - y) ** 2) / denom))
            renorm = default / weights[:, :k].sum(axis=1)[:, None]
            token_scale = np.einsum('td,td->t', target, default) / np.einsum('td,td->t', default, default)
            # The target determines every coefficient: this is an optimistic
            # geometric floor, not an implementable online choice.
            oracle = np.empty_like(target)
            gram_condition = []
            for t in range(len(target)):
                basis = retained[t].T
                coeff, _, rank, singular = np.linalg.lstsq(basis, target[t], rcond=1e-12)
                if rank != k:
                    raise ValueError(f'non-full-rank retained outputs at token {t}')
                gram_condition.append(float(singular[0] / singular[-1]))
                oracle[t] = basis @ coeff
            best_subset_rms = None
            if k in (1, 2, 4):
                best_squared = 0.0
                for t in range(len(target)):
                    all_weighted = out[t] * weights[t, :, None]
                    minimum = float('inf')
                    for subset in itertools.combinations(range(8), k):
                        basis = all_weighted[list(subset)].T
                        coeff = np.linalg.lstsq(basis, target[t], rcond=1e-12)[0]
                        residual = target[t] - basis @ coeff
                        minimum = min(minimum, float(residual @ residual))
                    best_squared += minimum
                best_subset_rms = float(np.sqrt(best_squared / denom))
            result[str(k)][name] = {
                'score_sum_rms': rms(default),
                'score_renormalized_rms': rms(renorm),
                'train_scalar_rms': rms(global_scale * default),
                'token_scalar_oracle_rms': rms(token_scale[:, None] * default),
                'token_full_span_oracle_rms': rms(oracle),
                'best_subset_and_coefficients_oracle_rms': best_subset_rms,
                'max_basis_condition': max(gram_condition),
            }
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('capture', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    inputs = {split: load(args.capture, split) for split in ('train', 'held')}
    result = {
        'contract': 'layer-0 GGUF FP32 expert down outputs and scores, FP64 sums and orthogonal projections; oracle coefficients read the complete target',
        'source_sha256': sha(Path(__file__)),
        'model_sha256': json.loads((args.capture.parent / 'acquisition.json').read_text())['sha256'],
        'inputs': {split: {'count': len(a['ids']), 'sha256': hashes} for split, (a, hashes) in inputs.items()},
        'results': evaluate(*(cases(inputs[s][0]) for s in ('train', 'held'))),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['results'], indent=2))


if __name__ == '__main__':
    main()
