#!/usr/bin/env python3
"""Best smooth linear score-carrier tangent approximation on actual Qwen routes."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

BASE = Path('../../data/qwen-moe/all-producers')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def tangent_basis():
    # Orthonormal Helmert basis for {delta in R^8: sum(delta)=0}.
    h = np.zeros((8, 7), dtype=np.float64)
    for j in range(1, 8):
        h[:j, j - 1] = 1 / np.sqrt(j * (j + 1))
        h[j, j - 1] = -j / np.sqrt(j * (j + 1))
    assert np.allclose(h.T @ h, np.eye(7), atol=1e-15)
    assert np.max(np.abs(h.sum(axis=0))) < 1e-15
    return h


def summarize(rows):
    a = np.asarray(rows, dtype=np.float64)
    return {'min': float(a.min()), 'median': float(np.median(a)),
            'p95': float(np.quantile(a, .95)), 'max': float(a.max())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    receipt = BASE / 'receipt.json'
    capture = json.loads(receipt.read_text())
    h = tangent_basis()
    output = {'contract': 'Fixed eight GGUF down vectors per real layer/token; ideal-real normalized score perturbations in the Euclidean simplex tangent. Best output-aware rank-r linear score carrier, unit isotropic tangent perturbation; no paid image, native FP32 equivalence, full-model NLL or timing.',
              'source_sha256': digest(Path(__file__)), 'capture_receipt_sha256': digest(receipt),
              'model_sha256': capture['model_sha256'], 'splits': {}}
    for split in ('train', 'held'):
        file_hashes = {}
        spectrum, reference_norms = [], []
        for layer in range(40):
            files = [BASE / f'{split}.layer-{layer}.ffn_moe_{s}.f32' for s in ('weights_norm', 'down')]
            for p in files:
                assert digest(p) == capture['splits'][split]['files_sha256'][p.name]
                file_hashes[p.name] = digest(p)
            scores = np.memmap(files[0], dtype='<f4', mode='r', shape=(64, 8))
            down = np.memmap(files[1], dtype='<f4', mode='r', shape=(64, 8, 2048))
            assert np.all(np.isfinite(scores)) and np.all(scores > 0) and np.all(np.isfinite(down))
            # The Gram is seven-by-seven, never an 2048x2048 decomposition.
            basis_outputs = np.einsum('ie,nid->ned', h, down.astype(np.float64), optimize=True)
            gram = basis_outputs @ basis_outputs.transpose(0, 2, 1)
            eigen = np.linalg.eigvalsh(gram)[:, ::-1]
            assert np.all(eigen[:, -1] > 0)
            spectrum.extend(eigen.tolist())
            y = np.einsum('ni,nid->nd', scores.astype(np.float64), down.astype(np.float64), optimize=True)
            reference_norms.extend(np.linalg.norm(y, axis=1).tolist())
        eigen = np.asarray(spectrum)
        norm = np.asarray(reference_norms)
        total = eigen.sum(axis=1)
        # For delta drawn uniformly on the unit sphere of the 7-D tangent,
        # E ||D delta||^2 = trace(D^T D)/7. Eckart-Young gives tail eigen sum/7.
        panels = {}
        for rank in range(7):
            tail = eigen[:, rank:].sum(axis=1)
            relative = np.sqrt(tail / total)
            # At a hypothetical tangent score displacement of L2=.01, this is
            # the expected squared error divided by the actual scored-sum norm.
            local = .01 * np.sqrt(tail / 7) / norm
            panels[str(rank)] = {
                'relative_to_full_tangent_rms': summarize(relative),
                'pooled_relative_tangent_rms': float(np.sqrt(tail.sum() / total.sum())),
                'unit_percent_score_displacement_relative_to_actual_sum': summarize(local),
                'pooled_unit_percent_score_displacement_relative_to_actual_sum': float(np.sqrt(np.sum(tail) / 7 * .01**2 / np.sum(norm**2))),
            }
        output['splits'][split] = {'observations': len(eigen), 'file_sha256': file_hashes,
                                   'condition_number': summarize(np.sqrt(eigen[:, 0] / eigen[:, -1])),
                                   'panels': panels}
        print(split, 'condition', output['splits'][split]['condition_number'],
              'r3', panels['3'], 'r6', panels['6'], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + '\n')
    print(args.output)


if __name__ == '__main__':
    main()
