#!/usr/bin/env python3
"""Construct shared pre-RoPE K witnesses for both rounded row-4 saved keys.

This tests an ideal real rotation followed by FP16 rounding, not native CUDA
FP32 arithmetic or the actual unsaved producer. Each of the 22 temporal pairs
has four observed FP16 words (two per execution). Solve the four rounding-bin
constraints for one common two-coordinate pre-rotation vector.
"""
import argparse
import hashlib
import json
import mmap
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import linprog

sys.path.insert(0, str(Path(__file__).parent))
from gdn_k_origin import key
from gdn_state_diff import regions

ROOT = Path('../../data/qwen-moe/gdn-permutation-equiv')
CONFIG = Path('../../data/qwen-moe/official/config.json')


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def rounding_bin(value):
    h = np.asarray(value, dtype=np.float16)
    lower = np.nextafter(h, np.float16('-inf')).astype(np.float64)
    upper = np.nextafter(h, np.float16('inf')).astype(np.float64)
    v = h.astype(np.float64)
    assert np.isfinite(lower).all() and np.isfinite(upper).all()
    return (lower + v) / 2, (upper + v) / 2


def witness(normal, swapped, angle):
    """Maximize minimum fractional rounding-bin slack of the four observations."""
    co, si = np.cos(angle), np.sin(angle)
    R = np.array([[co, -si], [si, co]])
    B, A = np.asarray(swapped), np.asarray(normal)
    M = np.concatenate([np.eye(2), R])
    values = np.concatenate([B, A])
    lo, hi = rounding_bin(values)
    widths = hi - lo
    # Solve for delta to the observed position-zero center, avoiding absolute
    # magnitudes in HiGHS tolerances. The third variable is the fractional
    # distance to *every* rounding-bin endpoint, scaled by its local width.
    constraints = np.concatenate([np.column_stack([M, widths]),
                                  np.column_stack([-M, widths])])
    rhs = np.concatenate([hi - M @ B, M @ B - lo])
    fit = linprog([0, 0, -1], A_ub=constraints, b_ub=rhs,
                  bounds=[(None, None), (None, None), (0, 0.49)], method='highs',
                  options={'primal_feasibility_tolerance': 1e-9,
                           'dual_feasibility_tolerance': 1e-9})
    if not fit.success:
        return {'feasible': False, 'solver': fit.message}
    x = B + fit.x[:2]
    observed = M @ x
    slack = np.minimum(observed - lo, hi - observed) / widths
    # Independent end-to-end observed-word check; the LP itself is not enough.
    reconstructed = observed.astype(np.float16).view(np.uint16)
    target = values.astype(np.float16).view(np.uint16)
    assert np.array_equal(reconstructed, target), (reconstructed, target)
    assert np.min(slack) > 0, slack
    # A separate approximate FP32 check, *not* an emulator of native rope_yarn
    # or a promise about contraction/rounding on the GPU.
    a32 = np.float32(angle)
    c32, s32 = np.cos(a32), np.sin(a32)
    x0, x1 = np.float32(x[0]), np.float32(x[1])
    fp32 = np.array([np.float32(x0*c32 - x1*s32),
                     np.float32(x0*s32 + x1*c32)])
    fp32_equal = np.array_equal(fp32.astype(np.float16).view(np.uint16), target[2:])
    return {'feasible': True, 'input': x.tolist(),
            'simple_fp32_position9_matches': bool(fp32_equal),
            'rounded_bits': [int(v) for v in reconstructed],
            'minimum_fractional_bin_margin': float(np.min(slack)),
            'minimum_absolute_bin_margin': float(np.min(np.minimum(observed-lo, hi-observed))),
            'position9_ideal': observed[2:].tolist()}


def analyze(root, config):
    cfg = json.loads(config.read_text())['text_config']
    rope = cfg['rope_parameters']
    assert rope['mrope_interleaved'] and rope['mrope_section'] == [11, 11, 10]
    assert cfg['head_dim'] == 256 and cfg['num_key_value_heads'] == 2
    theta = rope['rope_theta']
    paths = [root / f'{arm}-gather.state' for arm in ('normal', 'swap')]
    outputs = []
    for path in paths:
        with path.open('rb') as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as blob:
            lookup = {name: (start, end, info) for name, start, end, info in regions(blob)}
            outputs.append(key(blob, lookup, 4).copy())
    A, B = outputs
    pairs = []
    reconstructed_A, reconstructed_B = np.zeros(512, dtype=np.float64), np.zeros(512, dtype=np.float64)
    temporal_indices = set()
    for head in range(2):
        for i in range(32):
            idx = [256*head+i, 256*head+i+32]
            if i % 3 != 0:
                # Non-temporal sectors are unchanged, hence shared post-rotation
                # observations admit the same preimage under the same positions.
                assert np.array_equal(A[idx], B[idx])
                continue
            temporal_indices.update(idx)
            angle = 9 * theta ** (-i / 32)
            result = witness(A[idx], B[idx], angle)
            result.update(head=head, pair=i, indices=idx, angle=angle,
                          normal_bits=[int(v) for v in A[idx].astype('<f2').view('<u2')],
                          swapped_bits=[int(v) for v in B[idx].astype('<f2').view('<u2')])
            assert result['feasible'], result
            pairs.append(result)
            reconstructed_A[idx] = result['position9_ideal']
            reconstructed_B[idx] = result['input']
    assert np.array_equal(A.astype('<f2').view('<u2')[list(sorted(temporal_indices))],
                          reconstructed_A.astype('<f2').view('<u2')[list(sorted(temporal_indices))])
    assert np.array_equal(B.astype('<f2').view('<u2')[list(sorted(temporal_indices))],
                          reconstructed_B.astype('<f2').view('<u2')[list(sorted(temporal_indices))])
    assert np.array_equal(A[[j for j in range(512) if j not in temporal_indices]],
                          B[[j for j in range(512) if j not in temporal_indices]])
    return {
        'contract': 'For the saved layer-3 row-4 newest key in normal/swapped gathered 31+1 graphs: existential common real pre-RoPE K, ideal real IMRoPE with temporal positions 9/0 and identical other positions, independent nearest-FP16 output rounding. No native FP32/graph or actual producer identity assertion.',
        'config_sha256': sha(config),
        'state_sha256': {p.name: sha(p) for p in paths},
        'source_sha256': sha(Path(__file__)),
        'parser_sha256': sha(Path(__file__).with_name('gdn_state_diff.py')),
        'prior_angle_receipt_sha256': sha(root/'k-origin-receipt.json'),
        'temporal_pairs': len(pairs), 'observed_word_count': 512 * 2,
        'minimum_fractional_bin_margin': min(p['minimum_fractional_bin_margin'] for p in pairs),
        'minimum_absolute_bin_margin': min(p['minimum_absolute_bin_margin'] for p in pairs),
        'simple_fp32_matching_pairs': sum(p['simple_fp32_position9_matches'] for p in pairs),
        'pairs': pairs,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=ROOT)
    ap.add_argument('--config', type=Path, default=CONFIG)
    ap.add_argument('--output', type=Path, default=ROOT/'row4-position-receipt.json')
    args = ap.parse_args()
    result = analyze(args.root, args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('temporal_pairs', 'observed_word_count', 'minimum_fractional_bin_margin', 'minimum_absolute_bin_margin', 'simple_fp32_matching_pairs')}, indent=2))

if __name__ == '__main__':
    main()
