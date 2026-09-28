#!/usr/bin/env python3
"""Identify the coordinate responsible for the first saved gathered row-4 key mismatch.

Uses only the retained post-step FP16 key cells; fitting a rotation is an
observed-boundary diagnosis, not a capture of the unrotated K producer.
"""
import argparse
import hashlib
import json
import mmap
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from gdn_state_diff import regions


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def key(blob, lookup, row, layer=0, token=9):
    start, end, n = lookup[f'kv/{row}/k/{layer}']
    stride = (end-start-12)//n
    assert n == 10 and stride == 1024 and blob[start:start+4] == (1).to_bytes(4, 'little')
    return np.frombuffer(blob[start+12+token*stride:start+12+(token+1)*stride], '<f2').astype(np.float64)


def analyze(root, config):
    paths = [root / f'{arm}-gather.state' for arm in ('normal', 'swap')]
    cfg = json.loads(config.read_text())['text_config']
    rope = cfg['rope_parameters']
    assert rope['mrope_interleaved'] and rope['mrope_section'] == [11, 11, 10]
    assert cfg['head_dim'] == 256 and cfg['num_key_value_heads'] == 2
    assert rope['partial_rotary_factor'] == 0.25
    theta = rope['rope_theta']
    with paths[0].open('rb') as af, paths[1].open('rb') as bf, \
            mmap.mmap(af.fileno(), 0, access=mmap.ACCESS_READ) as a, \
            mmap.mmap(bf.fileno(), 0, access=mmap.ACCESS_READ) as b:
        la = {name: (start, end, info) for name, start, end, info in regions(a)}
        lb = {name: (start, end, info) for name, start, end, info in regions(b)}
        assert list(la) == list(lb)
        A, B = key(a, la, 4), key(b, lb, 4)
        # IMRoPE couples (i, i+32) in each 256-dimensional head and chooses
        # temporal position for i%3==0 within the first 32 rotary pairs.
        pairs = np.array([(256*h+i, 256*h+i+32) for h in range(2) for i in range(32)], dtype=np.int64)
        temporal = np.array([i%3 == 0 and i < 33 for h in range(2) for i in range(32)])
        changed = A != B
        bad = np.flatnonzero(changed).tolist()
        assert set(bad).issubset(set(pairs[temporal].flatten()))
        X = A[pairs]; Y = B[pairs]
        dot = (X*Y).sum(axis=1)
        cross = X[:,0]*Y[:,1]-X[:,1]*Y[:,0]
        angle = np.arctan2(cross, dot)
        norm = np.linalg.norm(X, axis=1)
        length_error = abs(norm - np.linalg.norm(Y, axis=1))
        freq = theta ** (-np.array([i for h in range(2) for i in range(32)])/32.)
        candidates = []
        for step in range(-32,33):
            residual = np.angle(np.exp(1j*(angle[temporal] - step*freq[temporal])))
            candidates.append({'position_delta': step,
                               'weighted_angular_rms': float(np.sqrt(np.sum((residual*norm[temporal])**2)/np.sum(norm[temporal]**2))),
                               'max_angular_error': float(np.max(abs(residual)))})
        candidates.sort(key=lambda x:x['weighted_angular_rms'])
        best = candidates[0]['position_delta']
        predicted = A.copy()
        for h in range(2):
            for i in range(32):
                if i % 3:
                    continue
                j = h*256+i
                d = best * theta**(-i/32.)
                co, si = np.cos(d), np.sin(d)
                predicted[j] = A[j]*co-A[j+32]*si
                predicted[j+32] = A[j]*si+A[j+32]*co
        # The input here is already rounded FP16, so this is only a tolerance
        # check; an exact FP16 rerotation is not expected after double rounding.
        rounded = predicted.astype('<f2').astype(np.float64)
        return {
            'contract': 'Post-step saved FP16 K row4 layer3 last cell; two gathered no-callback executions with identical logical input and positions (9); IMRoPE temporal-sector rotation fitted on these rounded outputs, not a causal capture of pre-RoPE producer.',
            'model_config_sha256': sha(config), 'state_sha256': {p.name:sha(p) for p in paths},
            'prior_four_arm_receipt_sha256': sha(root/'receipt.json'),
            'parser_sha256': sha(Path(__file__).with_name('gdn_state_diff.py')),
            'source_sha256': sha(Path(__file__)),
            'changed_half_count': len(bad), 'changed_half_indices': bad,
            'changed_outside_temporal_sector': int(sum(changed[j] for j in range(512) if j not in set(pairs[temporal].flatten()))),
            'temporal_pairs': int(temporal.sum()),
            'unchanged_non_temporal_halves': int(512-2*temporal.sum()),
            'best_integer_position_deltas': candidates[:5],
            'best_delta': best,
            'maximum_temporal_pair_length_difference': float(np.max(length_error[temporal])),
            'mean_temporal_pair_length_difference': float(np.mean(length_error[temporal])),
            'rerotation_fp16_equal_halves': int(np.count_nonzero(rounded == B)),
            'rerotation_max_absolute_error': float(np.max(abs(rounded-B))),
            'swapped_row1_zero_half_count': int(np.count_nonzero(key(b, lb, 1)==0)),
        }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=Path('../../data/qwen-moe/gdn-permutation-equiv'))
    ap.add_argument('--config', type=Path, default=Path('../../data/qwen-moe/official/config.json'))
    ap.add_argument('--output', required=True, type=Path)
    args = ap.parse_args()
    result = analyze(args.root, args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('changed_half_count','best_integer_position_deltas','maximum_temporal_pair_length_difference','rerotation_fp16_equal_halves','rerotation_max_absolute_error','swapped_row1_zero_half_count')}, indent=2))

if __name__ == '__main__':
    main()
