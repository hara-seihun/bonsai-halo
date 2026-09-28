#!/usr/bin/env python3
"""Compare selected-logit softmax to captured native normalized MoE scores."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path('../../data/qwen-moe')
CAP = ROOT / 'route-capture'
OUT = ROOT / 'router-map'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def panel(split):
    p = OUT / f'{split}.logits.f32'
    logits = np.fromfile(p, '<f4').reshape(-1, 256)
    n = logits.shape[0]
    ids_path = CAP / f'{split}.0.ffn_moe_topk-0.bin'
    weights_path = CAP / f'{split}.0.ffn_moe_weights_norm-0.bin'
    down_path = CAP / f'{split}.0.ffn_moe_down-0.bin'
    ids = np.fromfile(ids_path, '<i4').reshape(n, 8)
    reference = np.fromfile(weights_path, '<f4').reshape(n, 8)
    down = np.memmap(down_path, dtype='<f4', mode='r', shape=(n, 8, 2048))
    sorted_ids = np.argsort(-logits, axis=1, kind='stable')[:, :8]
    assert np.array_equal(ids, sorted_ids), f'{split}: router IDs changed'
    assert np.isfinite(logits).all() and np.isfinite(reference).all()
    chosen = np.take_along_axis(logits, ids, axis=1)
    # Single eight-element normalization, not the native 256-element softmax
    # followed by SUM_ROWS/CLAMP/DIV. Each NumPy operator is rounded to f32.
    exps = np.exp(chosen - np.max(chosen, axis=1, keepdims=True)).astype(np.float32)
    candidate = exps / np.sum(exps, axis=1, keepdims=True, dtype=np.float32)
    delta = candidate.astype(np.float64) - reference.astype(np.float64)
    source = np.einsum('ne,ned->nd', reference.astype(np.float64), down.astype(np.float64))
    change = np.einsum('ne,ned->nd', delta, down.astype(np.float64))
    relative = np.linalg.norm(change) / np.linalg.norm(source)
    # Also calculate the dense two-normalization formula with sequential f32
    # operations; this is a diagnostic, not a GPU operation-order emulator.
    dense_exp = np.exp(logits - np.max(logits, axis=1, keepdims=True)).astype(np.float32)
    probabilities = dense_exp / np.sum(dense_exp, axis=1, keepdims=True, dtype=np.float32)
    picked = np.take_along_axis(probabilities, ids, axis=1)
    dense = picked / np.sum(picked, axis=1, keepdims=True, dtype=np.float32)
    return {
        'rows': n, 'ids_match': True,
        'changed_weight_bits_selected_only': int(np.count_nonzero(candidate.view('u4') != reference.view('u4'))),
        'changed_weight_bits_dense_numpy': int(np.count_nonzero(dense.view('u4') != reference.view('u4'))),
        'max_weight_abs_error_selected_only': float(np.max(np.abs(delta))),
        'mean_weight_l1_per_token': float(np.mean(np.sum(np.abs(delta), axis=1))),
        'weighted_down_relative_rms': float(relative),
        'max_token_weighted_down_relative_norm': float(np.max(np.linalg.norm(change, axis=1) / np.linalg.norm(source, axis=1))),
        'cutoff_margin_min': float(np.min(np.sort(logits, axis=1)[:, -8] - np.sort(logits, axis=1)[:, -9])),
        'sha256': {key: sha(path) for key, path in [('router_logits', p), ('selected_ids', ids_path), ('native_weights', weights_path), ('native_down', down_path)]},
    }


def main():
    result = {
        'contract': 'Layer-0 real prompt producer, selected eight-score FP32 normalization vs callback-cut native reference; identical selected IDs checked. NumPy f32 exponential and sum are not native GPU emulation. Output perturbation is local FP64 weighted sum over captured down vectors, not language quality.',
        'selected_only_exp_per_token': 8, 'native_two_stage_exp_per_token': 256,
        'splits': {split: panel(split) for split in ('train', 'held')},
        'selected_runtime': 'f60e4fbfabe735b15b4c74bfb1c9392f07329d2b',
        'model_sha256': 'ac0e2c1189e055faa36eff361580e79c5bd6f8e76bffb4ce547f167d53e31a61',
        'sha256': {str(p): sha(p) for p in [Path(__file__), Path(__file__).with_name('capture_router_logits.cpp'),
            OUT/'capture-router-logits', OUT/'train.wrapper.log', OUT/'held.wrapper.log',
            ROOT/'runtime/current/bin/libllama.so.0.2.0', ROOT/'runtime/current/bin/libggml-hip.so.0.21.0',
            CAP/'train.txt', CAP/'held.txt', CAP/'train.tokens', CAP/'held.tokens']},
    }
    (OUT / 'receipt.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps(result['splits'], indent=2))


if __name__ == '__main__':
    main()
