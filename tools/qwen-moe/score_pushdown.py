#!/usr/bin/env python3
"""Observed Qwen routed-down Q8_1 scale equivariance, on actual captured producers."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quantize(x):
    amax = np.max(np.abs(x), axis=-1)
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        inv = np.float32(127.0) / amax
        v = np.float32(x * inv[..., None])
    # CUDA roundf uses ties away from zero, unlike numpy.rint.
    q = np.where(amax[..., None] == 0, 0, np.copysign(np.floor(np.abs(v) + np.float32(.5)), v))
    assert np.isfinite(q).all() and np.max(np.abs(q)) <= 127
    with np.errstate(divide='ignore', invalid='ignore'):
        d = np.float32(1.0) / inv
    # DS4 Q5_K input stores the FP16 sum of the same 32 source floats.
    lane = np.float32(np.float32(np.float32(x[..., 0::4] + x[..., 1::4]) + x[..., 2::4]) + x[..., 3::4])
    for offset in (4, 2, 1):
        lane = np.float32(lane[..., :offset] + lane[..., offset:2 * offset])
    return q.astype(np.int8), np.where(amax == 0, 0, d).astype(np.float32), lane[..., 0]


def observe(root, split):
    totals = dict(blocks=0, nonzero_blocks=0, code_changed_blocks=0, code_changed_values=0,
                  scale_f16_changed=0, scale_f16_ideal_changed=0, scale_f32_changed=0, sum_f16_changed=0, sum_f16_ideal_changed=0, nonfinite_inputs=0, max_code_change=0)
    hashes = {}
    for layer in range(40):
        stem = root / f'{split}.layer-{layer}.'
        files = {name: Path(str(stem) + suffix) for name, suffix in (
            ('hidden', 'ffn_moe_swiglu.f32'), ('score', 'ffn_moe_weights_norm.f32'),
            ('ids', 'ffn_moe_topk.i32'))}
        hashes[str(layer)] = {name: sha(path) for name, path in files.items()}
        x = np.fromfile(files['hidden'], dtype='<f4').reshape(-1, 8, 512)
        score = np.fromfile(files['score'], dtype='<f4').reshape(-1, 8)
        ids = np.fromfile(files['ids'], dtype='<i4').reshape(-1, 8)
        assert x.shape[0] == 64 and score.shape == ids.shape == (64, 8)
        assert np.all((ids >= 0) & (ids < 256)) and np.all(score > 0) and np.isfinite(score).all()
        assert np.isfinite(x).all()
        x = x.reshape(64, 8, 16, 32)
        scaled = np.float32(x * score[:, :, None, None])
        q, d, total = quantize(x)
        q2, d2, total2 = quantize(scaled)
        delta = np.abs(q.astype(np.int16) - q2.astype(np.int16))
        totals['blocks'] += int(delta.shape[0] * delta.shape[1] * delta.shape[2])
        totals['nonzero_blocks'] += int(np.count_nonzero(np.max(np.abs(x), axis=-1)))
        totals['code_changed_blocks'] += int(np.count_nonzero(np.any(delta != 0, axis=-1)))
        totals['code_changed_values'] += int(np.count_nonzero(delta))
        totals['max_code_change'] = max(totals['max_code_change'], int(delta.max()))
        # The selected input stores FP16 d/sum *before* the score is available.
        # Also record the free oracle with unrounded metadata for comparison.
        early = np.float16(d2)
        ideal = np.float16(np.float32(d * score[:, :, None]))
        late = np.float16(np.float32(np.float16(d).astype(np.float32) * score[:, :, None]))
        totals['scale_f16_changed'] += int(np.count_nonzero(late.view(np.uint16) != early.view(np.uint16)))
        totals['scale_f16_ideal_changed'] += int(np.count_nonzero(ideal.view(np.uint16) != early.view(np.uint16)))
        totals['scale_f32_changed'] += int(np.count_nonzero(np.float32(d * score[:, :, None]).view(np.uint32) != d2.view(np.uint32)))
        sum_early = np.float16(total2)
        sum_ideal = np.float16(np.float32(total * score[:, :, None]))
        sum_late = np.float16(np.float32(np.float16(total).astype(np.float32) * score[:, :, None]))
        totals['sum_f16_changed'] += int(np.count_nonzero(sum_late.view(np.uint16) != sum_early.view(np.uint16)))
        totals['sum_f16_ideal_changed'] += int(np.count_nonzero(sum_ideal.view(np.uint16) != sum_early.view(np.uint16)))
        totals['nonfinite_inputs'] += int(np.count_nonzero(~np.isfinite(scaled)))
    return totals, hashes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--capture', type=Path, default=Path('../../data/qwen-moe/all-down-zero'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--native-source', type=Path, default=Path('../bonsai-hip/ggml/src/ggml-cuda'))
    args = parser.parse_args()
    train, train_hashes = observe(args.capture, 'train')
    held, held_hashes = observe(args.capture, 'held')
    result = {'domain': '40 layers x 64 tokens x 8 actual routed slots x 16 post-SwiGLU Q8_1 input blocks, each split',
              'grammar': 'CPU float32 reciprocal/max MMQ quantization with CUDA roundf ties-away rule; separately FP32-multiply positive router score before quantization, versus same code plus FP16(score * already-FP16 Q8_1 scale/sum); separate free oracle starts from unrounded FP32 metadata. Scores from callback capture; no native FP32/graph identity.',
              'train': train, 'held': held, 'inputs_sha256': {'train': train_hashes, 'held': held_hashes},
              'script_sha256': sha(Path(__file__)), 'capture_receipt_sha256': sha(args.capture / 'receipt.json'),
              'native_sha256': {name: sha(args.native_source / name) for name in ('quantize.cu', 'mmq.cuh')}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'train': train, 'held': held}, sort_keys=True))


if __name__ == '__main__':
    main()
