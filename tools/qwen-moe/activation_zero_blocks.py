#!/usr/bin/env python3
"""Count exact native Q8_1 operand-skipping opportunities on captured MoE inputs."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


DEFAULT_CAPTURE = Path('../../data/qwen-moe/route-capture')
NATIVE_QUANT = Path('../bonsai-hip/ggml/src/ggml-cuda/quantize.cu')
NATIVE_DOT = Path('../bonsai-hip/ggml/src/ggml-cuda/vecdotq.cuh')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def measure(path, width):
    values = np.fromfile(path, dtype='<f4')
    if values.size % width:
        raise ValueError(f'{path}: not a multiple of {width} floats')
    rows = values.reshape(-1, width)
    if not np.isfinite(rows).all():
        raise ValueError(f'{path}: nonfinite input')
    groups = rows.reshape(-1, 32)
    amax = np.max(np.abs(groups), axis=1)
    # quantize_q8_1 in quantize.cu uses amax/127, roundf(x/d), and the
    # original float sum in ds.y. No part of this study changes the operand.
    with np.errstate(divide='ignore', invalid='ignore'):
        q = np.rint(groups / (amax[:, None] / np.float32(127))).astype(np.int16)
    q[amax == 0] = 0
    if np.any((amax != 0) & (np.max(np.abs(q), axis=1) < 126)):
        raise AssertionError('a nonzero native block lost its maximum')
    result = {
        'rows': len(rows), 'width': width, 'sha256': digest(path),
        'zero_float_elements': int(np.count_nonzero(rows == 0)),
        'elements': int(values.size),
        'blocks_32': int(len(groups)),
        'zero_input_blocks_32': int(np.count_nonzero(amax == 0)),
        'nonzero_q8_blocks_32': int(np.count_nonzero(amax != 0)),
        'q8_zero_elements': int(np.count_nonzero(q == 0)),
        'subgroups': {},
    }
    for n in (4, 8, 16, 32):
        sub = q.reshape(-1, n)
        original_sub = values.reshape(-1, n)
        count = int(np.count_nonzero(np.all(sub == 0, axis=1)))
        original_count = int(np.count_nonzero(np.all(original_sub == 0, axis=1)))
        result['subgroups'][str(n)] = {
            'all_zero_q8': count, 'all_zero_original': original_count,
            'total': len(sub),
        }
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--capture', type=Path, default=DEFAULT_CAPTURE)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    files = {'quantize.cu': NATIVE_QUANT, 'vecdotq.cuh': NATIVE_DOT}
    receipt = {
        'observation': 'layer-0 installed Qwen GGUF post-attention norm and routed post-SwiGLU, before Q8_1 quantization',
        'source_sha256': digest(Path(__file__)),
        'native_source_sha256': {name: digest(path) for name, path in files.items()},
        'capture_receipt_sha256': digest(args.capture / 'receipt.json'),
        'installed_runtime_sha256': digest(Path('../../data/qwen-moe/runtime/current/runtime.json')),
        'model_acquisition_sha256': digest(Path('../../data/qwen-moe/acquisition.json')),
        'arrays': {},
    }
    for split in ('train', 'held'):
        for name, width in (('attn_post_norm', 2048), ('ffn_moe_swiglu', 512)):
            path = args.capture / f'{split}.0.{name}-0.bin'
            receipt['arrays'][f'{split}.{name}'] = measure(path, width)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    print(json.dumps(receipt['arrays'], indent=2))


if __name__ == '__main__':
    main()
