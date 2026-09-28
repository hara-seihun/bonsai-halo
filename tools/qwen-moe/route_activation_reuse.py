#!/usr/bin/env python3
"""Measure exact packed activation reuse on actual layer-0 Qwen routed inputs."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

CAPTURE = Path('../../data/qwen-moe/route-capture')
QUANT = Path('../bonsai-hip/ggml/src/ggml-cuda/quantize.cu')
RUNTIME = Path('../../data/qwen-moe/runtime/current/runtime.json')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(capture, split):
    input_path = capture / f'{split}.0.attn_post_norm-0.bin'
    route_path = capture / f'{split}.0.ffn_moe_topk-0.bin'
    inputs = np.fromfile(input_path, dtype='<f4').reshape(-1, 64, 32)
    routes = np.fromfile(route_path, dtype='<i4').reshape(-1, 8)
    if len(inputs) != len(routes) or not np.isfinite(inputs).all():
        raise ValueError(f'{split}: producer/route mismatch or nonfinite input')
    if any(len(set(row)) != 8 for row in routes):
        raise ValueError(f'{split}: expected eight distinct experts')
    # The native quantize_q8_1 rule: one maximum per 32 FP32 inputs, d=amax/127,
    # roundf(x/d), plus a separately stored FP16 scale and FP16 input sum.
    maxima = np.max(np.abs(inputs), axis=-1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        scaled = inputs / (maxima / np.float32(127))
        codes = (np.copysign(np.floor(np.abs(scaled) + np.float32(0.5)), scaled)).astype(np.int8)
    codes[maxima[..., 0] == 0] = 0
    return codes, routes, {str(p): digest(p) for p in (input_path, route_path)}


def counts(codes, routes):
    n = len(codes)
    keys = [(k, block.tobytes()) for row in codes for k, block in enumerate(row)]
    routed = [(int(expert), k, block.tobytes())
              for row, ids in zip(codes, routes)
              for expert in ids for k, block in enumerate(row)]
    flat_ids = [int(i) for row in routes for i in row]
    expert_counts = Counter(flat_ids)
    return {
        'tokens': n, 'routed_assignments': n * 8,
        'revisited_expert_assignments': sum(v - 1 for v in expert_counts.values()),
        'distinct_experts': len(expert_counts),
        'producer_blocks': n * 64,
        'duplicate_codes_same_K': len(keys) - len(set(keys)),
        'duplicate_codes_any_K': len(keys) - len({code for _, code in keys}),
        'duplicate_full_code_rows': n - len({row.tobytes() for row in codes}),
        'routed_block_uses': len(routed),
        'duplicate_routed_expert_K_codes': len(routed) - len(set(routed)),
        'zero_code_blocks': sum(not any(code) for _, code in keys),
        'raw_q8_1_operand_bytes_per_token': 64 * 36,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--capture', type=Path, default=CAPTURE)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    data = {split: load(args.capture, split) for split in ('train', 'held')}
    codes = np.concatenate([data[s][0] for s in ('train', 'held')])
    routes = np.concatenate([data[s][1] for s in ('train', 'held')])
    result = {
        'grammar': 'exact memoization of same Q8_1 32-code producer block at fixed K and expert; free lookup and cache; gate/up share the same producer per token already',
        'source_sha256': digest(Path(__file__)),
        'quantize_source_sha256': digest(QUANT),
        'capture_receipt_sha256': digest(args.capture / 'receipt.json'),
        'installed_runtime_receipt_sha256': digest(RUNTIME),
        'splits': {s: {'input_sha256': data[s][2], **counts(data[s][0], data[s][1])} for s in data},
        'combined': counts(codes, routes),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + '\n')
    print(json.dumps({s: result['splits'][s] for s in data} | {'combined': result['combined']}, indent=2))


if __name__ == '__main__':
    main()
