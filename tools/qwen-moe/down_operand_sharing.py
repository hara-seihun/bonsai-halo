#!/usr/bin/env python3
"""Test exact cross-expert sharing of real routed Q8_1 down operands."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

CAPTURE = Path('../../data/qwen-moe/route-capture')
QUANT = Path('../bonsai-hip/ggml/src/ggml-cuda/quantize.cu')
RUNTIME = Path('../../data/qwen-moe/runtime/current/runtime.json')
MODEL = Path('../../data/qwen-moe/acquisition.json')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze(path, ids_path):
    values = np.fromfile(path, dtype='<f4')
    ids = np.fromfile(ids_path, dtype='<i4').reshape(-1, 8)
    if values.size != ids.shape[0] * 8 * 512:
        raise ValueError('routed hidden and ID shapes disagree')
    values = values.reshape(-1, 8, 16, 32)
    if not np.isfinite(values).all():
        raise ValueError('nonfinite routed hidden')
    maxima = np.max(np.abs(values), axis=-1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        codes = np.rint(values / (maxima / np.float32(127))).astype(np.int8)
    codes[maxima[..., 0] == 0] = 0
    # A 32-code match is necessary for exact reuse of one Q8_1 operand.
    # The native block also holds FP16 scale and FP16 original group sum.
    same_position = 0
    any_position = 0
    max_equal_coordinates = 0
    matches_at_least_24 = 0
    pairs_at_position = 0
    positions_with_duplicate = 0
    for token in codes:
        seen = set()
        for expert in token:
            for block in expert:
                key = block.tobytes()
                any_position += key in seen
                seen.add(key)
        for coordinate in range(16):
            block_codes = token[:, coordinate, :]
            keys = [block.tobytes() for block in block_codes]
            positions_with_duplicate += len(set(keys)) < 8
            same_position += 8 - len(set(keys))
            for i in range(8):
                for j in range(i + 1, 8):
                    equal = int(np.count_nonzero(block_codes[i] == block_codes[j]))
                    max_equal_coordinates = max(max_equal_coordinates, equal)
                    matches_at_least_24 += equal >= 24
                    pairs_at_position += 1
    return {
        'hidden_sha256': sha(path), 'ids_sha256': sha(ids_path),
        'tokens': len(values), 'selected_experts_per_token': 8,
        'coordinates_per_expert': 16,
        'operand_blocks': len(values) * 128,
        'same_position_pairs': pairs_at_position,
        'same_position_equal_codes': same_position,
        'position_groups_with_duplicate_codes': positions_with_duplicate,
        'free_permutation_equal_codes': any_position,
        'max_equal_code_coordinates_in_pair': max_equal_coordinates,
        'pairs_with_at_least_24_equal_coordinates': matches_at_least_24,
        'zero_blocks': int(np.count_nonzero(maxima == 0)),
        'distinct_ids_per_token_min': int(np.min([len(set(row)) for row in ids])),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, default=CAPTURE)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = {
        'source_sha256': sha(Path(__file__)),
        'quantize_source_sha256': sha(QUANT),
        'capture_receipt_sha256': sha(args.capture / 'receipt.json'),
        'installed_runtime_receipt_sha256': sha(RUNTIME),
        'model_acquisition_sha256': sha(MODEL),
        'grammar': 'same native group-of-32 Q8_1 activation code among eight routed down operands at one token',
    }
    for split in ('train', 'held'):
        result[split] = analyze(
            args.capture / f'{split}.0.ffn_moe_swiglu-0.bin',
            args.capture / f'{split}.0.ffn_moe_topk-0.bin',
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps({split: result[split] for split in ('train', 'held')}, indent=2))


if __name__ == '__main__':
    main()
