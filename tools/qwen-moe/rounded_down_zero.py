#!/usr/bin/env python3
"""Census rounded-zero down-operand groups on actual routed Qwen producers."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

CAPTURE = Path('../../data/qwen-moe/all-down-zero')
TRAFFIC = Path('../../data/qwen-moe/traffic.json')
QUANT = Path('../bonsai-hip/ggml/src/ggml-cuda/quantize.cu')
DOT = Path('../bonsai-hip/ggml/src/ggml-cuda/vecdotq.cuh')
MMQ = Path('../bonsai-hip/ggml/src/ggml-cuda/mmq.cuh')
RUNTIME = Path('../../data/qwen-moe/runtime/current/runtime.json')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def quant_codes(x):
    maximum = np.max(np.abs(x), axis=-1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        # Native quantize_mmq_q8_1: d_inv = 127.0f / amax; q=roundf(x*d_inv).
        inverse = np.float32(127) / maximum
        scaled = np.float32(x * inverse)
        q = np.copysign(np.floor(np.abs(scaled) + np.float32(.5)), scaled)
    q = np.where(maximum == 0, 0, q).astype(np.int8)
    assert np.max(np.abs(q.astype(np.int16))) <= 127
    return q


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, default=CAPTURE)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    traffic = json.loads(TRAFFIC.read_text())
    bank = {int(k): next(t for t in v['tensors'] if 'ffn_down_exps.weight' in t['name'])
            for k, v in traffic['layers'].items()}
    assert sorted(bank) == list(range(40))
    receipt = json.loads((args.capture / 'receipt.json').read_text())
    result = {'contract': 'Observed actual callback FP32 post-SwiGLU, CPU float32 emulation of native MMQ Q8_1 roundf(x*(127/amax)) code only. Four-code zero omits an integer dot term but NOT Q5_K affine minimum (which uses FP16-rounded sum of all 32 original floats), nor necessarily any physical dense weight byte. Generous free independent four-coordinate image-byte fraction is NOT an executable exact-map speedup, and callback graph may differ from no-cut native producer.',
              'domain': 'two disjoint 64-token selected-GGUF prompts; eight routed slots/token at each of forty layers',
              'source_sha256': sha(Path(__file__)),
              'reference_sha256': {str(p): sha(p) for p in (QUANT, DOT, MMQ, RUNTIME, TRAFFIC, args.capture / 'receipt.json')},
              'model_sha256': json.loads((args.capture.parent / 'all-producers' / 'receipt.json').read_text())['model_sha256'],
              'one_read_weight_bytes_per_token': traffic['one_token_weight_stream_bytes'],
              'splits': {}}
    for split in ('train', 'held'):
        layers, hashes = [], {}
        for index in range(40):
            p = args.capture / f'{split}.layer-{index}.ffn_moe_swiglu.f32'
            assert sha(p) == receipt['capture_sha256'][p.name]
            hashes[p.name] = sha(p)
            x = np.fromfile(p, dtype='<f4').reshape(64, 8, 16, 32)
            assert np.isfinite(x).all()
            q = quant_codes(x)
            groups4 = (q.reshape(-1, 4) == 0).all(axis=-1)
            original4 = (x.reshape(-1, 4) == 0).all(axis=-1)
            assert np.all(~original4 | groups4)
            groups8 = (q.reshape(-1, 8) == 0).all(axis=-1)
            groups16 = (q.reshape(-1, 16) == 0).all(axis=-1)
            groups32 = (q == 0).all(axis=-1)
            assert not np.any(groups32) # any finite nonzero block retains its maximum
            per_expert = bank[index]['bytes'] // 256
            # Counts are across 64 tokens and eight routes; one expert has 128 four-tuples.
            ideal = int(groups4.sum()) * per_expert / (64 * 128)
            layers.append({'layer': index, 'type': bank[index]['type'],
                           'zero_codes': int((q == 0).sum()),
                           'zero_original_four_groups': int(original4.sum()),
                           'zero_code_groups': {'4': int(groups4.sum()), '8': int(groups8.sum()),
                                                '16': int(groups16.sum()), '32': int(groups32.sum())},
                           'code_only_four_groups': int((groups4 & ~original4).sum()),
                           'ideal_free_four_tuple_bytes_per_token': ideal})
        totals = {key: sum(row['zero_code_groups'][key] for row in layers) for key in ('4', '8', '16', '32')}
        ideal = sum(row['ideal_free_four_tuple_bytes_per_token'] for row in layers)
        result['splits'][split] = {'layers': layers, 'inputs_sha256': hashes,
                                   'zero_code_groups': totals,
                                   'zero_codes': sum(row['zero_codes'] for row in layers),
                                   'code_only_four_groups': sum(row['code_only_four_groups'] for row in layers),
                                   'ideal_free_four_tuple_bytes_per_token': ideal,
                                   'ideal_fraction_complete_one_read': ideal / traffic['one_token_weight_stream_bytes'],
                                   'ideal_percent_complete_one_read': 100 * ideal / traffic['one_token_weight_stream_bytes']}
        print(split, totals, 'code-only-4', result['splits'][split]['code_only_four_groups'],
              'free %', result['splits'][split]['ideal_percent_complete_one_read'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')


if __name__ == '__main__':
    main()
