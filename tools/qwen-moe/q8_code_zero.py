#!/usr/bin/env python3
"""Census MMQ Q8_1 zero codes on real all-layer routed gate/up producers."""
import hashlib
import json
from pathlib import Path

import numpy as np

CAPTURE = Path('../../data/qwen-moe/all-producers')
DATA = Path('../../data/qwen-moe/q8-code-zero')
ROOT = Path(__file__).resolve().parents[2]
QUANT = Path('../bonsai-hip/ggml/src/ggml-cuda/quantize.cu')
RUNTIME = Path('../../data/qwen-moe/runtime/current/runtime.json')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def panel(split, expected, traffic):
    layers = []
    files = {}
    for layer in range(40):
        path = CAPTURE / f'{split}.layer-{layer}.attn_post_norm.f32'
        files[path.name] = sha(path)
        assert files[path.name] == expected[path.name]
        x = np.fromfile(path, dtype='<f4').reshape(64, 64, 32)
        assert np.isfinite(x).all()
        maximum = np.max(np.abs(x), axis=-1, keepdims=True)
        assert (maximum > 0).all()
        with np.errstate(over='raise', invalid='raise'):
            # CUDA quantize_mmq_q8_1 uses d_inv = 127 / amax, then roundf(x*d_inv).
            # Half-away-from-zero agrees with roundf; inspect exact zero predicates,
            # which are separated from ties for this observed capture.
            scaled = x * (np.float32(127) / maximum)
            code = np.copysign(np.floor(np.abs(scaled) + np.float32(.5)), scaled).astype(np.int8)
        assert (np.max(np.abs(code), axis=-1) >= 126).all()
        groups = {}
        for width in (1, 2, 4, 8, 16, 32):
            c = code.reshape(64, 2048 // width, width)
            z = ~np.any(c, axis=-1)
            groups[str(width)] = {'zero': int(z.sum()), 'total': int(z.size),
                                  'zero_token0': int(z[0].sum()),
                                  'zero_tokens_1_to_63': int(z[1:].sum()),
                                  'tokens_with_any': int(z.any(axis=1).sum())}
        # Ideal independent-addressed Q4_K code traffic: 4.5 bits/input/output,
        # two 512-output banks, eight selected experts.  This grants skipping
        # individual zero-code coordinates without reading Q4 superblock scales.
        zero = groups['1']['zero']
        four = groups['4']['zero']
        layers.append({'layer': layer, 'groups': groups,
                       'free_coordinate_gate_up_bytes_per_token': zero * 8 * 512 * 2 * 4.5 / 8 / 64,
                       'free_four_group_gate_up_bytes_per_token': four * 4 * 8 * 512 * 2 * 4.5 / 8 / 64})
    totals = {str(w): {key: sum(row['groups'][str(w)][key] for row in layers)
                       for key in ('zero', 'total', 'zero_token0', 'zero_tokens_1_to_63')}
              for w in (1, 2, 4, 8, 16, 32)}
    bytes_four = sum(row['free_four_group_gate_up_bytes_per_token'] for row in layers)
    bytes_one = sum(row['free_coordinate_gate_up_bytes_per_token'] for row in layers)
    return {'layers': layers, 'totals': totals, 'input_sha256': files,
            'free_four_group_gate_up_bytes_per_token': bytes_four,
            'free_four_group_percent_complete_one_read': 100 * bytes_four / traffic,
            'free_coordinate_gate_up_bytes_per_token': bytes_one,
            'free_coordinate_percent_complete_one_read': 100 * bytes_one / traffic}


def main():
    expected = json.loads((Path('../../data/qwen-moe/all-down-zero') / 'receipt.json').read_text())['capture_sha256']
    traffic_path = Path('../../data/qwen-moe/traffic.json')
    traffic = json.loads(traffic_path.read_text())['one_token_weight_stream_bytes']
    result = {'contract': 'CPU exact Q8_1 integer-code zero predicates under native MMQ reciprocal quantizer, two 64-token actual GGUF post-attention producer splits and 40 layers. Ideal independently addressable gate/up Q4_K code-byte skip; no Q4_K scale omission, no native floating bit identity, no generation/TPS claim.',
              'model_sha256': json.loads((CAPTURE / 'receipt.json').read_text())['model_sha256'],
              'capture_receipt_sha256': sha(CAPTURE / 'receipt.json'),
              'traffic_sha256': sha(traffic_path), 'quantizer_sha256': sha(QUANT),
              'installed_runtime_sha256': sha(RUNTIME), 'source_sha256': sha(Path(__file__)),
              'full_one_read_weight_bytes_per_token': traffic,
              'splits': {s: panel(s, expected, traffic) for s in ('train', 'held')}}
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / 'receipt.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    for split, value in result['splits'].items():
        print(split, value['totals'], 'free_four_bytes/token', value['free_four_group_gate_up_bytes_per_token'],
              'free_one_bytes/token', value['free_coordinate_gate_up_bytes_per_token'])


if __name__ == '__main__':
    main()
