#!/usr/bin/env python3
"""Census of zero producer blocks in the selected Qwen routed gate/up operand."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def measure(capture, traffic):
    tensors = {}
    for layer in range(40):
        names = {kind: next(t for t in traffic['layers'][str(layer)]['tensors']
                            if f'ffn_{kind}_exps.weight' in t['name'])
                 for kind in ('gate', 'up')}
        assert all(t['bytes'] % 256 == 0 for t in names.values())
        tensors[layer] = names
    result = {'inputs_sha256': {}, 'splits': {}, 'contract':
              'native Q8_1 complete 32-float zero operand, and free independently '
              'addressed original-zero 4/8/16/32-coordinate weight slices; '
              'two disjoint 64-token callback prompt prefixes, not native time'}
    for split in ('train', 'held'):
        layers = []
        for layer in range(40):
            path = capture / f'{split}.layer-{layer}.attn_post_norm.f32'
            route = capture / f'{split}.layer-{layer}.ffn_moe_topk.i32'
            result['inputs_sha256'][str(path)] = sha256(path)
            result['inputs_sha256'][str(route)] = sha256(route)
            x = np.fromfile(path, dtype='<f4')
            ids = np.fromfile(route, dtype='<i4').reshape(64, 8)
            assert x.size == 64 * 2048 and np.isfinite(x).all()
            assert np.all((ids >= 0) & (ids < 256))
            assert all(len(set(row)) == 8 for row in ids)
            z = (x.reshape(64, 2048) == 0)
            groups = {str(n): int(z.reshape(64, 2048 // n, n).all(axis=2).sum())
                      for n in (4, 8, 16, 32)}
            bank = tensors[layer]
            selected_gate_up = 8 * sum(t['bytes'] // 256 for t in bank.values())
            layers.append({'layer': layer, 'gate_up_types': [t['type'] for t in bank.values()],
                           'gate_up_selected_bytes_per_token': selected_gate_up,
                           'zero_floats': int(z.sum()), 'zero_groups': groups,
                           'zero_q8_1_blocks': groups['32'],
                           'free_32_slice_bytes_per_token': selected_gate_up * groups['32'] / (64 * 64),
                           'free_4_slice_bytes_per_token': selected_gate_up * groups['4'] / (64 * 512)})
        total = sum(r['gate_up_selected_bytes_per_token'] for r in layers)
        free32 = sum(r['free_32_slice_bytes_per_token'] for r in layers)
        free4 = sum(r['free_4_slice_bytes_per_token'] for r in layers)
        denominator = traffic['one_token_weight_stream_bytes']
        result['splits'][split] = {'layers': layers, 'tokens': 64,
            'producer_blocks': 40 * 64 * 64,
            'routed_block_uses': 40 * 64 * 64 * 8,
            'zero_floats': sum(r['zero_floats'] for r in layers),
            'zero_groups': {str(n): sum(r['zero_groups'][str(n)] for r in layers)
                            for n in (4, 8, 16, 32)},
            'gate_up_selected_bytes_per_token': total,
            'free_32_slice_bytes_per_token': free32,
            'free_4_slice_bytes_per_token': free4,
            'free_32_slice_percent_complete_stream': 100 * free32 / denominator,
            'free_4_slice_percent_complete_stream': 100 * free4 / denominator}
    result['complete_one_read_weight_bytes_per_token'] = traffic['one_token_weight_stream_bytes']
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture', type=Path, default=Path('../../data/qwen-moe/all-producers'))
    p.add_argument('--traffic', type=Path, default=Path('../../data/qwen-moe/traffic.json'))
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = measure(args.capture, json.loads(args.traffic.read_text()))
    sources = (Path(__file__), args.traffic, args.capture / 'receipt.json',
               Path('../../data/qwen-moe/acquisition.json'),
               Path('../../data/qwen-moe/runtime/current/runtime.json'),
               Path('../bonsai-hip/ggml/src/ggml-cuda/quantize.cu'))
    result['sources_sha256'] = {str(s): sha256(s) for s in sources}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    for split, row in result['splits'].items():
        print(split, row['zero_groups'], row['free_4_slice_percent_complete_stream'])


if __name__ == '__main__':
    main()
