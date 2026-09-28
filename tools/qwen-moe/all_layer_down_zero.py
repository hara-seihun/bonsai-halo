#!/usr/bin/env python3
"""Bound native exact down-operand zero skipping on actual Qwen MoE layer captures."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


NAMES = ('attn_post_norm.f32', 'ffn_moe_topk.i32', 'ffn_moe_weights_norm.f32',
         'ffn_moe_down.f32')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def scan(capture, predecessor, layer0, traffic, source):
    bank = {int(k): next(t for t in v['tensors'] if 'ffn_down_exps.weight' in t['name'])
            for k, v in traffic['layers'].items()}
    assert sorted(bank) == list(range(40))
    assert all(t['bytes'] % 256 == 0 for t in bank.values())
    hashes = {}
    summaries = {}
    for split in ('train', 'held'):
        token_file = capture / f'{split}.tokens'
        old_tokens = predecessor / f'{split}.tokens'
        assert token_file.read_bytes() == old_tokens.read_bytes()
        hashes[token_file.name] = digest(token_file)
        assert token_file.stat().st_size == 64 * 4
        layers = []
        for layer in range(40):
            for name in NAMES:
                new = capture / f'{split}.layer-{layer}.{name}'
                old = predecessor / new.name
                assert new.stat().st_size == old.stat().st_size
                assert digest(new) == digest(old), f'previously captured tensor changed: {new}'
                hashes[new.name] = digest(new)
            path = capture / f'{split}.layer-{layer}.ffn_moe_swiglu.f32'
            assert path.stat().st_size == 64 * 8 * 512 * 4
            hashes[path.name] = digest(path)
            a = np.fromfile(path, dtype='<f4').reshape(64, 8, 512)
            assert np.isfinite(a).all()
            if layer == 0:
                original = np.fromfile(layer0 / f'{split}.0.ffn_moe_swiglu-0.bin', dtype='<f4')
                assert np.array_equal(a.reshape(-1), original[:a.size]), 'prior full-text layer-0 hidden differs'
            z = a == 0
            count = {str(n): int(z.reshape(-1, n).all(axis=1).sum()) for n in (4, 8, 16, 32)}
            assert count['32'] == 0 or count['16'] >= 2 * count['32']
            for n in (4, 8, 16, 32):
                assert count[str(n)] <= a.size // n
            # The Q8_1 quantizer rounds x/(max(abs(x))/127): every nonzero finite
            # 32-float block necessarily retains a code of magnitude ~127.
            # Its original-float group sum also vanishes for an all-zero block.
            down_per_expert = bank[layer]['bytes'] // 256
            layers.append({'layer': layer, 'down_type': bank[layer]['type'],
                           'down_bytes_per_expert': down_per_expert,
                           'zero_floats': int(z.sum()), 'zero_groups': count,
                           'total_groups_32': a.size // 32,
                           'free_four_tuple_bytes_per_token':
                               down_per_expert * 8 * count['4'] / (64 * 8 * 128)})
        four_bytes = sum(r['free_four_tuple_bytes_per_token'] for r in layers)
        denominator = traffic['one_token_weight_stream_bytes']
        summary = {'tokens': 64, 'routed_slots_per_token': 8, 'layers': layers,
                   'zero_floats': sum(r['zero_floats'] for r in layers),
                   'zero_groups': {str(n): sum(r['zero_groups'][str(n)] for r in layers)
                                   for n in (4, 8, 16, 32)},
                   'total_groups_32': 40 * 64 * 8 * 16,
                   'total_groups_4': 40 * 64 * 8 * 128,
                   'free_four_tuple_bytes_per_token': four_bytes,
                   'free_four_tuple_fraction_complete_stream': four_bytes / denominator,
                   'free_four_tuple_percent_complete_stream': 100 * four_bytes / denominator}
        summaries[split] = summary
    return {'contract': 'actual callback post-SwiGLU [64,8,512], original FP32 zero groups; exact selected Q8_1 block grammar and a FREE independently repacked four-coordinate byte ceiling, not native throughput',
            'sources_sha256': {str(p): digest(p) for p in (Path(__file__), source, capture / 'capture_all_producers', capture / 'train.log', capture / 'held.log',
                layer0 / 'train.txt', layer0 / 'held.txt',
                Path('../../data/qwen-moe/acquisition.json'), Path('../../data/qwen-moe/runtime/current/runtime.json'),
                Path('../bonsai-hip/ggml/src/ggml-cuda/quantize.cu'),
                Path('../bonsai-hip/ggml/src/ggml-cuda/vecdotq.cuh'),
                Path('../../data/qwen-moe/traffic.json'), predecessor / 'receipt.json')},
            'capture_sha256': hashes,
            'complete_one_read_weight_bytes_per_token': traffic['one_token_weight_stream_bytes'],
            'splits': summaries}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture', type=Path, required=True)
    p.add_argument('--predecessor', type=Path, default=Path('../../data/qwen-moe/all-producers'))
    p.add_argument('--layer0', type=Path, default=Path('../../data/qwen-moe/route-capture'))
    p.add_argument('--traffic', type=Path, default=Path('../../data/qwen-moe/traffic.json'))
    p.add_argument('--source', type=Path, default=Path(__file__).with_name('capture_all_producers.cpp'))
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = scan(args.capture, args.predecessor, args.layer0, json.loads(args.traffic.read_text()), args.source)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    for split, record in result['splits'].items():
        print(split, record['zero_groups'], record['free_four_tuple_percent_complete_stream'])


if __name__ == '__main__':
    main()
