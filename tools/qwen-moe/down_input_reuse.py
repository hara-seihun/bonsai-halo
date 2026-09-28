#!/usr/bin/env python3
"""Census exact prepared down-input reuse on actual forty-layer Qwen routes."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

CAPTURE = Path('../../data/qwen-moe/all-down-zero')
QUANT = Path('../bonsai-hip/ggml/src/ggml-cuda/quantize.cu')
MMQ = Path('../bonsai-hip/ggml/src/ggml-cuda/mmq.cuh')
RUNTIME = Path('../../data/qwen-moe/runtime/current/runtime.json')
TRAFFIC = Path('../../data/qwen-moe/traffic.json')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def layer(capture, receipt, split, index):
    paths = [capture / f'{split}.layer-{index}.{suffix}' for suffix in
             ('ffn_moe_swiglu.f32', 'ffn_moe_topk.i32')]
    for path in paths:
        assert digest(path) == receipt['capture_sha256'][path.name], path
    x = np.fromfile(paths[0], dtype='<f4').reshape(64, 8, 16, 32)
    ids = np.fromfile(paths[1], dtype='<i4').reshape(64, 8)
    assert np.isfinite(x).all() and np.all((0 <= ids) & (ids < 256))
    assert all(len(set(row)) == 8 for row in ids)
    maximum = np.max(np.abs(x), axis=-1, keepdims=True)
    with np.errstate(invalid='ignore', divide='ignore'):
        scaled = x * (np.float32(127) / maximum)
        code = np.copysign(np.floor(np.abs(scaled) + np.float32(0.5)), scaled).astype(np.int8)
    code[maximum[..., 0] == 0] = 0
    assert np.all(np.max(np.abs(code), axis=-1) >= 126)

    # Codes alone are a necessary condition for whole Q8_1 operand equality;
    # the native FP16 scale and FP16 input sum only shrink these free ceilings.
    # A one-token dictionary is reset at every routed layer. Other dictionaries
    # persist across tokens, separately per observed text split.
    same_k, any_k, sign_any, temporal, temporal_any = set(), set(), set(), set(), set()
    count = Counter()
    examples = []
    for token in range(64):
        local = Counter()
        for slot in range(8):
            expert = int(ids[token, slot])
            for k in range(16):
                b = code[token, slot, k].tobytes()
                key = (k, b)
                neg = (-code[token, slot, k]).tobytes()
                orbit = (b if b < neg else neg)
                count['blocks'] += 1
                if key in same_k:
                    count['same_k_all_tokens'] += 1
                if b in any_k:
                    count['any_k_all_tokens'] += 1
                if orbit in sign_any:
                    count['sign_orbit_any_k_all_tokens'] += 1
                if (expert, k, b) in temporal:
                    count['same_expert_k_all_tokens'] += 1
                if (expert, b) in temporal_any:
                    count['same_expert_any_k_all_tokens'] += 1
                if key in local:
                    count['co_route_same_k'] += 1
                    if len(examples) < 5:
                        examples.append({'token': token, 'expert': expert, 'k': k,
                                         'first_expert': local[key]})
                local[key] = expert
                same_k.add(key)
                any_k.add(b)
                sign_any.add(orbit)
                temporal.add((expert, k, b))
                temporal_any.add((expert, b))
        count['complete_hidden_rows'] += 8
    count['repeated_expert_visits'] = sum(v - 1 for v in Counter(ids.flat).values())
    return dict(count), examples, {p.name: digest(p) for p in paths}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture', type=Path, default=CAPTURE)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    receipt = json.loads((a.capture / 'receipt.json').read_text())
    traffic = json.loads(TRAFFIC.read_text())
    down_type = {int(k): next(t['type'] for t in v['tensors'] if 'ffn_down_exps.weight' in t['name'])
                 for k, v in traffic['layers'].items()}
    assert len(down_type) == 40 and set(down_type.values()) == {'Q5_K', 'Q6_K'}
    output = {'contract': 'CPU round-away-from-zero MMQ Q8_1 32-code emulation on actual callback post-SwiGLU inputs. Code equality is necessary but not sufficient for full native operand equality (FP16 scales, plus input sum on Q5_K DS4); signed orbit grants free negation. Co-route same-K and across-token same-expert same-K are free exact-code preparation-reuse ceilings, not shared weight-dot reuse. No GPU map or native FP32 equality asserted.',
              'domain': 'two separate 64-token real-text prompt captures; all 40 layers, eight routed experts/token, sixteen 32-float blocks/expert',
              'capture_receipt_sha256': digest(a.capture / 'receipt.json'),
              'quantizer_sha256': digest(QUANT), 'mmq_layout_source_sha256': digest(MMQ),
              'installed_runtime_sha256': digest(RUNTIME),
              'traffic_sha256': digest(TRAFFIC), 'source_sha256': digest(Path(__file__)),
              'native_mmq_down_layout': 'Q5_K -> DS4 (32 values per scale and sum); Q6_K -> D4 (32 per scale, no input sum); both use d_inv=127/amax, q=roundf(x*d_inv)',
              'down_type_by_layer': down_type,
              'model_sha256': json.loads((a.capture.parent / 'all-producers' / 'receipt.json').read_text())['model_sha256'],
              'full_one_read_weight_bytes_per_token': traffic['one_token_weight_stream_bytes'],
              'panels': {}}
    for split in ('train', 'held'):
        totals = Counter()
        per_layer, files = [], {}
        for i in range(40):
            count, examples, hashes = layer(a.capture, receipt, split, i)
            totals.update(count)
            files.update(hashes)
            per_layer.append({'layer': i, 'counts': count, 'collision_examples': examples})
        assert totals['blocks'] == 40 * 64 * 8 * 16
        output['panels'][split] = {'totals': dict(totals), 'layers': per_layer, 'input_sha256': files,
                                   'free_all_prepared_down_input_bytes_per_token': 40 * 8 * 16 * 36,
                                   'free_all_prepared_down_input_percent_one_read_weights':
                                       100 * 40 * 8 * 16 * 36 / traffic['one_token_weight_stream_bytes']}
        print(split, dict(totals))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(output, indent=2, sort_keys=True) + '\n')


if __name__ == '__main__':
    main()
