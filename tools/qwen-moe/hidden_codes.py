#!/usr/bin/env python3
"""Price actual routed post-SwiGLU activation codes against the pinned Q5_K down bank."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path

import numpy as np

BASE = Path('../../data/qwen-moe')
LIB = Path('../../data/qwen-moe/runtime/current/bin/libggml-base.so')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def load(prefix):
    tokens = np.fromfile(str(prefix) + '.tokens', dtype=np.int32)
    def arr(name, dtype, shape):
        return np.memmap(str(prefix) + '.0.' + name + '-0.bin', dtype=dtype, mode='r', shape=(len(tokens), *shape))
    return dict(ids=arr('ffn_moe_topk', np.int32, (8,)),
                scores=arr('ffn_moe_weights_norm', np.float32, (8,)),
                hidden=arr('ffn_moe_swiglu', np.float32, (8, 512)),
                down=arr('ffn_moe_down', np.float32, (8, 2048)))


def quantize(x, bits, clip):
    # Signed symmetric group-32 integer code, a separate FP16 scale per group.
    g = np.asarray(x, dtype=np.float32).reshape(-1, 16, 32)
    qmax = 2 ** (bits - 1) - 1
    scale = (np.max(np.abs(g), axis=-1, keepdims=True) * clip / qmax).astype(np.float16).astype(np.float32)
    scale = np.maximum(scale, np.finfo(np.float16).tiny)
    codes = np.clip(np.rint(g / scale), -qmax, qmax).astype(np.int8)
    return (codes.astype(np.float32) * scale).reshape(-1, 512)


def measure(base, libpath, out):
    traffic = json.loads((base / 'traffic.json').read_text())
    info = next(t for t in traffic['tensors'] if t['name'] == 'blk.0.ffn_down_exps.weight')
    assert info['type'] == 'Q5_K' and info['shape'] == [512, 2048, 256]
    model = base / 'Qwen3.6-35B-A3B-UD-Q4_K_M.gguf'
    offset = (traffic['header_bytes'] + 31) // 32 * 32 + info['offset']
    bank = np.memmap(model, dtype=np.uint8, mode='r', offset=offset, shape=(256, 2048 * 512 // 256 * 176))
    lib = ctypes.CDLL(str(libpath))
    decode = lib.dequantize_row_q5_K
    decode.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int64)
    decode.restype = None
    splits = {s: load(base / 'route-capture' / s) for s in ('train', 'held')}
    for a in splits.values():
        assert np.all((a['ids'] >= 0) & (a['ids'] < 256))
        assert np.all(np.isfinite(a['hidden'])) and np.all(np.isfinite(a['down']))
        assert np.allclose(a['scores'].sum(axis=1), 1, atol=1e-5)
    clips = (1., .75, .5)
    arms = [(bits, clip) for bits in (4, 8) for clip in clips]
    accum = {}
    for split, a in splits.items():
        n = len(a['ids'])
        accum[split] = {'ref': np.zeros((n, 2048), np.float64),
                        'fp32': np.zeros((n, 2048), np.float64),
                        **{f'{b}_{c}': np.zeros((n, 2048), np.float64) for b, c in arms}}
        accum[split]['ref'][:] = np.einsum('te,ted->td', a['scores'].astype(np.float64), a['down'].astype(np.float64))
    w = np.empty((2048, 512), np.float32)
    for e in range(256):
        occurrences = {}
        for split, a in splits.items():
            row, slot = np.where(a['ids'] == e)
            if len(row):
                occurrences[split] = (row, slot)
        if not occurrences:
            continue
        source = bank[e]
        decode(source.ctypes.data_as(ctypes.c_void_p), w.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), w.size)
        for split, (row, slot) in occurrences.items():
            a = splits[split]
            x = np.asarray(a['hidden'][row, slot], np.float32)
            score = np.asarray(a['scores'][row, slot], np.float64)
            target = accum[split]
            target['fp32'][row] += score[:, None] * (x @ w.T).astype(np.float64)
            for bits, clip in arms:
                q = quantize(x, bits, clip)
                target[f'{bits}_{clip}'][row] += score[:, None] * (q @ w.T).astype(np.float64)
    result = {'scope': 'Qwen3.6 layer 0, 256 routed GGUF Q5_K down experts, actual train/held captures',
              'observation': 'FP64 score-weighted sum of FP32 dequantized-GGUF down products; local, not native FP32 logits',
              'model_sha256': json.loads((base / 'acquisition.json').read_text()).get('sha256'),
              'traffic_sha256': sha(base / 'traffic.json'), 'q5_down_bank_sha256': hashlib.sha256(bank).hexdigest(),
              'lib_sha256': sha(libpath.resolve()), 'source_sha256': sha(Path(__file__)),
              'captures': {s: {k: sha(base / 'route-capture' / f'{s}.0.{name}-0.bin') for k, name in (
                  ('ids', 'ffn_moe_topk'), ('scores', 'ffn_moe_weights_norm'),
                  ('hidden', 'ffn_moe_swiglu'), ('down', 'ffn_moe_down'))} for s in splits},
              'rate_bytes_per_expert': {'4': 512 // 2 + 16 * 2, '8': 512 + 16 * 2},
              'groups': 16, 'group_size': 32, 'round': 'nearest-even to integer, FP16-rounded per-group scale',
              'arms': {}, 'n': {s: len(a['ids']) for s, a in splits.items()}}
    for split, r in accum.items():
        denom = np.square(r['fp32']).sum(dtype=np.float64)
        def relative(a, b):
            return float(np.sqrt(np.square(a - b).sum(dtype=np.float64) / denom))
        result['arms'][split] = {'native_capture_vs_dequant_fp32': relative(r['ref'], r['fp32']),
                                **{key: {'vs_dequant_fp32': relative(value, r['fp32']),
                                         'vs_native_capture': relative(value, r['ref'])}
                                   for key, value in r.items() if key not in ('ref', 'fp32')}}
    for bits in (4, 8):
        best = min(clips, key=lambda c: result['arms']['train'][f'{bits}_{c}']['vs_dequant_fp32'])
        result[f'train_selected_clip_{bits}'] = best
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'arms': result['arms'], 'selected': {str(b): result[f'train_selected_clip_{b}'] for b in (4, 8)}}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--base', type=Path, default=BASE)
    p.add_argument('--lib', type=Path, default=LIB)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    measure(args.base, args.lib, args.output)
