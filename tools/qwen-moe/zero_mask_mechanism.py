#!/usr/bin/env python3
"""Locate layer-0 dormant expert channels in original BF16 and decoded installed GGUF."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path

import numpy as np

BASE = Path('../../data/qwen-moe')
SENTINEL = (469, 33237)  # BF16 bits for signed 7.82438427e-38


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def capture(base, split):
    prefix = base / 'zero-mask-mechanism' / split
    ids = np.fromfile(str(prefix) + '.ffn_moe_topk-0.bin', '<i4').reshape(-1, 8)
    channels = {}
    for name in ('gate', 'up', 'swiglu'):
        path = Path(str(prefix) + '.ffn_moe_' + name + '-0.bin')
        channels[name] = np.fromfile(path, '<f4').reshape(len(ids), 8, 512)
    old_prefix = base / 'route-capture' / f'{split}.0.'
    old_ids = np.fromfile(str(old_prefix) + 'ffn_moe_topk-0.bin', '<i4').reshape(-1, 8)
    old_hidden = np.fromfile(str(old_prefix) + 'ffn_moe_swiglu-0.bin', '<f4').reshape(len(ids), 8, 512)
    assert np.array_equal(ids, old_ids) and np.array_equal(channels['swiglu'], old_hidden)
    assert np.isfinite(np.stack(list(channels.values()))).all()
    masks = {n: x == 0 for n, x in channels.items()}
    assert np.array_equal(masks['gate'], masks['up']) and np.array_equal(masks['gate'], masks['swiglu'])
    return ids, masks['gate'], {'ids': digest(Path(str(prefix) + '.ffn_moe_topk-0.bin')),
                               **{n: digest(Path(str(prefix) + '.ffn_moe_' + n + '-0.bin')) for n in channels}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=BASE)
    parser.add_argument('--first', type=int, default=0)
    parser.add_argument('--last', type=int, default=256)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not (0 <= args.first < args.last <= 256):
        parser.error('expected 0 <= first < last <= 256')
    base = args.base
    traffic_path = base / 'traffic.json'
    meta = json.loads(traffic_path.read_text())
    model = base / 'Qwen3.6-35B-A3B-UD-Q4_K_M.gguf'
    data_start = (meta['header_bytes'] + 31) // 32 * 32
    libpath = (base / 'runtime/current/bin/libggml-base.so').resolve()
    lib = ctypes.CDLL(str(libpath))
    specs = {}
    for name, quant in (('gate', 'q4_K'), ('up', 'q4_K'), ('down', 'q5_K')):
        info = next(t for t in meta['tensors'] if t['name'] == f'blk.0.ffn_{name}_exps.weight')
        assert info['type'] == quant.upper() and info['shape'][2] == 256
        bank = np.memmap(model, np.uint8, mode='r', offset=data_start + info['offset'],
                         shape=(256, info['bytes'] // 256))
        decode = getattr(lib, 'dequantize_row_' + quant)
        decode.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int64)
        decode.restype = None
        specs[name] = (bank, decode, info)
    splits = {s: capture(base, s) for s in ('train', 'held')}
    original = base / 'experts/layer-0-0-16'
    bf_gate = np.memmap(original / 'gate_up_proj.bf16', dtype='<u2', mode='r', shape=(16, 1024, 2048))
    bf_down = np.memmap(original / 'down_proj.bf16', dtype='<u2', mode='r', shape=(16, 2048, 512))
    records = []
    for expert in range(args.first, args.last):
        masks = {}
        for name, (bank, decode, _) in specs.items():
            shape = (512, 2048) if name != 'down' else (2048, 512)
            weights = np.empty(shape, dtype=np.float32)
            data = bank[expert]
            decode(data.ctypes.data_as(ctypes.c_void_p),
                   weights.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), weights.size)
            masks[name] = ~np.any(weights != 0, axis=1 if name != 'down' else 0)
        # The installed gate/up arithmetic produces zero for these channels on
        # finite inputs. Down may decode nonzero quantizer offsets; its operand
        # remains zero, so only the inactive input positions are skippable.
        inactive = masks['gate'] & masks['up']
        record = {'expert': expert, 'gate_zero': int(masks['gate'].sum()),
                  'up_zero': int(masks['up'].sum()), 'down_zero': int(masks['down'].sum()),
                  'inactive_gate_up': int(inactive.sum()),
                  'mask_hex': np.packbits(inactive, bitorder='little').tobytes().hex(),
                  'gate_up_mask_mismatch': int(np.count_nonzero(masks['gate'] != masks['up'])),
                  'down_decoded_nonzero_on_inactive': int(np.count_nonzero(inactive & ~masks['down'])), 
                  'splits': {}}
        for split, (ids, zeros, _) in splits.items():
            rows = zeros[ids == expert]
            if len(rows):
                record['splits'][split] = {'rows': len(rows),
                                           'mismatch_gate_up': int(np.count_nonzero(rows != inactive)),
                                           'mask_varying_rows': int(np.count_nonzero(rows != rows[0]))}
        if expert < 16:
            pairs = np.isin(bf_gate[expert], SENTINEL).all(axis=1).reshape(2, 512)
            down = np.isin(bf_down[expert], SENTINEL).all(axis=0)
            original_common = pairs[0] & pairs[1] & down
            record['original_bf16'] = {'gate_sentinel': int(pairs[0].sum()),
                                       'up_sentinel': int(pairs[1].sum()),
                                       'down_sentinel': int(down.sum()),
                                       'common_sentinel': int(original_common.sum()),
                                       'mismatch_with_gguf_gate_up': int(np.count_nonzero(original_common != inactive)),
                                       'mismatched_original_matrices': int(np.count_nonzero((pairs[0] != pairs[1]) |
                                                                                           (pairs[0] != down)))}
        records.append(record)
    result = {'contract': 'exact zero rows in decoded Q4_K gate/up for layer 0, so SwiGLU input and output zero on finite inputs. Q5_K down columns are not generally decoded zero, but multiply zero h. Original BF16 signed sentinel matched in all three tensors only for experts 0..15; captured gate/up/SwiGLU masks confirm the finite producer observation.',
              'scope': [args.first, args.last], 'source_sha256': digest(__file__),
              'binary_sha256': digest(base / 'qwen-capture-zero-producers'),
              'capture_source_sha256': digest(Path(__file__).with_name('capture_zero_producers.cpp')),
              'model_sha256': json.loads((base / 'acquisition.json').read_text())['sha256'],
              'traffic_sha256': digest(traffic_path), 'library_sha256': digest(libpath),
              'original_sha256': {n: digest(original / n) for n in ('gate_up_proj.bf16', 'down_proj.bf16')},
              'captures': {s: v[2] for s, v in splits.items()},
              'records': records,
              'totals': {key: sum(r[key] for r in records) for key in
                         ('gate_zero', 'up_zero', 'down_zero', 'inactive_gate_up', 'gate_up_mask_mismatch',
                          'down_decoded_nonzero_on_inactive')}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'scope': result['scope'], 'totals': result['totals'],
                      'captured_mismatches': sum(v['mismatch_gate_up'] for r in records for v in r['splits'].values()),
                      'original_mismatches': sum(r.get('original_bf16', {}).get('mismatch_with_gguf_gate_up', 0) for r in records)}))


if __name__ == '__main__':
    main()
