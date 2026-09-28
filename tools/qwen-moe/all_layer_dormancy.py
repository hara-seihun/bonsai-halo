#!/usr/bin/env python3
"""Census exact decoded Q4_K gate/up zero rows in all Qwen MoE layers.

Run bounded layer shards with --first/--last; --aggregate combines all forty.
No GPU, inference, or native runtime changes are involved.
"""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path

import numpy as np

BASE = Path('../../data/qwen-moe')


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def layer_masks(base, traffic, lib, layer):
    gguf = base / 'Qwen3.6-35B-A3B-UD-Q4_K_M.gguf'
    start = (traffic['header_bytes'] + 31) // 32 * 32
    banks = []
    for name in ('gate', 'up'):
        t = next(t for t in traffic['tensors'] if t['name'] == f'blk.{layer}.ffn_{name}_exps.weight')
        assert t['type'] == 'Q4_K' and t['shape'] == [2048, 512, 256]
        banks.append(np.memmap(gguf, dtype='u1', mode='r', offset=start + t['offset'], shape=(256, t['bytes'] // 256)))
    decoder = lib.dequantize_row_q4_K
    decoder.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int64)
    decoder.restype = None
    weights = np.empty((512, 2048), dtype=np.float32)
    masks = []
    mismatches = []
    for expert in range(256):
        both = []
        for bank in banks:
            row = bank[expert]
            decoder(row.ctypes.data_as(ctypes.c_void_p), weights.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), weights.size)
            both.append(~np.any(weights != 0, axis=1))
        mismatches.extend([{'expert': expert, 'channel': int(i), 'gate_zero': bool(both[0][i]), 'up_zero': bool(both[1][i])}
                           for i in np.flatnonzero(both[0] != both[1])])
        masks.append(np.packbits(both[0] & both[1], bitorder='little').tobytes().hex())
    return masks, mismatches


def routes(base, split, layer):
    path = base / 'all-layer-routes' / f'{split}.layer-{layer}.i32'
    arr = np.fromfile(path, dtype='<i4').reshape(-1, 8)
    assert arr.min() >= 0 and arr.max() < 256 and all(len(set(row)) == 8 for row in arr)
    return arr, sha(path)


def run(args):
    base = args.base
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    traffic_path = base / 'traffic.json'
    traffic = json.loads(traffic_path.read_text())
    libpath = (base / 'runtime/current/bin/libggml-base.so').resolve()
    lib = ctypes.CDLL(str(libpath))
    for layer in range(args.first, args.last):
        masks, mismatch = layer_masks(base, traffic, lib, layer)
        counts = [int.from_bytes(bytes.fromhex(x), 'little').bit_count() for x in masks]
        stats = {}
        for split in ('train', 'held'):
            ids, digest = routes(base, split, layer)
            stats[split] = {'rows': len(ids), 'route_sha256': digest,
                            'dormant_slots': int(np.asarray(counts)[ids].sum()),
                            'slot_count': ids.size,
                            'seen_experts': int(np.unique(ids).size)}
        result = {'layer': layer, 'contract': 'Decoded installed Q4_K all-zero output rows; paired gate/up mask. Down Q5_K unchanged.',
                  'source_sha256': sha(__file__), 'traffic_sha256': sha(traffic_path),
                  'library_sha256': sha(libpath),
                  'model_sha256': json.loads((base / 'acquisition.json').read_text())['sha256'],
                  'masks_hex': masks, 'gate_up_mismatches': mismatch,
                  'dormant_channels': sum(counts), 'min_channels': min(counts), 'max_channels': max(counts),
                  'splits': stats}
        path = output / f'layer-{layer}.json'
        path.write_text(json.dumps(result, indent=2) + '\n')
        print(f'layer {layer}: dormant={sum(counts)} gate/up mismatches={len(mismatch)}', flush=True)


def aggregate(args):
    source = args.output
    layers = [json.loads((source / f'layer-{i}.json').read_text()) for i in range(40)]
    identities = ('source_sha256', 'traffic_sha256', 'library_sha256', 'model_sha256')
    assert all(all(layer[k] == layers[0][k] for k in identities) for layer in layers)
    assert all(layer['layer'] == i and len(layer['masks_hex']) == 256 for i, layer in enumerate(layers))
    stats = {}
    # One-read complete-model comparator: each token reads eight selected experts per
    # layer, plus nonexpert weights and one embedding lookup. Gate/up Q4_K rows
    # are independently packed 2048-wide, 2304 bytes per dormant pair of rows.
    whole = 2626187904
    for split in ('train', 'held'):
        rows = {r['splits'][split]['rows'] for r in layers}
        assert len(rows) == 1
        counts = [r['splits'][split]['dormant_slots'] for r in layers]
        saved = 2304 * sum(counts) / next(iter(rows))
        stats[split] = {'tokens': next(iter(rows)), 'dormant_slots': sum(counts),
                        'total_slot_coordinates': 512 * sum(r['splits'][split]['slot_count'] for r in layers),
                        'saved_gate_up_bytes_per_token_free_row_compaction': saved,
                        'fraction_complete_one_read_bytes': saved / whole,
                        'layers': counts}
    result = {'contract': 'Free paired gate/up row-only compaction of decoded all-zero Q4_K rows, with original per-row quantization; down Q5_K remains dense. Unpaired zero rows excluded. Real-number map only, not FP32 identity or native savings.',
              'identities': {k: layers[0][k] for k in identities},
              'shard_sha256': [sha(source / f'layer-{i}.json') for i in range(40)],
              'dormant_channels': sum(r['dormant_channels'] for r in layers),
              'total_channels': 40 * 256 * 512,
              'gate_up_mismatches': [dict(layer=r['layer'], **m) for r in layers for m in r['gate_up_mismatches']],
              'per_layer': [r['dormant_channels'] for r in layers], 'splits': stats,
              'row_image_bytes_saved': 2304 * sum(r['dormant_channels'] for r in layers),
              'mask_bytes': 40 * 256 * 64}
    path = source / 'aggregate.json'
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('dormant_channels', 'total_channels', 'gate_up_mismatches', 'row_image_bytes_saved', 'mask_bytes', 'splits')}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, default=BASE)
    p.add_argument('--output', type=Path, default=BASE / 'all-layer-dormancy')
    p.add_argument('--first', type=int, default=0)
    p.add_argument('--last', type=int, default=40)
    p.add_argument('--aggregate', action='store_true')
    a = p.parse_args()
    assert 0 <= a.first <= a.last <= 40
    aggregate(a) if a.aggregate else run(a)
