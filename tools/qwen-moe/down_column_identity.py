#!/usr/bin/env python3
"""Exact decoded-Q5_K down-column equality census, sharded by layer (CPU only).

Hash only proposes equal columns; every proposed pair is compared bitwise. A
column is an output-width vector, not one independently addressed Q5_K block.
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


def census(args):
    base = args.base
    inventory = base / 'traffic.json'
    traffic = json.loads(inventory.read_text())
    libpath = (base / 'runtime/current/bin/libggml-base.so').resolve()
    lib = ctypes.CDLL(str(libpath))
    image = base / 'Qwen3.6-35B-A3B-UD-Q4_K_M.gguf'
    start = (traffic['header_bytes'] + 31) // 32 * 32
    model = np.memmap(image, dtype='u1', mode='r')
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    for layer in range(args.first, args.last):
        t = next(t for t in traffic['tensors'] if t['name'] == f'blk.{layer}.ffn_down_exps.weight')
        assert t['type'] in ('Q5_K', 'Q6_K') and t['shape'] == [512, 2048, 256]
        block_bytes = {'Q5_K': 176, 'Q6_K': 210}[t['type']]
        decode = getattr(lib, f'dequantize_row_{t["type"][0].lower()}{t["type"][1:]}')
        decode.argtypes = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64)
        decode.restype = None
        stride = 2048 * 512 // 256 * block_bytes
        assert t['bytes'] == 256 * stride
        bank = model[start + t['offset']:start + t['offset'] + t['bytes']].reshape(256, stride)
        weights = np.empty((2048, 512), dtype='<f4')
        zero = np.zeros(256, dtype='i4')
        repeated = np.zeros(256, dtype='i4')
        nonzero_repeated = np.zeros(256, dtype='i4')
        largest = np.zeros(256, dtype='i4')
        repeated_masks = []
        for expert, packed in enumerate(bank):
            decode(packed.ctypes.data, weights.ctypes.data, weights.size)
            assert np.all(np.isfinite(weights))
            columns = np.ascontiguousarray(weights.T).view('<u4').reshape(512, 2048)
            # FP32 word identity, including signed zero. Hash collisions are
            # resolved by equality on the complete 2048-word vector.
            labels = {}
            classes = {}
            membership = []
            for col in columns:
                key = hashlib.blake2b(col, digest_size=16).digest()
                candidates = labels.setdefault(key, [])
                match = next((i for i in candidates if np.array_equal(columns[i], col)), None)
                if match is None:
                    match = len(membership)
                    candidates.append(match)
                    classes[match] = 1
                else:
                    classes[match] += 1
                membership.append(match)
            assert sum(classes.values()) == 512
            is_zero = ~np.any(columns != 0, axis=1)
            zero[expert] = int(is_zero.sum())
            repeated[expert] = 512 - len(classes)
            # For each collision class, determine whether its representative
            # is a zero column; nonzero repetitions cannot share dot work with
            # zero channels.
            nonzero_repeated[expert] = sum(n - 1 for i, n in classes.items() if n > 1 and not is_zero[i])
            largest[expert] = max(classes.values())
            repeated_masks.append(np.packbits([classes[i] > 1 for i in membership], bitorder='little').tobytes().hex())
        stats = {}
        for split in ('train', 'held'):
            routepath = base / 'all-producers' / f'{split}.layer-{layer}.ffn_moe_topk.i32'
            ids = np.fromfile(routepath, dtype='<i4').reshape(-1, 8)
            assert len(ids) == 64 and ids.min() >= 0 and ids.max() < 256
            hpath = base / 'all-down-zero' / f'{split}.layer-{layer}.ffn_moe_swiglu.f32'
            hidden = np.memmap(hpath, dtype='<f4', mode='r', shape=(len(ids), 8, 512))
            mask = np.unpackbits(np.frombuffer(b''.join(bytes.fromhex(x) for x in repeated_masks), 'u1'),
                                 bitorder='little').reshape(256, 512).astype(bool)
            repeated_on_routes = mask[ids]
            stats[split] = {'tokens': len(ids), 'route_sha256': sha(routepath), 'hidden_sha256': sha(hpath),
                            'repeated_coordinates_on_routes': int(repeated_on_routes.sum()),
                            'repeated_active_coordinates': int(np.count_nonzero(hidden[repeated_on_routes])),
                            'duplicate_selected_slots': int(repeated[ids].sum()),
                            'nonzero_duplicate_selected_slots': int(nonzero_repeated[ids].sum()),
                            'zero_selected_slots': int(zero[ids].sum()),
                            'selected_coordinates': int(ids.size * 512)}
        result = {'layer': layer, 'identity': 'bitwise decoded FP32 [2048]-vector columns within each expert',
                  'source_sha256': sha(__file__), 'inventory_sha256': sha(inventory),
                  'library_sha256': sha(libpath), 'model_sha256': json.loads((base / 'acquisition.json').read_text())['sha256'],
                  'tensor': t['name'], 'quantization': t['type'], 'column_bytes': 8 * block_bytes,
                  'tensor_payload_sha256': hashlib.sha256(bank).hexdigest(),
                  'zero_by_expert': zero.tolist(), 'duplicate_by_expert': repeated.tolist(),
                  'nonzero_duplicate_by_expert': nonzero_repeated.tolist(),
                  'max_class_by_expert': largest.tolist(), 'repeated_masks_hex': repeated_masks,
                  'splits': stats}
        (output / f'layer-{layer}.json').write_text(json.dumps(result, indent=2) + '\n')
        print(f'layer {layer}: duplicate={repeated.sum()}, nonzero={nonzero_repeated.sum()}, zero={zero.sum()}', flush=True)


def aggregate(args):
    layers = [json.loads((args.output / f'layer-{i}.json').read_text()) for i in range(40)]
    keys = ('source_sha256', 'inventory_sha256', 'library_sha256', 'model_sha256')
    assert all(layer['layer'] == i and all(layer[k] == layers[0][k] for k in keys) for i, layer in enumerate(layers))
    result = {'domain': 'complete 40-layer installed Q5_K bank, exact FP32 decoded column identity within experts',
              'identities': {k: layers[0][k] for k in keys},
              'shard_sha256': [sha(args.output / f'layer-{i}.json') for i in range(40)],
              'expert_columns': 40 * 256 * 512,
              'duplicates': sum(sum(r['duplicate_by_expert']) for r in layers),
              'nonzero_duplicates': sum(sum(r['nonzero_duplicate_by_expert']) for r in layers),
              'zero_columns': sum(sum(r['zero_by_expert']) for r in layers),
              'per_layer_duplicates': [sum(r['duplicate_by_expert']) for r in layers]}
    for split in ('train', 'held'):
        selected = {r['splits'][split]['tokens'] for r in layers}
        assert selected == {64}
        counts = {k: sum(r['splits'][split][k] for r in layers) for k in
                  ('duplicate_selected_slots', 'nonzero_duplicate_selected_slots', 'zero_selected_slots',
                   'selected_coordinates', 'repeated_coordinates_on_routes', 'repeated_active_coordinates')}
        counts['per_layer_active_repeats'] = [r['splits'][split]['repeated_active_coordinates'] for r in layers]
        # A decoded column has eight quantized blocks. This GRANTS a free
        # column repack/deletion, not achievable with the installed layout.
        counts['free_saved_bytes_per_token'] = sum(
            r['splits'][split]['duplicate_selected_slots'] * r['column_bytes'] for r in layers) / 64
        counts['fraction_complete_one_read'] = counts['free_saved_bytes_per_token'] / 2626187904
        result[split] = counts
    (args.output / 'receipt.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, default=BASE)
    p.add_argument('--output', type=Path, default=BASE / 'down-column-identity')
    p.add_argument('--first', type=int, default=0)
    p.add_argument('--last', type=int, default=40)
    p.add_argument('--aggregate', action='store_true')
    a = p.parse_args()
    assert 0 <= a.first <= a.last <= 40
    aggregate(a) if a.aggregate else census(a)
