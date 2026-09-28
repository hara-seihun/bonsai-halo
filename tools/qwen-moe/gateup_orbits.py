#!/usr/bin/env python3
"""Exact same-coordinate eight-byte Q4_K gate/up packet census on actual routes."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

MODEL = Path('../../data/qwen-moe/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf')
INV = Path('../../data/qwen-moe/traffic.json')
ROUTES = Path('../../data/qwen-moe/all-producers')
OUT = Path('../../data/qwen-moe/gateup-orbits')
TOKEN_INDEX = (1, 32, 63)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def packet_counts(banks, ids):
    # Shape [gate/up, expert, output row, input block, packet, byte].
    payload = np.stack([b[ids, :, :, 16:].reshape(8, 512, 8, 16, 8) for b in banks])
    # A packet at a different K position is not a reusable input dot.
    packet = np.ascontiguousarray(payload.transpose(3, 4, 0, 1, 2, 5))
    keys = packet.view('<u8').reshape(128, 2, 4096)
    header = np.stack([b[ids, :, :, :4] for b in banks])
    nonzero_scale = np.any((header.copy().view('<u2').reshape(2, 8, 512, 8, 2) & 0x7fff) != 0, axis=-1)
    zero_block = ~nonzero_scale
    paired_zero_rows = np.all(zero_block[0] & zero_block[1], axis=-1)
    unpaired_zero_blocks = int(np.count_nonzero(zero_block & ~paired_zero_rows[None, :, :, None]))
    active = np.repeat(nonzero_scale.transpose(3, 0, 1, 2).reshape(8, 2, 4096), 16, axis=0)
    active_saved = 0
    active_intra = [0, 0]
    for pos in range(128):
        for family in range(2):
            values = np.sort(keys[pos, family, active[pos, family]])
            active_intra[family] += int(np.count_nonzero(np.diff(values) == 0))
        values = np.sort(keys[pos].reshape(-1)[active[pos].reshape(-1)])
        active_saved += int(np.count_nonzero(np.diff(values) == 0))
    sorted_family = [np.sort(keys[:, i], axis=1) for i in range(2)]
    intra = [int(np.count_nonzero(np.diff(s, axis=1) == 0)) for s in sorted_family]
    zero_uses = [int(np.count_nonzero(keys[:, i] == 0)) for i in range(2)]
    # Zero-code packets still have an affine Q4_K minimum term, so keep both counts.
    zero_savings = [int(np.maximum(np.count_nonzero(keys[:, i] == 0, axis=1) - 1, 0).sum()) for i in range(2)]
    nonzero_intra = [v - z for v, z in zip(intra, zero_savings)]
    sorted_union = np.sort(keys.reshape(128, 8192), axis=1)
    union = int(np.count_nonzero(np.diff(sorted_union, axis=1) == 0))
    union_zero_uses = sum(zero_uses)
    nonzero_union = union - int(np.maximum(np.count_nonzero(keys.reshape(128, 8192) == 0, axis=1) - 1, 0).sum())
    return {'gate_saved': intra[0], 'up_saved': intra[1], 'cross_saved': union - sum(intra),
            'union_saved': union, 'packet_uses': 128 * 8192,
            'zero_uses': union_zero_uses, 'nonzero_union_saved': nonzero_union,
            'nonzero_cross_saved': nonzero_union - sum(nonzero_intra),
            'active_packet_uses': int(active.sum()), 'active_union_saved': active_saved,
            'active_cross_saved': active_saved - sum(active_intra),
            'paired_zero_rows': int(paired_zero_rows.sum()),
            'unpaired_zero_blocks': unpaired_zero_blocks}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--first', type=int, default=0)
    p.add_argument('--count', type=int, default=40)
    p.add_argument('--summarize', action='store_true')
    p.add_argument('--output', type=Path, default=OUT)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    inv = json.loads(INV.read_text())
    assert inv['metadata']['general.architecture'] == 'qwen35moe'
    with MODEL.open('rb') as f:
        assert hashlib.sha256(f.read(inv['header_bytes'])).hexdigest() == inv['header_sha256']
    tensors = {t['name']: t for t in inv['tensors']}
    src = digest(Path(__file__))
    if a.summarize:
        items = [json.loads((a.output / f'layer-{i:02d}.json').read_text()) for i in range(40)]
        assert all(x['source_sha256'] == src and x['inventory_sha256'] == digest(INV) and
                   x['layer'] == i for i, x in enumerate(items))
        totals = {split: {field: sum(s[field] for x in items for s in x['routes'][split])
                          for field in ('gate_saved', 'up_saved', 'cross_saved', 'union_saved', 'packet_uses',
                                        'zero_uses', 'nonzero_union_saved', 'nonzero_cross_saved',
                                        'active_packet_uses', 'active_union_saved', 'active_cross_saved',
                                        'paired_zero_rows', 'unpaired_zero_blocks')}
                  for split in ('train', 'held')}
        acquisition = Path('../../data/qwen-moe/acquisition.json')
        capture = ROUTES / 'receipt.json'
        runtime = Path('../../data/qwen-moe/runtime/current/runtime.json')
        result = {'source_sha256': src, 'inventory_sha256': digest(INV),
                  'acquisition_sha256': digest(acquisition), 'capture_receipt_sha256': digest(capture),
                  'installed_runtime_sha256': digest(runtime),
                  'model_sha256': json.loads(acquisition.read_text())['sha256'],
                  'token_indices_per_split': TOKEN_INDEX, 'one_read_model_bytes': inv['one_token_weight_stream_bytes'],
                  'layer_receipts': [{'sha256': digest(a.output / f'layer-{i:02d}.json'),
                                      'layer': i} for i in range(40)], 'totals': totals}
        (a.output / 'receipt.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(totals, indent=2))
        return
    image = np.memmap(MODEL, dtype='u1', mode='r')
    align = inv['metadata'].get('general.alignment', 32)
    base = (inv['header_bytes'] + align - 1) // align * align
    for layer in range(a.first, min(a.first + a.count, 40)):
        route_paths = {split: ROUTES / f'{split}.layer-{layer}.ffn_moe_topk.i32' for split in ('train', 'held')}
        banks = []
        tensor_hashes = {}
        for family in ('gate', 'up'):
            t = tensors[f'blk.{layer}.ffn_{family}_exps.weight']
            assert t['shape'] == [2048, 512, 256] and t['type'] == 'Q4_K' and t['bytes'] == 256*512*8*144
            view = image[base+t['offset']:base+t['offset']+t['bytes']]
            tensor_hashes[family] = hashlib.sha256(view).hexdigest()
            banks.append(view.reshape(256, 512, 8, 144))
        result = {'layer': layer, 'source_sha256': src, 'inventory_sha256': digest(INV),
                  'tensor_sha256': tensor_hashes, 'routes_sha256': {k: digest(v) for k, v in route_paths.items()},
                  'routes': {}}
        for split, path in route_paths.items():
            ids = np.fromfile(path, dtype='<i4').reshape(-1, 8)
            assert ids.shape[0] >= 64 and np.all((ids >= 0) & (ids < 256))
            assert all(len(set(ids[t])) == 8 for t in TOKEN_INDEX)
            result['routes'][split] = [dict(token=t, ids=ids[t].tolist(), **packet_counts(banks, ids[t]))
                                        for t in TOKEN_INDEX]
        (a.output / f'layer-{layer:02d}.json').write_text(json.dumps(result, indent=2) + '\n')
        print(layer, [(s, sum(r['union_saved'] for r in result['routes'][s])) for s in ('train','held')], flush=True)


if __name__ == '__main__':
    main()
