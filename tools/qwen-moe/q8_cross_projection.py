#!/usr/bin/env python3
"""Upper-bound same-K integer-dot reuse across distinct ordinary Q8 tensors per layer."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

BASE = Path('../../data/qwen-moe')
MODEL = BASE / 'Qwen3.6-35B-A3B-UD-Q4_K_M.gguf'
INVENTORY = BASE / 'traffic.json'
OUT = BASE / 'q8-cross-projection'
BLOCK = np.dtype([('scale', '<u2'), ('codes', 'u1', (32,))])


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for piece in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(piece)
    return h.hexdigest()


def census_layer(tensors, image, base):
    matrices = []
    for tensor in tensors:
        k = tensor['shape'][0] // 32
        assert tensor['shape'][0] % 32 == 0 and tensor['bytes'] % (34 * k) == 0
        payload = image[base + tensor['offset']:base + tensor['offset'] + tensor['bytes']]
        blocks = payload.view(BLOCK).reshape(-1, k)
        matrices.append((tensor, blocks, hashlib.sha256(payload).hexdigest()))
    result = dict(blocks=sum(b.size for _, b, _ in matrices), cross_code_groups=0,
                  cross_code_extra_tensor_uses=0, cross_full_groups=0,
                  cross_full_extra_tensor_uses=0, cross_nonzero_code_groups=0,
                  cross_nonzero_code_extra_tensor_uses=0)
    # A source vector may only be shared at equal K. Grant identical activation
    # codes across *all* matrices of a layer, including unrelated producers.
    for k in range(max((b.shape[1] for _, b, _ in matrices), default=0)):
        participating = [(i, b[:, k]) for i, (_, b, _) in enumerate(matrices) if k < b.shape[1]]
        if len(participating) < 2:
            continue
        origin = np.concatenate([np.full(len(b), i, np.int16) for i, b in participating])
        packed = np.concatenate([np.ascontiguousarray(b).view('u1').reshape(-1, 34) for _, b in participating])
        for kind, values in (('code', packed[:, 2:]), ('full', packed)):
            key = np.ascontiguousarray(values).view(f'V{values.shape[1]}').reshape(-1)
            unique, inverse = np.unique(key, return_inverse=True)
            lo = np.full(len(unique), len(matrices), np.int16)
            hi = np.full(len(unique), -1, np.int16)
            np.minimum.at(lo, inverse, origin)
            np.maximum.at(hi, inverse, origin)
            cross = np.flatnonzero(lo != hi)
            result[f'cross_{kind}_groups'] += len(cross)
            if not len(cross):
                continue
            # Distinct matrix IDs per code. Within-matrix repeats were already
            # priced in the independent same-tensor census and add no cross gain.
            pair = np.unique(np.stack((inverse, origin), axis=1), axis=0)
            by_group = np.bincount(pair[:, 0], minlength=len(unique))
            result[f'cross_{kind}_extra_tensor_uses'] += int((by_group[cross] - 1).sum())
            if kind == 'code':
                nonzero = np.any(unique[cross].view('u1').reshape(-1, 32) != 0, axis=1)
                result['cross_nonzero_code_groups'] += int(nonzero.sum())
                result['cross_nonzero_code_extra_tensor_uses'] += int((by_group[cross][nonzero] - 1).sum())
    result['tensors'] = [dict(name=t['name'], shape=t['shape'], bytes=t['bytes'], payload_sha256=digest)
                         for t, _, digest in matrices]
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--first', type=int, default=0)
    p.add_argument('--count', type=int, default=40)
    p.add_argument('--model', type=Path, default=MODEL)
    p.add_argument('--inventory', type=Path, default=INVENTORY)
    p.add_argument('--out', type=Path, default=OUT)
    p.add_argument('--summarize', action='store_true')
    a = p.parse_args()
    inv = json.loads(a.inventory.read_text())
    assert inv['metadata']['general.architecture'] == 'qwen35moe'
    with a.model.open('rb') as f:
        assert hashlib.sha256(f.read(inv['header_bytes'])).hexdigest() == inv['header_sha256']
    source = sha(Path(__file__))
    inventory = sha(a.inventory)
    a.out.mkdir(parents=True, exist_ok=True)
    if a.summarize:
        layers = [json.loads((a.out / f'{i:02d}.json').read_text()) for i in range(40)]
        assert all(r['source_sha256'] == source and r['inventory_sha256'] == inventory and r['layer'] == i
                   for i, r in enumerate(layers))
        fields = [k for k in layers[0] if k == 'blocks' or k.startswith('cross_')]
        totals = {k: sum(r[k] for r in layers) for k in fields}
        totals['free_nonzero_code_byte_ceiling'] = 32 * totals['cross_nonzero_code_extra_tensor_uses']
        totals['free_full_block_byte_ceiling'] = 34 * totals['cross_full_extra_tensor_uses']
        totals['conditional_one_read_fraction_percent'] = 100 * totals['free_nonzero_code_byte_ceiling'] / inv['one_token_weight_stream_bytes']
        receipt = dict(contract='Same-K exact Q8_0 code/full-block equality between distinct nonembedding nonexpert tensors in each layer; all tensor pairs granted identical inputs and free lookup even if their producers differ. Equal integer codes reuse a single dot only under this granted common activation; differing scales still require separate FP32 folds. Per-tensor reuse excluded. No FP32 bit identity, native time, or DRAM transactions follow.',
                       model_sha256=sha(a.model), inventory_sha256=inventory, source_sha256=source,
                       header_sha256=inv['header_sha256'], complete_one_read_bytes=inv['one_token_weight_stream_bytes'],
                       q8_bytes=sum(t['bytes'] for layer in layers for t in layer['tensors']),
                       totals=totals, layers=[dict(layer=i, receipt_sha256=sha(a.out / f'{i:02d}.json'), **{k: r[k] for k in fields}) for i, r in enumerate(layers)])
        (a.out / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
        print(json.dumps(totals, indent=2))
        return
    image = np.memmap(a.model, dtype='u1', mode='r')
    base = (inv['header_bytes'] + inv['metadata'].get('general.alignment', 32) - 1) // inv['metadata'].get('general.alignment', 32) * inv['metadata'].get('general.alignment', 32)
    for i in range(a.first, min(a.first + a.count, 40)):
        tensors = [t for t in inv['tensors'] if t['name'].startswith(f'blk.{i}.') and t['type'] == 'Q8_0' and '_exps.' not in t['name']]
        r = census_layer(tensors, image, base)
        r.update(layer=i, source_sha256=source, inventory_sha256=inventory)
        (a.out / f'{i:02d}.json').write_text(json.dumps(r, indent=2) + '\n')
        print(i, len(tensors), r['cross_nonzero_code_extra_tensor_uses'], flush=True)


if __name__ == '__main__':
    main()
