#!/usr/bin/env python3
"""Count same-K Q8_0 integer-dot reuse under exact sign reversal."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from q8_block_reuse import BLOCK, INVENTORY, MODEL, sha

OUTPUT = Path('../../data/qwen-moe/q8-signed-orbits')


def signed_orbits(blocks, chunk=8):
    rows, kb = blocks.shape
    counts = dict(blocks=rows * kb, invertible_nonzero=0, nonzero_orbit_reuse=0,
                  opposite_sign_reuse=0, mixed_orbits=0, max_orbit=0,
                  proportional_reuse=0, nonprimitive=0, max_proportional_orbit=0)
    for start in range(0, kb, chunk):
        width = min(chunk, kb - start)
        codes = np.ascontiguousarray(blocks['codes'][:, start:start + width]).reshape(-1, 32).view('i1').reshape(-1, 32)
        nonzero = np.any(codes != 0, axis=1)
        valid = nonzero & ~np.any(codes == -128, axis=1)
        counts['invertible_nonzero'] += int(valid.sum())
        if not valid.any():
            raise ValueError('chunk without invertible vectors would skip primitive census')
        positions = np.broadcast_to(np.arange(start, start + width, dtype='<u4'), (rows, width)).reshape(-1)[valid]
        code = codes[valid].astype(np.int16)
        first = np.argmax(code != 0, axis=1)
        reversed_sign = code[np.arange(len(code)), first] < 0
        code[reversed_sign] *= -1
        key = np.empty((len(code), 36), dtype='u1')
        key[:, :4] = positions.view('u1').reshape(-1, 4)
        key[:, 4:] = code.astype('i1').view('u1')
        _, inverse, sizes = np.unique(key.view('V36').reshape(-1), return_inverse=True, return_counts=True)
        counts['nonzero_orbit_reuse'] += int((sizes - 1).sum())
        plus = np.bincount(inverse, weights=reversed_sign.astype(np.int64), minlength=len(sizes)).astype(np.int64)
        mixed = (plus > 0) & (plus < sizes)
        counts['mixed_orbits'] += int(mixed.sum())
        counts['opposite_sign_reuse'] += int(np.minimum(plus, sizes - plus).sum())
        counts['max_orbit'] = max(counts['max_orbit'], int(sizes.max()))
        # Two nonzero integer dots are proportional on unrestricted inputs iff
        # their primitive signed vectors agree. Unlike sign reversal, a rational
        # ratio must also pay a new output scale and its FP32 rounding.
        all_codes = codes[nonzero].astype(np.int16)
        divisors = np.gcd.reduce(np.abs(all_codes), axis=1)
        counts['nonprimitive'] += int((divisors > 1).sum())
        all_codes //= divisors[:, None]
        first = np.argmax(all_codes != 0, axis=1)
        all_codes[all_codes[np.arange(len(all_codes)), first] < 0] *= -1
        primitive_key = np.empty((len(all_codes), 68), dtype='u1')
        primitive_key[:, :4] = np.broadcast_to(
            np.arange(start, start + width, dtype='<u4'), (rows, width)
        ).reshape(-1)[nonzero].view('u1').reshape(-1, 4)
        primitive_key[:, 4:] = all_codes.astype('<i2').view('u1').reshape(-1, 64)
        _, primitive_sizes = np.unique(primitive_key.view('V68').reshape(-1), return_counts=True)
        counts['proportional_reuse'] += int((primitive_sizes - 1).sum())
        counts['max_proportional_orbit'] = max(counts['max_proportional_orbit'], int(primitive_sizes.max()))
    return counts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, default=MODEL)
    p.add_argument('--inventory', type=Path, default=INVENTORY)
    p.add_argument('--output', type=Path, default=OUTPUT)
    p.add_argument('--first', type=int, default=0)
    p.add_argument('--count', type=int, default=251)
    p.add_argument('--summarize', action='store_true')
    a = p.parse_args()
    inv = json.loads(a.inventory.read_text())
    assert inv['metadata']['general.architecture'] == 'qwen35moe'
    assert inv['header_sha256'] == hashlib.sha256(a.model.open('rb').read(inv['header_bytes'])).hexdigest()
    align = inv['metadata'].get('general.alignment', 32)
    base = (inv['header_bytes'] + align - 1) // align * align
    tensors = [t for t in inv['tensors'] if t['type'] == 'Q8_0' and
               t['name'] != 'token_embd.weight' and '_exps.' not in t['name']]
    a.output.mkdir(parents=True, exist_ok=True)
    source_hash = sha(Path(__file__))
    inventory_hash = sha(a.inventory)
    if a.summarize:
        receipts = [json.loads((a.output / f'{i:03d}.json').read_text()) for i in range(len(tensors))]
        assert all(r['name'] == t['name'] and r['source_sha256'] == source_hash and
                   r['inventory_sha256'] == inventory_hash for r, t in zip(receipts, tensors))
        fields = ('blocks', 'invertible_nonzero', 'nonzero_orbit_reuse', 'opposite_sign_reuse',
                  'mixed_orbits', 'proportional_reuse', 'nonprimitive')
        result = {field: sum(r[field] for r in receipts) for field in fields}
        result.update(max_orbit=max(r['max_orbit'] for r in receipts),
                      max_proportional_orbit=max(r['max_proportional_orbit'] for r in receipts),
                      tensors=len(receipts),
                      q8_bytes=sum(r['bytes'] for r in receipts),
                      complete_one_read_bytes=inv['one_token_weight_stream_bytes'],
                      source_sha256=source_hash, inventory_sha256=inventory_hash,
                      header_sha256=inv['header_sha256'], model_sha256=inv.get('model_sha256'),
                      tensor_receipts=[dict(name=r['name'], sha256=sha(a.output / f'{i:03d}.json'),
                                            payload_sha256=r['tensor_sha256']) for i, r in enumerate(receipts)])
        (a.output / 'receipt.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps({k: v for k, v in result.items() if k != 'tensor_receipts'}, indent=2))
        return
    image = np.memmap(a.model, dtype='u1', mode='r')
    for i in range(a.first, min(a.first + a.count, len(tensors))):
        t = tensors[i]
        destination = a.output / f'{i:03d}.json'
        if destination.exists():
            old = json.loads(destination.read_text())
            if old['name'] == t['name'] and old['source_sha256'] == source_hash and old['inventory_sha256'] == inventory_hash:
                continue
        kb = t['shape'][0] // 32
        assert t['shape'][0] % 32 == 0 and t['bytes'] % (34 * kb) == 0
        rows = t['bytes'] // (34 * kb)
        payload = image[base + t['offset']:base + t['offset'] + t['bytes']]
        result = signed_orbits(payload.view(BLOCK).reshape(rows, kb))
        result.update(name=t['name'], bytes=t['bytes'], tensor_sha256=hashlib.sha256(payload).hexdigest(),
                      source_sha256=source_hash, inventory_sha256=inventory_hash,
                      header_sha256=inv['header_sha256'], model_sha256=inv.get('model_sha256'), tensor_index=i)
        destination.write_text(json.dumps(result, indent=2) + '\n')
        print(i, t['name'], result['nonzero_orbit_reuse'], flush=True)


if __name__ == '__main__':
    main()
