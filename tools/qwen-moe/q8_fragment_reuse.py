#!/usr/bin/env python3
"""Same-K exact Q8 code-fragment dot reuse in the installed nonexpert bank."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from q8_block_reuse import BLOCK, INVENTORY, MODEL, sha

OUTPUT = Path('../../data/qwen-moe/q8-fragment-reuse')


def census(blocks, widths=(4, 8), chunk=8):
    rows, kb = blocks.shape
    result = {}
    for width in widths:
        fragments = 32 // width
        repeated = nonzero = zero = groups = maximum = 0
        for first in range(0, kb, chunk):
            end = min(first + chunk, kb)
            codes = np.ascontiguousarray(blocks['codes'][:, first:end]).reshape(rows, end-first, fragments, width)
            # Input K coordinate and position within its 32-code block are part of
            # the key; different positions multiply different activation inputs.
            for j in range(fragments):
                payload = np.ascontiguousarray(codes[:, :, j, :]).reshape(-1, width)
                key = np.empty((len(payload), 4 + width), dtype='u1')
                key[:, :4] = np.broadcast_to(np.arange(first, end, dtype='<u4'), (rows, end-first)).reshape(-1).view('u1').reshape(-1, 4)
                key[:, 4:] = payload
                unique, counts = np.unique(key.view(f'V{4+width}').reshape(-1), return_counts=True)
                repetitions = counts - 1
                nz = np.any(unique.view('u1').reshape(-1, 4+width)[:, 4:] != 0, axis=1)
                repeated += int(repetitions.sum())
                nonzero += int(repetitions[nz].sum())
                zero += int(repetitions[~nz].sum())
                groups += int((counts > 1).sum())
                maximum = max(maximum, int(counts.max()))
        result[str(width)] = dict(positions=rows*kb*fragments, repeated=repeated,
                                  nonzero_repeated=nonzero, zero_repeated=zero,
                                  repeated_groups=groups, max_group=maximum,
                                  free_code_bytes_saved=repeated*width,
                                  # One fixed-width local dictionary index per fragment;
                                  # its lookup/indirection work is not free.
                                  fixed_index_bytes=math.ceil(rows*kb*fragments*math.ceil(math.log2(rows))/8))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, default=MODEL)
    p.add_argument('--inventory', type=Path, default=INVENTORY)
    p.add_argument('--output', type=Path, default=OUTPUT)
    p.add_argument('--first', type=int, default=0)
    p.add_argument('--count', type=int, default=250)
    p.add_argument('--summarize', action='store_true')
    a = p.parse_args()
    inv = json.loads(a.inventory.read_text())
    assert inv['metadata']['general.architecture'] == 'qwen35moe'
    assert inv['header_sha256'] == hashlib.sha256(a.model.open('rb').read(inv['header_bytes'])).hexdigest()
    align = inv['metadata'].get('general.alignment', 32)
    base = (inv['header_bytes'] + align - 1) // align * align
    tensors = [t for t in inv['tensors'] if t['type'] == 'Q8_0' and
               t['name'] != 'token_embd.weight' and '_exps.' not in t['name']]
    source_hash, inventory_hash = sha(Path(__file__)), sha(a.inventory)
    a.output.mkdir(parents=True, exist_ok=True)
    if a.summarize:
        receipts = [json.loads((a.output / f'{i:03d}.json').read_text()) for i in range(len(tensors))]
        assert all(r['name'] == t['name'] and r['source_sha256'] == source_hash and
                   r['inventory_sha256'] == inventory_hash for r, t in zip(receipts, tensors))
        widths = {w: {field: sum(r['widths'][w][field] for r in receipts)
                       for field in ('positions', 'repeated', 'nonzero_repeated', 'zero_repeated',
                                     'repeated_groups', 'free_code_bytes_saved', 'fixed_index_bytes')}
                  for w in ('4', '8')}
        for w in widths:
            widths[w]['max_group'] = max(r['widths'][w]['max_group'] for r in receipts)
        summary = dict(tensors=len(receipts), q8_bytes=sum(r['bytes'] for r in receipts),
                       complete_one_read_bytes=inv['one_token_weight_stream_bytes'], widths=widths,
                       source_sha256=source_hash, inventory_sha256=inventory_hash,
                       header_sha256=inv['header_sha256'], model_sha256=inv.get('model_sha256'),
                       tensor_receipts=[dict(name=r['name'], sha256=sha(a.output / f'{i:03d}.json'),
                                             payload_sha256=r['tensor_sha256']) for i, r in enumerate(receipts)])
        (a.output / 'receipt.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(json.dumps({k:v for k,v in summary.items() if k != 'tensor_receipts'}, indent=2))
        return
    image = np.memmap(a.model, dtype='u1', mode='r')
    for i in range(a.first, min(a.first+a.count, len(tensors))):
        t = tensors[i]
        dest = a.output / f'{i:03d}.json'
        if dest.exists():
            old = json.loads(dest.read_text())
            if old['name'] == t['name'] and old['source_sha256'] == source_hash and old['inventory_sha256'] == inventory_hash:
                continue
        kb = t['shape'][0] // 32
        assert t['shape'][0] % 32 == 0 and t['bytes'] % (34*kb) == 0
        rows = t['bytes'] // (34*kb)
        payload = image[base+t['offset']:base+t['offset']+t['bytes']]
        result = dict(name=t['name'], tensor_index=i, rows=rows, input_blocks=kb, bytes=t['bytes'],
                      tensor_sha256=hashlib.sha256(payload).hexdigest(),
                      source_sha256=source_hash, inventory_sha256=inventory_hash,
                      widths=census(payload.view(BLOCK).reshape(rows, kb)))
        dest.write_text(json.dumps(result, indent=2) + '\n')
        print(i, t['name'], result['widths']['4']['nonzero_repeated'], flush=True)


if __name__ == '__main__':
    main()
