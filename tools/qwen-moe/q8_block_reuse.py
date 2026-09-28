#!/usr/bin/env python3
"""Count same-input-block reuse among installed nonembedding Q8_0 output rows."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np

MODEL = Path('../../data/qwen-moe/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf')
INVENTORY = Path('../../data/qwen-moe/traffic.json')
OUTPUT = Path('../../data/qwen-moe/q8-block-reuse')
BLOCK = np.dtype([('scale', '<u2'), ('codes', 'u1', (32,))])


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def census(b, chunk=16):
    rows, kb = b.shape
    result = dict(blocks=rows * kb, rows=rows, input_blocks=kb, full_saved=0,
                  code_saved=0, full_max_group=0, code_max_group=0,
                  full_repeated_groups=0, code_repeated_groups=0, nonzero_code_saved=0)
    for start in range(0, kb, chunk):
        width = min(chunk, kb - start)
        part = np.ascontiguousarray(b[:, start:start + width])
        index = np.broadcast_to(np.arange(start, start + width, dtype='<u4'), (rows, width)).reshape(-1)
        # Include the input-block coordinate: only equal K positions see the same activation codes.
        for name, payload, nbytes in (('full', part.view('u1').reshape(-1, 34), 34),
                                       ('code', part['codes'].reshape(-1, 32), 32)):
            key = np.empty((rows * width, 4 + nbytes), dtype='u1')
            key[:, :4] = index.view('u1').reshape(-1, 4)
            key[:, 4:] = payload
            unique, counts = np.unique(key.view(np.dtype(('V', key.shape[1]))).reshape(-1), return_counts=True)
            result[name + '_saved'] += int(np.maximum(counts - 1, 0).sum())
            if name == 'code':
                repeated = counts > 1
                codes = unique[repeated].view('u1').reshape(-1, 4 + nbytes)[:, 4:]
                result['nonzero_code_saved'] += int((counts[repeated] - 1)[np.any(codes != 0, axis=1)].sum())
            result[name + '_repeated_groups'] += int((counts > 1).sum())
            result[name + '_max_group'] = max(result[name + '_max_group'], int(counts.max()))
    return result


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
    if a.summarize:
        results = [json.loads((a.output / f'{i:03d}.json').read_text()) for i in range(len(tensors))]
        assert all(r['name'] == t['name'] and r['source_sha256'] == sha(Path(__file__)) and
                   r['inventory_sha256'] == sha(a.inventory) for r, t in zip(results, tensors))
        fields = ('blocks', 'full_saved', 'code_saved', 'nonzero_code_saved',
                  'full_repeated_groups', 'code_repeated_groups',
                  'full_fixed_index_net_bytes', 'code_fixed_index_net_bytes')
        summary = {field: sum(r[field] for r in results) for field in fields}
        summary.update(tensors=len(results), q8_bytes=sum(r['bytes'] for r in results),
                       complete_one_read_bytes=inv['one_token_weight_stream_bytes'],
                       source_sha256=sha(Path(__file__)), inventory_sha256=sha(a.inventory),
                       header_sha256=inv['header_sha256'], model_sha256=inv.get('model_sha256'),
                       tensor_receipts=[dict(name=r['name'], sha256=sha(a.output / f'{i:03d}.json'),
                                             payload_sha256=r['tensor_sha256']) for i, r in enumerate(results)])
        (a.output / 'receipt.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(json.dumps({k: v for k, v in summary.items() if k != 'tensor_receipts'}, indent=2))
        return
    image = np.memmap(a.model, dtype='u1', mode='r')
    for i in range(a.first, min(a.first + a.count, len(tensors))):
        t = tensors[i]
        destination = a.output / f'{i:03d}.json'
        if destination.exists():
            old = json.loads(destination.read_text())
            if old['name'] == t['name'] and old['source_sha256'] == sha(Path(__file__)):
                continue
        kb = t['shape'][0] // 32
        assert t['shape'][0] % 32 == 0 and t['bytes'] % (34 * kb) == 0
        rows = t['bytes'] // (34 * kb)
        payload = image[base + t['offset']:base + t['offset'] + t['bytes']]
        b = payload.view(BLOCK).reshape(rows, kb)
        result = census(b)
        result.update(name=t['name'], shape=t['shape'], bytes=t['bytes'],
                      tensor_sha256=hashlib.sha256(payload).hexdigest(),
                      source_sha256=sha(Path(__file__)), inventory_sha256=sha(a.inventory),
                      header_sha256=inv['header_sha256'], model_sha256=inv.get('model_sha256'),
                      tensor_index=i)
        # Fixed-width row-local dictionary indices, even granting free dictionary lookup.
        # One dictionary per input block, scale retained for code-only reuse.
        result['full_fixed_index_net_bytes'] = result['full_saved'] * 34 - math.ceil(rows * kb * math.ceil(math.log2(rows)) / 8)
        result['code_fixed_index_net_bytes'] = result['code_saved'] * 32 - math.ceil(rows * kb * math.ceil(math.log2(rows)) / 8)
        destination.write_text(json.dumps(result, indent=2) + '\n')
        print(i, t['name'], result['full_saved'], result['code_saved'], flush=True)


if __name__ == '__main__':
    main()
