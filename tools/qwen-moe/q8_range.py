#!/usr/bin/env python3
"""Lossless signed-bit-width census of installed nonexpert Q8_0 blocks; CPU only."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

MODEL = Path('../../data/qwen-moe/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf')
INVENTORY = Path('../../data/qwen-moe/traffic.json')
OUTPUT = Path('../../data/qwen-moe/q8-range')
BLOCK = np.dtype([('scale', '<u2'), ('codes', 'i1', (32,))])


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    tensors = [t for t in inv['tensors'] if t['type'] == 'Q8_0' and t['name'] != 'token_embd.weight' and '_exps.' not in t['name']]
    a.output.mkdir(parents=True, exist_ok=True)
    source_hash = sha(Path(__file__))
    inventory_hash = sha(a.inventory)
    if a.summarize:
        results = [json.loads((a.output / f'{i:03d}.json').read_text()) for i in range(len(tensors))]
        assert all(r['name'] == t['name'] and r['source_sha256'] == source_hash and r['inventory_sha256'] == inventory_hash for r, t in zip(results, tensors))
        totals = {k: sum(r[k] for r in results) for k in ('blocks', 'rows', 'bytes')}
        widths = {str(w): sum(r['fit'][str(w)] for r in results) for w in range(4, 9)}
        exceptions = {str(w): [sum(r['exceptions'][str(w)][e] for r in results) for e in range(33)] for w in range(4, 8)}
        n = totals['blocks']
        # A best-case combinatorial support index: every block gets its exception
        # count and bit width for free. 32 low-w bits, one indexed support set,
        # and the high (8-w) bits of each exceptional signed byte.
        def rank_bits(e):
            return (math.comb(32, e) - 1).bit_length()
        # Per-block joint minimum cannot be reconstructed from marginal histograms;
        # report each fixed-width lower-cost family independently.
        ideal = {str(w): sum(min(256, 32*w + rank_bits(e) + (8-w)*e) * count
                             for e, count in enumerate(exceptions[str(w)]))
                 for w in range(4, 8)}
        # 1 tag bit/block, four-bit codes for fit blocks, otherwise original eight-bit codes.
        # Exact scale bits retained; tag and row offset overhead paid, decoder work not free.
        eligible = widths['4']
        saved_code = 16 * eligible
        tag_bytes = (n + 7) // 8
        # 64-bit row offset addresses a >4GiB installed payload; real compact image can
        # use per-tensor 32-bit offsets, since every Q8 tensor fits under 4GiB.
        offset_bytes = 4 * totals['rows']
        saved = saved_code - tag_bytes - offset_bytes
        totals.update(tensors=len(results), fit=widths, exceptions=exceptions,
                      free_exception_index_bits=ideal,
                      free_exception_index_saved_bytes={w: (256*n - bits)/8 for w, bits in ideal.items()},
                      narrow4_fraction=eligible / n,
                      scale_bytes=2 * n, code_bytes=32 * n, four_eight_tag_bytes=tag_bytes,
                      row_offset_bytes=offset_bytes, four_eight_saved_bytes=saved,
                      four_eight_fraction_q8=saved / totals['bytes'],
                      four_eight_fraction_complete=saved / inv['one_token_weight_stream_bytes'])
        receipt = dict(summary=totals, source_sha256=source_hash, inventory_sha256=inventory_hash,
                       header_sha256=inv['header_sha256'], model_sha256=inv.get('model_sha256'),
                       per_tensor=[dict(name=r['name'], receipt_sha256=sha(a.output / f'{i:03d}.json'),
                                        payload_sha256=r['payload_sha256']) for i, r in enumerate(results)])
        (a.output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
        print(json.dumps(totals, indent=2))
        return
    image = np.memmap(a.model, dtype='u1', mode='r')
    for i in range(a.first, min(a.first + a.count, len(tensors))):
        t = tensors[i]
        destination = a.output / f'{i:03d}.json'
        if destination.exists():
            old = json.loads(destination.read_text())
            if old['name'] == t['name'] and old['source_sha256'] == source_hash and old['inventory_sha256'] == inventory_hash:
                continue
        k = t['shape'][0]
        assert k % 32 == 0 and t['bytes'] % (34 * (k // 32)) == 0
        rows = t['bytes'] // (34 * (k // 32))
        payload = image[base + t['offset']:base + t['offset'] + t['bytes']]
        blocks = payload.view(BLOCK)
        fit = {str(w): 0 for w in range(4, 9)}
        exceptions = {str(w): [0]*33 for w in range(4, 8)}
        # Chunking bounds temporary bool arrays independently of giant tensors.
        for start in range(0, len(blocks), 262144):
            q = blocks[start:start + 262144]['codes']
            minimum = q.min(axis=1)
            maximum = q.max(axis=1)
            for w in range(4, 9):
                fit[str(w)] += int(np.count_nonzero((minimum >= -(1 << (w - 1))) & (maximum < (1 << (w - 1)))))
                if w < 8:
                    exceptional = np.count_nonzero((q < -(1 << (w - 1))) | (q >= (1 << (w - 1))), axis=1)
                    h = np.bincount(exceptional, minlength=33)
                    exceptions[str(w)] = [x + int(y) for x, y in zip(exceptions[str(w)], h)]
        result = dict(name=t['name'], shape=t['shape'], bytes=t['bytes'], rows=rows,
                      blocks=len(blocks), fit=fit, exceptions=exceptions,
                      payload_sha256=hashlib.sha256(payload).hexdigest(),
                      source_sha256=source_hash, inventory_sha256=inventory_hash, header_sha256=inv['header_sha256'])
        destination.write_text(json.dumps(result, indent=2) + '\n')
        print(i, t['name'], fit['4'], fit['5'], fit['6'], fit['7'], flush=True)


if __name__ == '__main__':
    main()
