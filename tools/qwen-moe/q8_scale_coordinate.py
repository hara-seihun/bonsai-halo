#!/usr/bin/env python3
"""Census exact modal/exception Q8_0 scale codes with paid random-access ranks."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

MODEL = Path('../../data/qwen-moe/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf')
INVENTORY = Path('../../data/qwen-moe/traffic.json')
ACQUISITION = Path('../../data/qwen-moe/acquisition.json')
OUT = Path('../../data/qwen-moe/q8-scale-coordinate')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def census(scales):
    rows, kb = scales.shape
    totals = dict(columns=kb, blocks=rows*kb, fixed_payload_bytes=0,
                  paid_payload_bytes=0, free_rank_payload_bytes=0,
                  paid_winning_columns=0, free_rank_winning_columns=0,
                  max_mode_count=0, max_mode_fraction=0, min_paid_deficit=2**63,
                  min_free_rank_deficit=2**63)
    for k in range(kb):
        _, counts = np.unique(scales[:, k], return_counts=True)
        distinct = len(counts)
        mode_count = int(counts.max())
        other = rows - mode_count
        fixed_bits = (distinct-1).bit_length()
        other_bits = (distinct-2).bit_length() if distinct > 1 else 0
        fixed = 2*distinct + (rows*fixed_bits+7)//8
        # Exact bitmap marks exceptions. A directory of 16-bit ranks at each
        # 32-row chunk gives independent O(1) lookup: prefix + popcount within
        # the chunk, then the fixed-width nonmodal index and FP16 dictionary.
        free_rank = 2*distinct + (rows+7)//8 + (other*other_bits+7)//8
        paid = free_rank + 2*((rows+31)//32)
        totals['fixed_payload_bytes'] += fixed
        totals['free_rank_payload_bytes'] += min(fixed, free_rank)
        totals['paid_payload_bytes'] += min(fixed, paid)
        totals['paid_winning_columns'] += paid < fixed
        totals['free_rank_winning_columns'] += free_rank < fixed
        totals['max_mode_count'] = max(totals['max_mode_count'], mode_count)
        totals['max_mode_fraction'] = max(totals['max_mode_fraction'], mode_count/rows)
        totals['min_paid_deficit'] = min(totals['min_paid_deficit'], paid-fixed)
        totals['min_free_rank_deficit'] = min(totals['min_free_rank_deficit'], free_rank-fixed)
    return totals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--first', type=int, default=0)
    parser.add_argument('--count', type=int, default=250)
    args = parser.parse_args()
    inv = json.loads(INVENTORY.read_text())
    acquisition = json.loads(ACQUISITION.read_text())
    assert inv['metadata']['general.architecture'] == 'qwen35moe'
    assert MODEL.stat().st_size == acquisition['size']
    assert hashlib.sha256(MODEL.open('rb').read(inv['header_bytes'])).hexdigest() == inv['header_sha256']
    align = inv['metadata'].get('general.alignment', 32)
    base = (inv['header_bytes']+align-1)//align*align
    tensors = [t for t in inv['tensors'] if t['type'] == 'Q8_0' and t['name'] != 'token_embd.weight' and '_exps.' not in t['name']]
    OUT.mkdir(parents=True, exist_ok=True)
    image = np.memmap(MODEL, mode='r', dtype='u1')
    source_hash = sha(Path(__file__))
    for i in range(args.first, min(args.first+args.count, len(tensors))):
        t = tensors[i]
        target = OUT/f'{i:03d}.json'
        if target.exists() and json.loads(target.read_text()).get('source_sha256') == source_hash:
            continue
        payload = image[base+t['offset']:base+t['offset']+t['bytes']]
        kb = t['shape'][0]//32
        assert t['shape'][0] % 32 == 0 and t['bytes'] % (34*kb) == 0
        rows = t['bytes']//(34*kb)
        scales = payload.reshape(-1, 34)[:, :2].copy().view('<u2').reshape(rows, kb)
        result = census(scales)
        result.update(name=t['name'], rows=rows, input_blocks=kb,
                      payload_sha256=hashlib.sha256(payload).hexdigest(), source_sha256=source_hash)
        target.write_text(json.dumps(result, indent=2)+'\n')
        print(i, t['name'], result['paid_winning_columns'], flush=True)
    if args.first == 0 and args.count >= len(tensors):
        results = [json.loads((OUT/f'{i:03d}.json').read_text()) for i in range(len(tensors))]
        fields = ('columns', 'blocks', 'fixed_payload_bytes', 'paid_payload_bytes',
                  'free_rank_payload_bytes', 'paid_winning_columns', 'free_rank_winning_columns')
        sums = {f: sum(r[f] for r in results) for f in fields}
        sums['max_mode_count'] = max(r['max_mode_count'] for r in results)
        sums['max_mode_fraction'] = max(r['max_mode_fraction'] for r in results)
        sums['min_paid_deficit'] = min(r['min_paid_deficit'] for r in results)
        sums['min_free_rank_deficit'] = min(r['min_free_rank_deficit'] for r in results)
        sums['directory_bytes'] = 8*sums['columns']
        sums['paid_saved_bytes'] = sums['fixed_payload_bytes']-sums['paid_payload_bytes']
        sums['free_rank_saved_bytes'] = sums['fixed_payload_bytes']-sums['free_rank_payload_bytes']
        sums['complete_stream_fraction'] = sums['paid_saved_bytes']/inv['one_token_weight_stream_bytes']
        existing = json.loads(Path('../../data/qwen-moe/q8-scale-sharing/receipt.json').read_text())
        assert sums['fixed_payload_bytes']+sums['directory_bytes'] == existing['summary']['packed_scale_bytes']
        receipt = dict(summary=sums, acquisition_sha256=sha(ACQUISITION), model_sha256=acquisition['sha256'],
                       inventory_sha256=sha(INVENTORY), header_sha256=inv['header_sha256'], source_sha256=source_hash,
                       tensors=[dict(name=r['name'], receipt_sha256=sha(OUT/f'{i:03d}.json'), payload_sha256=r['payload_sha256']) for i,r in enumerate(results)])
        (OUT/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
        print(json.dumps(sums, indent=2))


if __name__ == '__main__':
    main()
