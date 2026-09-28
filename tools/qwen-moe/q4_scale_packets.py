#!/usr/bin/env python3
"""Census same-K exact Q4_K gate/up scale/min metadata over the installed bank."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path('../../data/qwen-moe')
MODEL = ROOT / 'Qwen3.6-35B-A3B-UD-Q4_K_M.gguf'
INVENTORY = ROOT / 'traffic.json'
OUT = ROOT / 'q4-scale-packets'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def count(values, live):
    values = np.ascontiguousarray(values)
    all_keys = values.view(f'V{values.shape[-1]}').reshape(-1)
    active_keys = all_keys[live.reshape(-1)]
    return [int(len(keys) - len(np.unique(keys))) for keys in (all_keys, active_keys)]


def layer_record(layer, tensors, model, base):
    banks = []
    hashes = {}
    for family in ('gate', 'up'):
        t = tensors[f'blk.{layer}.ffn_{family}_exps.weight']
        assert t['shape'] == [2048, 512, 256] and t['type'] == 'Q4_K'
        assert t['bytes'] == 256 * 512 * 8 * 144
        raw = model[base+t['offset']:base+t['offset']+t['bytes']]
        hashes[family] = hashlib.sha256(raw).hexdigest()
        banks.append(raw.reshape(256, 512, 8, 144))
    # Same input block K only: unlike a cross-K match, a metadata match here
    # could potentially share lookup/decode for one activation block.
    records = []
    for k in range(8):
        metadata = np.stack([bank[:, :, k, :16] for bank in banks])
        header = np.ascontiguousarray(metadata[..., :4])
        # d and dmin are bit patterns, including signed zero. A block is
        # called dormant only when BOTH absolute FP16 values are bit-zero.
        live = np.any((header.view('<u2').reshape(2, 256, 512, 2) & 0x7fff) != 0, axis=-1)
        records.append(dict(k=k, active=int(live.sum()), dormant=int((~live).sum()),
                            metadata16_repeats=count(metadata, live),
                            scales12_repeats=count(metadata[..., 4:16], live),
                            header4_repeats=count(header, live)))
    return dict(layer=layer, tensor_sha256=hashes, blocks=2*256*512*8, positions=records)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--first', type=int, default=0)
    p.add_argument('--count', type=int, default=40)
    p.add_argument('--summarize', action='store_true')
    a = p.parse_args()
    OUT.mkdir(exist_ok=True)
    inv = json.loads(INVENTORY.read_text())
    assert inv['metadata']['general.architecture'] == 'qwen35moe'
    with MODEL.open('rb') as f:
        assert hashlib.sha256(f.read(inv['header_bytes'])).hexdigest() == inv['header_sha256']
    src = sha(Path(__file__))
    inventory = sha(INVENTORY)
    if a.summarize:
        rows = [json.loads((OUT / f'layer-{i:02d}.json').read_text()) for i in range(40)]
        assert all(r['source_sha256'] == src and r['inventory_sha256'] == inventory
                   and r['layer'] == i for i, r in enumerate(rows))
        totals = {key: [sum(p[key][j] for r in rows for p in r['positions']) for j in (0,1)]
                  for key in ('metadata16_repeats', 'scales12_repeats', 'header4_repeats')}
        totals['active_blocks'] = sum(p['active'] for r in rows for p in r['positions'])
        totals['dormant_blocks'] = sum(p['dormant'] for r in rows for p in r['positions'])
        totals['all_blocks'] = sum(r['blocks'] for r in rows)
        assert totals['all_blocks'] == totals['active_blocks'] + totals['dormant_blocks']
        assert totals['metadata16_repeats'][1] <= totals['scales12_repeats'][1]
        assert totals['metadata16_repeats'][1] <= totals['header4_repeats'][1]
        reference = inv['one_token_weight_stream_bytes']
        n = 2 * 256 * 512
        raw_headers = 4 * totals['all_blocks']
        paid_by_layer = [sum(4 * (n - p['header4_repeats'][0]) +
                             math.ceil(n * ((n - p['header4_repeats'][0] - 1).bit_length()) / 8) + 8
                             for p in r['positions']) for r in rows]
        totals['raw_header_bytes'] = raw_headers
        totals['fixed_width_same_k_header_bytes'] = sum(paid_by_layer)
        totals['fixed_width_same_k_growth_bytes'] = sum(paid_by_layer) - raw_headers
        totals['free_per_layer_hybrid_saved_bytes'] = sum(max(0, 4*n*8 - paid) for paid in paid_by_layer)
        totals['per_layer_paid_header_bytes'] = paid_by_layer
        # These are free independent-portion deletion bounds, not additive:
        # the same bytes can be charged by the full tuple and its subfields.
        totals['free_active_header_bytes_per_token'] = 4 * totals['header4_repeats'][1] * 8 / 256
        totals['free_active_header_fraction'] = totals['free_active_header_bytes_per_token'] / reference
        acquisition = ROOT / 'acquisition.json'
        receipt = dict(source_sha256=src, inventory_sha256=inventory,
                       acquisition_sha256=sha(acquisition),
                       model_sha256=json.loads(acquisition.read_text())['sha256'],
                       header_sha256=inv['header_sha256'],
                       one_read_model_bytes=reference, totals=totals,
                       layers=[dict(layer=i, receipt_sha256=sha(OUT / f'layer-{i:02d}.json'),
                                    tensor_sha256=rows[i]['tensor_sha256']) for i in range(40)])
        (OUT / 'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
        print(json.dumps(totals, indent=2))
        return
    tensors = {t['name']: t for t in inv['tensors']}
    align = inv['metadata'].get('general.alignment', 32)
    base = (inv['header_bytes'] + align-1)//align*align
    model = np.memmap(MODEL, dtype='u1', mode='r')
    for layer in range(a.first, min(40, a.first+a.count)):
        r = layer_record(layer, tensors, model, base)
        r.update(source_sha256=src, inventory_sha256=inventory)
        (OUT / f'layer-{layer:02d}.json').write_text(json.dumps(r, indent=2)+'\n')
        print(layer, sum(p['metadata16_repeats'][1] for p in r['positions']),
              sum(p['header4_repeats'][1] for p in r['positions']), flush=True)


if __name__ == '__main__':
    main()
