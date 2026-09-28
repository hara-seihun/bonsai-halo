#!/usr/bin/env python3
"""Lossless per-expert Q4_K code-byte storage on actual selected routes (CPU)."""
import argparse
import hashlib
import json
import math
import zlib
from pathlib import Path

import numpy as np

ROOT = Path('../../data/qwen-moe')
MODEL = ROOT / 'Qwen3.6-35B-A3B-UD-Q4_K_M.gguf'
INVENTORY = ROOT / 'traffic.json'
CAPTURE = ROOT / 'all-producers'
OUT = ROOT / 'routed-code-entropy'


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(4 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--first', type=int, default=0)
    ap.add_argument('--count', type=int, default=40)
    ap.add_argument('--summarize', action='store_true')
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    inv = json.loads(INVENTORY.read_text())
    header = inv['header_bytes']
    with MODEL.open('rb') as f:
        assert hashlib.sha256(f.read(header)).hexdigest() == inv['header_sha256']
    src_hash = sha(Path(__file__))
    inventory_hash = sha(INVENTORY)
    capture_hash = sha(CAPTURE / 'receipt.json')
    if a.summarize:
        rows = [json.loads((OUT / f'layer-{i:02d}.json').read_text()) for i in range(40)]
        assert all(r['source_sha256'] == src_hash and r['inventory_sha256'] == inventory_hash
                   and r['capture_sha256'] == capture_hash for r in rows)
        selected = {s: [e for r in rows for e in r['selected'][s]] for s in ('train', 'held')}
        ref = inv['one_token_weight_stream_bytes']
        result = dict(source_sha256=src_hash, inventory_sha256=inventory_hash,
                      capture_sha256=capture_hash, model_sha256=json.loads((ROOT/'acquisition.json').read_text())['sha256'],
                      one_read_model_bytes=ref,
                      layers=[dict(layer=i, sha256=sha(OUT/f'layer-{i:02d}.json'),
                                   tensor_sha256=rows[i]['tensor_sha256']) for i in range(40)])
        for split, entries in selected.items():
            # On one real token, each distinct selected expert is charged once;
            # both gate/up images are independent random-access units.
            n = len(entries)
            raw = sum(e['code_bytes'] for e in entries)
            ideal = sum(e['iid_byte_entropy_bits'] for e in entries)/8
            paid = sum(e['paid_code_bytes'] for e in entries)
            assert n == 320 and raw == n*2*512*8*128
            result[split] = dict(images=n, raw_code_bytes=raw,
                                 iid_byte_ideal_bytes=ideal,
                                 iid_byte_savings=raw-ideal,
                                 iid_byte_savings_fraction_of_one_read=(raw-ideal)/ref,
                                 paid_code_bytes=paid, paid_savings=raw-paid,
                                 paid_savings_fraction_of_one_read=(raw-paid)/ref,
                                 compressed_images=sum(e['codec'] == 'zlib1' for e in entries),
                                 dormant_code_bytes=sum(e['dormant_code_bytes'] for e in entries),
                                 active_only_zlib_savings_free_index=sum(e['active_only_zlib_savings_free_index'] for e in entries),
                                 zero_byte_fraction=sum(e['zero_bytes'] for e in entries)/raw,
                                 routed_ids_sha256=sha(CAPTURE/f'{split}.layer-0.ffn_moe_topk.i32'))
        (OUT/'receipt.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps({s:result[s] for s in selected}, indent=2))
        return
    tensors = {t['name']: t for t in inv['tensors']}
    align = inv['metadata'].get('general.alignment', 32)
    base = (header+align-1)//align*align
    model = np.memmap(MODEL, dtype='u1', mode='r')
    for layer in range(a.first, min(40, a.first+a.count)):
        banks = {}
        hashes = {}
        for family in ('gate', 'up'):
            t = tensors[f'blk.{layer}.ffn_{family}_exps.weight']
            assert t['shape'] == [2048,512,256] and t['type'] == 'Q4_K'
            raw = model[base+t['offset']:base+t['offset']+t['bytes']]
            hashes[family] = hashlib.sha256(raw).hexdigest()
            banks[family] = raw.reshape(256,512,8,144)
        selected = {}
        routes = {}
        for split in ('train','held'):
            route_file = CAPTURE/f'{split}.layer-{layer}.ffn_moe_topk.i32'
            routes[split] = sha(route_file)
            ids = np.fromfile(route_file,dtype='<i4').reshape(64,8)[0]
            assert len(set(ids)) == 8 and all(0 <= i < 256 for i in ids)
            entries=[]
            for expert in ids:
                blocks = np.concatenate([banks[f][expert].reshape(-1,144) for f in ('gate','up')])
                code_matrix = blocks[:,16:]
                code = code_matrix.tobytes()
                assert len(code) == 2*512*8*128
                header = blocks[:,:4].copy().view('<u2').reshape(-1,2)
                live = np.any((header & 0x7fff) != 0,axis=1)
                active_code = code_matrix[live].tobytes()
                active_packed = zlib.compress(active_code,1)
                hist = np.bincount(np.frombuffer(code,dtype='u1'),minlength=256)
                nz = hist[hist>0]
                bits = float(-np.dot(nz, np.log2(nz/len(code))))
                compressed = zlib.compress(code,1)
                assert zlib.decompress(compressed) == code
                # 8-byte address/length record per independently addressable expert;
                # raw images also pay it, hence raw-vs-zlib choice includes just 8
                # incremental bytes plus the payload, conservatively.
                use = len(compressed)+8 < len(code)
                entries.append(dict(expert=int(expert), code_bytes=len(code), zero_bytes=int(hist[0]),
                                    dormant_code_bytes=int((~live).sum())*128,
                                    active_only_zlib_savings_free_index=len(active_code)-len(active_packed),
                                    iid_byte_entropy_bits=bits, codec='zlib1' if use else 'raw',
                                    compressed_bytes=len(compressed), paid_code_bytes=(len(compressed)+8 if use else len(code)),
                                    code_sha256=hashlib.sha256(code).hexdigest(),
                                    compressed_sha256=hashlib.sha256(compressed).hexdigest()))
            selected[split] = entries
        r = dict(layer=layer, source_sha256=src_hash, inventory_sha256=inventory_hash,
                 capture_sha256=capture_hash, tensor_sha256=hashes, route_sha256=routes,
                 selected=selected)
        (OUT/f'layer-{layer:02d}.json').write_text(json.dumps(r,indent=2)+'\n')
        print(layer, {s:sum(e['code_bytes']-e['paid_code_bytes'] for e in selected[s]) for s in selected},flush=True)


if __name__ == '__main__':
    main()
