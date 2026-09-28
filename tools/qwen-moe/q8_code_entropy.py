#!/usr/bin/env python3
"""Paid independently decodable Q8_0 code pages on the installed ordinary bank."""
import argparse
import hashlib
import json
import math
import struct
import zlib
from pathlib import Path

import numpy as np

MODEL = Path('../../data/qwen-moe/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf')
INVENTORY = Path('../../data/qwen-moe/traffic.json')
OUTPUT = Path('../../data/qwen-moe/q8-code-pages')
ACQUISITION = Path('../../data/qwen-moe/acquisition.json')
PAGE = 16384  # exactly 512 Q8 blocks; the final tensor page may be shorter


def digest(data):
    return hashlib.sha256(data).hexdigest()


def tensors_of(inv):
    return [t for t in inv['tensors'] if t['type'] == 'Q8_0' and t['name'] != 'token_embd.weight' and '_exps.' not in t['name']]


def encode(codes):
    pages = []
    lengths = []
    counts = np.zeros(256, dtype=np.int64)
    raw_pages = 0
    for start in range(0, len(codes), PAGE):
        raw = codes[start:start + PAGE]
        compressed = zlib.compress(raw, 1)
        is_raw = len(compressed) >= len(raw)
        selected = raw if is_raw else compressed
        pages.append(selected)
        lengths.append(len(selected) | (0x80000000 if is_raw else 0))
        raw_pages += is_raw
        counts += np.bincount(np.frombuffer(raw, dtype=np.uint8), minlength=256)
    image = struct.pack('<I', len(lengths)) + struct.pack('<' + 'I'*len(lengths), *lengths) + b''.join(pages)
    return image, counts, raw_pages


def decode(image, scales, original_length):
    n, = struct.unpack_from('<I', image)
    lengths = struct.unpack_from('<' + 'I'*n, image, 4)
    offset = 4 + 4*n
    out = hashlib.sha256()
    reconstructed = hashlib.sha256()
    assert len(scales)*16 == original_length
    for i, tagged in enumerate(lengths):
        size = tagged & 0x7fffffff
        encoded = image[offset:offset+size]
        assert len(encoded) == size
        raw = encoded if tagged >> 31 else zlib.decompress(encoded)
        assert len(raw) == min(PAGE, original_length-i*PAGE)
        out.update(raw)
        nblocks = len(raw)//32
        original = np.empty((nblocks,34),dtype=np.uint8)
        original[:,:2] = np.frombuffer(scales, dtype=np.uint8, count=nblocks*2, offset=i*(PAGE//32)*2).reshape(nblocks,2)
        original[:,2:] = np.frombuffer(raw, dtype=np.uint8).reshape(nblocks,32)
        reconstructed.update(original)
        offset += size
    assert offset == len(image)
    assert n == (original_length+PAGE-1)//PAGE
    return out.hexdigest(), reconstructed.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--first', type=int, default=0)
    p.add_argument('--count', type=int, default=250)
    p.add_argument('--summarize', action='store_true')
    p.add_argument('--model', type=Path, default=MODEL)
    p.add_argument('--inventory', type=Path, default=INVENTORY)
    p.add_argument('--output', type=Path, default=OUTPUT)
    a = p.parse_args()
    inv = json.loads(a.inventory.read_text())
    acquisition = json.loads(ACQUISITION.read_text())
    assert acquisition['verified'] and a.model.resolve() == Path(acquisition['path']).resolve()
    assert a.model.stat().st_size == acquisition['size']
    assert inv['metadata']['general.architecture'] == 'qwen35moe'
    assert inv['header_sha256'] == digest(a.model.open('rb').read(inv['header_bytes']))
    align = inv['metadata'].get('general.alignment', 32)
    base = (inv['header_bytes'] + align - 1)//align*align
    tensors = tensors_of(inv)
    source_sha = digest(Path(__file__).read_bytes())
    inventory_sha = digest(a.inventory.read_bytes())
    a.output.mkdir(parents=True, exist_ok=True)
    if a.summarize:
        records = [json.loads((a.output/f'{i:03d}.json').read_text()) for i in range(len(tensors))]
        assert all(r['name'] == t['name'] and r['source_sha256'] == source_sha and r['inventory_sha256'] == inventory_sha for r,t in zip(records,tensors))
        for i,r in enumerate(records):
            assert digest((a.output/f'{i:03d}.pages').read_bytes()) == r['pages_sha256']
            assert digest((a.output/f'{i:03d}.scales').read_bytes()) == r['scales_sha256']
        totals = {k:sum(r[k] for r in records) for k in ('blocks','raw_code_bytes','page_bytes','raw_pages','pages','ideal_iid_bits','zero_code_bytes')}
        totals['scale_bytes'] = 2*totals['blocks']
        totals['saved_page_bytes'] = totals['raw_code_bytes']-totals['page_bytes']
        totals['saved_complete_fraction'] = totals['saved_page_bytes']/inv['one_token_weight_stream_bytes']
        totals['ideal_iid_saved_bytes'] = totals['raw_code_bytes']-totals['ideal_iid_bits']/8
        totals['additional_free_iid_bytes'] = totals['page_bytes']-totals['ideal_iid_bits']/8
        totals['model_one_read_bytes'] = inv['one_token_weight_stream_bytes']
        receipt = dict(summary=totals, model_sha256=acquisition['sha256'], header_sha256=inv['header_sha256'],
                       acquisition_sha256=digest(ACQUISITION.read_bytes()),
                       source_sha256=source_sha, inventory_sha256=inventory_sha,
                       tensors=[dict(name=r['name'], receipt_sha256=digest((a.output/f'{i:03d}.json').read_bytes()),
                                     payload_sha256=r['payload_sha256'], pages_sha256=r['pages_sha256'],
                                     scales_sha256=r['scales_sha256']) for i,r in enumerate(records)])
        (a.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
        print(json.dumps(totals,indent=2))
        return
    image = np.memmap(a.model, mode='r', dtype=np.uint8)
    for i in range(a.first, min(a.first+a.count,len(tensors))):
        t = tensors[i]
        result_path, pages_path, scales_path = a.output/f'{i:03d}.json', a.output/f'{i:03d}.pages', a.output/f'{i:03d}.scales'
        if result_path.exists() and pages_path.exists() and scales_path.exists():
            r = json.loads(result_path.read_text())
            if r['source_sha256'] == source_sha and r['inventory_sha256'] == inventory_sha and r['name'] == t['name'] and digest(pages_path.read_bytes()) == r['pages_sha256'] and digest(scales_path.read_bytes()) == r['scales_sha256']:
                continue
        payload = image[base+t['offset']:base+t['offset']+t['bytes']]
        assert len(payload)%34 == 0 and t['shape'][0]%32 == 0
        blocks = payload.reshape(-1,34)
        codes = blocks[:,2:].copy().tobytes()
        scales = blocks[:,:2].copy().tobytes()
        packed, counts, raw_pages = encode(codes)
        code_sha = digest(codes)
        assert decode(packed,scales,len(codes)) == (code_sha,digest(payload))
        positive = counts[counts>0]
        ideal_bits = float(len(codes)*math.log2(len(codes))-np.dot(positive.astype(float),np.log2(positive)))
        result = dict(name=t['name'], shape=t['shape'], blocks=len(payload)//34,
                      raw_code_bytes=len(codes), page_bytes=len(packed), pages=(len(codes)+PAGE-1)//PAGE,
                      raw_pages=raw_pages, ideal_iid_bits=ideal_bits, zero_code_bytes=int(counts[0]),
                      payload_sha256=digest(payload), codes_sha256=code_sha, pages_sha256=digest(packed),
                      scales_sha256=digest(scales),
                      source_sha256=source_sha, inventory_sha256=inventory_sha)
        pages_path.write_bytes(packed)
        scales_path.write_bytes(scales)
        result_path.write_text(json.dumps(result,indent=2)+'\n')
        print(i,t['name'],len(codes)-len(packed),flush=True)


if __name__ == '__main__':
    main()
