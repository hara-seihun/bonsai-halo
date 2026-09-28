#!/usr/bin/env python3
"""Exact row-local adjacent-K Q8_0 scale page image and paid rate census."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

BASE = Path('../../data/qwen-moe')
MODEL = BASE / 'Qwen3.6-35B-A3B-UD-Q4_K_M.gguf'
INV = BASE / 'traffic.json'
OUT = BASE / 'q8-scale-adjacent'
PAGE = 16


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def image_page(values):
    """Byte header (bit width), 16-bit anchor, little-endian fixed-width XOR deltas."""
    anchor = int(values[0])
    deltas = [int(v) ^ anchor for v in values[1:]]
    width = max((v.bit_length() for v in deltas), default=0)
    packed = sum(v << (i * width) for i, v in enumerate(deltas))
    data = bytes((width,)) + anchor.to_bytes(2, 'little') + packed.to_bytes((len(deltas)*width+7)//8, 'little')
    return data


def check_page(data, values):
    width = data[0]
    anchor = int.from_bytes(data[1:3], 'little')
    packed = int.from_bytes(data[3:], 'little')
    reconstructed = [anchor] + [anchor ^ ((packed >> (i*width)) & ((1 << width)-1)) for i in range(len(values)-1)]
    assert reconstructed == [int(v) for v in values]


def tensor(scales, output, write_image):
    rows, kb = scales.shape
    # Row-major page directory makes every scale addressable without scanning
    # previous rows; 32-bit absolute offsets, including one terminal offset.
    pages = (kb + PAGE - 1) // PAGE
    words = scales.reshape(rows, kb)
    widths = np.zeros((rows, pages), np.uint8)
    for p in range(pages):
        v = words[:, p*PAGE:min((p+1)*PAGE, kb)]
        xor = np.bitwise_xor(v[:, 1:], v[:, :1])
        # Exact bit width of unsigned 16-bit XOR; zero gets width zero.
        for bit in range(15, -1, -1):
            widths[:, p] = np.where((widths[:, p] == 0) & np.any(xor & (1 << bit), axis=1), bit+1, widths[:, p])
    count = np.array([min(PAGE, kb - p*PAGE)-1 for p in range(pages)], np.int64)
    lengths = 3 + (widths.astype(np.int64)*count[None, :] + 7)//8
    payload_bytes = int(lengths.sum())
    directory_bytes = 4*(rows*pages+1)
    assert directory_bytes + payload_bytes < 2**32
    if write_image:
        offsets = np.empty(rows*pages+1, '<u4')
        offsets[0] = directory_bytes
        offsets[1:] = directory_bytes + np.cumsum(lengths.ravel(), dtype=np.uint32)
        with open(output, 'wb') as f:
            f.write(offsets.tobytes())
            for row in range(rows):
                for p in range(pages):
                    v = words[row, p*PAGE:min((p+1)*PAGE, kb)]
                    data = image_page(v)
                    assert len(data) == lengths[row, p]
                    check_page(data, v)
                    f.write(data)
        assert output.stat().st_size == directory_bytes+payload_bytes
    return dict(rows=rows, input_blocks=kb, blocks=rows*kb, pages=rows*pages,
                payload_bytes=payload_bytes, directory_bytes=directory_bytes,
                paid_bytes=payload_bytes+directory_bytes,
                width_histogram={str(i):int((widths==i).sum()) for i in range(17)},
                image_sha256=sha(output) if write_image else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--first', type=int, default=0)
    ap.add_argument('--count', type=int, default=250)
    ap.add_argument('--image', action='store_true', help='save and decode-check every page')
    ap.add_argument('--summarize', action='store_true')
    args = ap.parse_args()
    inv = json.loads(INV.read_text())
    assert inv['metadata']['general.architecture'] == 'qwen35moe'
    assert hashlib.sha256(MODEL.open('rb').read(inv['header_bytes'])).hexdigest() == inv['header_sha256']
    base = (inv['header_bytes']+31)//32*32
    tensors = [t for t in inv['tensors'] if t['type'] == 'Q8_0' and t['name'] != 'token_embd.weight' and '_exps.' not in t['name']]
    OUT.mkdir(parents=True, exist_ok=True)
    source_sha, inventory_sha = sha(Path(__file__)), sha(INV)
    if args.summarize:
        receipts = [json.loads((OUT/f'{i:03d}.json').read_text()) for i in range(len(tensors))]
        assert all(r['name']==t['name'] and r['source_sha256']==source_sha and r['inventory_sha256']==inventory_sha for r,t in zip(receipts,tensors))
        totals = {field:sum(r[field] for r in receipts) for field in ('blocks','pages','payload_bytes','directory_bytes','paid_bytes')}
        totals['scale_bytes'] = totals['blocks']*2
        totals['dictionary_paid_bytes'] = json.loads((BASE/'q8-scale-sharing/receipt.json').read_text())['summary']['packed_scale_bytes']
        totals['best_of_dictionary_and_adjacent_bytes'] = sum(min(r['paid_bytes'], json.loads((BASE/'q8-scale-sharing'/f'{i:03d}.json').read_text())['packed_bytes']) for i,r in enumerate(receipts))
        totals['conditional_complete_one_read_bytes'] = inv['one_token_weight_stream_bytes']
        totals['adjacent_winning_tensors'] = sum(r['paid_bytes'] < json.loads((BASE/'q8-scale-sharing'/f'{i:03d}.json').read_text())['packed_bytes'] for i,r in enumerate(receipts))
        result = dict(contract='Exact 16-consecutive-input-block page: first FP16 scale bits plus fixed-width unsigned XOR from page anchor, one byte width per page, 32-bit random-access page offsets including terminal; unchanged Q8 code bytes. Each scale needs offset lookup, anchor and shift/mask/XOR. CPU image, not native FP32 or TPS.',
                      summary=totals, source_sha256=source_sha, inventory_sha256=inventory_sha,
                      header_sha256=inv['header_sha256'], model_sha256=inv.get('model_sha256'),
                      tensors=[dict(name=r['name'], receipt_sha256=sha(OUT/f'{i:03d}.json'), payload_sha256=r['payload_sha256'], image_sha256=r['image_sha256']) for i,r in enumerate(receipts)])
        (OUT/'receipt.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(totals, indent=2))
        return
    image = np.memmap(MODEL, dtype='u1', mode='r')
    for i in range(args.first, min(args.first+args.count, len(tensors))):
        t = tensors[i]
        kb = t['shape'][0]//32
        assert t['shape'][0]%32==0 and t['bytes']%(34*kb)==0
        rows = t['bytes']//(34*kb)
        payload = image[base+t['offset']:base+t['offset']+t['bytes']]
        scales = payload.reshape(-1,34)[:,:2].copy().view('<u2').reshape(rows,kb)
        image_path = OUT/f'{i:03d}.scales'
        r = tensor(scales, image_path, args.image)
        r.update(name=t['name'], payload_sha256=hashlib.sha256(payload).hexdigest(),
                 source_sha256=source_sha, inventory_sha256=inventory_sha)
        (OUT/f'{i:03d}.json').write_text(json.dumps(r,indent=2)+'\n')
        print(i, t['name'], r['paid_bytes'], flush=True)


if __name__ == '__main__':
    main()
