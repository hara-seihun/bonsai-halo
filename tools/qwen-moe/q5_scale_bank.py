#!/usr/bin/env python3
"""Exact paid per-layer Q5_K expert superblock-scale coordinate, CPU only.

The 4-byte (d,dmin) pair is replaced by one fixed-width packed index into
independent FP16-word dictionaries. Original 172 non-scale bytes remain in place.
"""
import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np

MODEL = Path('../../data/qwen-moe/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf')
INVENTORY = Path('../../data/qwen-moe/traffic.json')
OUTPUT = Path('../../data/qwen-moe/q5-scale-bank')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encode(scales):
    words = scales.reshape(-1, 2)
    dictionaries = []
    codes = []
    widths = []
    for column in range(2):
        dictionary, indices = np.unique(words[:, column], return_inverse=True)
        dictionaries.append(dictionary.astype('<u2'))
        codes.append(indices.astype('u4'))
        widths.append((len(dictionary) - 1).bit_length())
    bits = sum(widths)
    combined = (codes[0] << widths[1]) | codes[1]
    masks = np.arange(bits - 1, -1, -1, dtype='u4')
    payload = np.packbits(((combined[:, None] >> masks) & 1).astype('u1').reshape(-1)).tobytes()
    # Fixed 32-byte header: count, two dictionary lengths, two index widths,
    # and the size of the bitstream. The paired dictionary follows immediately.
    header = struct.pack('<8sIIIIII', b'Q5SCALE1', len(words), len(dictionaries[0]),
                         len(dictionaries[1]), widths[0], widths[1], len(payload))
    return header + b''.join(d.tobytes() for d in dictionaries) + payload, widths


def decode(image):
    magic, count, na, nb, wa, wb, size = struct.unpack_from('<8sIIIIII', image)
    assert magic == b'Q5SCALE1' and wa == (na - 1).bit_length() and wb == (nb - 1).bit_length()
    da = np.frombuffer(image, '<u2', count=na, offset=32)
    db = np.frombuffer(image, '<u2', count=nb, offset=32 + 2 * na)
    stream = image[32 + 2 * (na + nb):]
    assert len(stream) == size == (count * (wa + wb) + 7) // 8
    bits = np.unpackbits(np.frombuffer(stream, 'u1'), count=count * (wa + wb))
    idx = bits.reshape(count, wa + wb).astype('u4') @ (1 << np.arange(wa + wb - 1, -1, -1, dtype='u4'))
    a, b = idx >> wb, idx & ((1 << wb) - 1)
    assert np.all(a < na) and np.all(b < nb)
    return np.column_stack((da[a], db[b]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=MODEL)
    parser.add_argument('--inventory', type=Path, default=INVENTORY)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--first', type=int, default=0)
    parser.add_argument('--count', type=int, default=37)
    parser.add_argument('--summarize', action='store_true')
    args = parser.parse_args()
    inv = json.loads(args.inventory.read_text())
    assert inv['metadata']['general.architecture'] == 'qwen35moe'
    with args.model.open('rb') as f:
        assert digest(f.read(inv['header_bytes'])) == inv['header_sha256']
    tensors = [t for t in inv['tensors'] if 'ffn_down_exps.weight' in t['name'] and t['type'] == 'Q5_K']
    assert len(tensors) == 37 and all(t['shape'] == [512, 2048, 256] for t in tensors)
    args.output.mkdir(parents=True, exist_ok=True)
    source_hash = digest(Path(__file__).read_bytes())
    inventory_hash = digest(args.inventory.read_bytes())
    if args.summarize:
        records = [json.loads((args.output / f'{i:02d}.json').read_text()) for i in range(len(tensors))]
        assert all(r['tensor'] == t['name'] and r['source_sha256'] == source_hash
                   and r['inventory_sha256'] == inventory_hash for r, t in zip(records, tensors))
        totals = {key: sum(r[key] for r in records) for key in ('blocks', 'original_scale_bytes', 'paid_scale_bytes', 'full_bank_bytes', 'saved_bytes')}
        totals['selected_scale_bytes_per_token'] = totals['original_scale_bytes'] * inv['experts_per_token'] // inv['expert_count']
        totals['selected_saved_bytes_per_token'] = totals['saved_bytes'] * inv['experts_per_token'] // inv['expert_count']
        totals['fraction_complete_one_read'] = totals['selected_saved_bytes_per_token'] / inv['one_token_weight_stream_bytes']
        receipt = dict(summary=totals, source_sha256=source_hash, inventory_sha256=inventory_hash,
                       header_sha256=inv['header_sha256'], model_sha256=json.loads((args.model.parent / 'acquisition.json').read_text()).get('sha256'),
                       per_tensor=[dict(tensor=r['tensor'], payload_sha256=r['payload_sha256'],
                                        scale_image_sha256=r['scale_image_sha256'], receipt_sha256=digest((args.output / f'{i:02d}.json').read_bytes()))
                                   for i, r in enumerate(records)])
        (args.output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
        print(json.dumps(totals, indent=2))
        return
    align = inv['metadata'].get('general.alignment', 32)
    base = (inv['header_bytes'] + align - 1) // align * align
    model = np.memmap(args.model, dtype='u1', mode='r')
    for i in range(args.first, min(args.first + args.count, len(tensors))):
        t = tensors[i]
        dest = args.output / f'{i:02d}.json'
        sidecar = args.output / f'{i:02d}.scales'
        if dest.exists():
            prior = json.loads(dest.read_text())
            if (prior['source_sha256'] == source_hash and prior['inventory_sha256'] == inventory_hash
                    and prior['scale_image_sha256'] == digest(sidecar.read_bytes())):
                continue
        payload = model[base + t['offset']:base + t['offset'] + t['bytes']]
        assert len(payload) == 256 * 2048 * 2 * 176
        scales = payload.reshape(-1, 176)[:, :4].copy().view('<u2').reshape(-1, 2)
        image, widths = encode(scales)
        assert np.array_equal(decode(image), scales), t['name']
        sidecar.write_bytes(image)
        record = dict(tensor=t['name'], layer=int(t['name'].split('.')[1]), blocks=len(scales),
                      original_scale_bytes=scales.nbytes, paid_scale_bytes=len(image),
                      full_bank_bytes=t['bytes'], saved_bytes=scales.nbytes - len(image),
                      widths=widths, source_sha256=source_hash, inventory_sha256=inventory_hash,
                      payload_sha256=digest(payload), scale_image_sha256=digest(image),
                      all_scale_words_exact=True)
        dest.write_text(json.dumps(record, indent=2) + '\n')
        print(i, t['name'], widths, record['saved_bytes'], flush=True)


if __name__ == '__main__':
    main()
