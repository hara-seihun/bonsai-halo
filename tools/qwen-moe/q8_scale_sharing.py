#!/usr/bin/env python3
"""Installed Q8_0 scale-word reuse at a common input block, CPU only."""
import argparse
import hashlib
import json
import math
import struct
from pathlib import Path

import numpy as np

MODEL = Path('../../data/qwen-moe/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf')
INVENTORY = Path('../../data/qwen-moe/traffic.json')
OUTPUT = Path('../../data/qwen-moe/q8-scale-sharing')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def census(scales):
    rows, kb = scales.shape
    duplicate = 0
    index_bytes = 0
    entropy_bits = 0.0
    repeated_positions = 0
    singleton_positions = 0
    maximum_group = 0
    for k in range(kb):
        _, counts = np.unique(scales[:, k], return_counts=True)
        distinct = len(counts)
        duplicate += rows - distinct
        repeated_positions += int((counts > 1).sum())
        singleton_positions += int((counts == 1).sum())
        maximum_group = max(maximum_group, int(counts.max()))
        # Each K-specific dictionary stores original 16-bit words. Each row has a
        # fixed-width index. A single-value column needs zero index bits.
        index_bytes += 2 * distinct + (rows * (distinct - 1).bit_length() + 7) // 8
        entropy_bits += math.log2(rows) * rows - float(np.dot(counts, np.log2(counts)))
    return dict(rows=rows, input_blocks=kb, blocks=rows*kb,
                unique_scale_words=rows*kb-duplicate, duplicate_scale_words=duplicate,
                repeated_groups=repeated_positions, singleton_groups=singleton_positions,
                maximum_group=maximum_group, fixed_index_bytes=index_bytes,
                entropy_bits=entropy_bits)


def pack_and_check(scales, path):
    rows, kb = scales.shape
    records = []
    directory = bytearray()
    offset = 8 * kb
    for k in range(kb):
        dictionary, indices = np.unique(scales[:, k], return_inverse=True)
        bits = (len(dictionary) - 1).bit_length()
        if bits:
            shifts = np.arange(bits - 1, -1, -1, dtype=np.uint64)
            packed = np.packbits(((indices.astype(np.uint64)[:, None] >> shifts) & 1).astype('u1').reshape(-1)).tobytes()
        else:
            packed = b''
        record = dictionary.astype('<u2').tobytes() + packed
        directory.extend(struct.pack('<II', len(dictionary), offset))
        records.append(record)
        offset += len(record)
    path.write_bytes(bytes(directory) + b''.join(records))
    image = path.read_bytes()
    for k in range(kb):
        length, start = struct.unpack_from('<II', image, 8*k)
        dictionary = np.frombuffer(image, '<u2', count=length, offset=start)
        bits = (length - 1).bit_length()
        if bits:
            begin = start + length*2
            nbytes = (rows*bits + 7)//8
            digits = np.unpackbits(np.frombuffer(image, 'u1', count=nbytes, offset=begin), count=rows*bits).reshape(rows, bits)
            indices = digits.astype(np.uint64) @ (1 << np.arange(bits-1, -1, -1, dtype=np.uint64))
        else:
            indices = np.zeros(rows, dtype=np.int64)
        assert np.array_equal(dictionary[indices], scales[:, k]), (path, k)
    return len(image)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, default=MODEL)
    p.add_argument('--inventory', type=Path, default=INVENTORY)
    p.add_argument('--output', type=Path, default=OUTPUT)
    p.add_argument('--first', type=int, default=0)
    p.add_argument('--count', type=int, default=250)
    p.add_argument('--summarize', action='store_true')
    p.add_argument('--pack', action='store_true', help='write and fully decode-check a paid scale coordinate per tensor')
    a = p.parse_args()
    inv = json.loads(a.inventory.read_text())
    assert inv['metadata']['general.architecture'] == 'qwen35moe'
    assert inv['header_sha256'] == hashlib.sha256(a.model.open('rb').read(inv['header_bytes'])).hexdigest()
    align = inv['metadata'].get('general.alignment', 32)
    base = (inv['header_bytes'] + align - 1) // align * align
    tensors = [t for t in inv['tensors'] if t['type'] == 'Q8_0' and t['name'] != 'token_embd.weight' and '_exps.' not in t['name']]
    a.output.mkdir(parents=True, exist_ok=True)
    source_hash, inventory_hash = sha(Path(__file__)), sha(a.inventory)
    if a.summarize:
        results = [json.loads((a.output / f'{i:03d}.json').read_text()) for i in range(len(tensors))]
        assert all(r['name'] == t['name'] and r['source_sha256'] == source_hash and r['inventory_sha256'] == inventory_hash for r, t in zip(results, tensors))
        fields = ('blocks', 'unique_scale_words', 'duplicate_scale_words', 'fixed_index_bytes', 'entropy_bits')
        totals = {k: sum(r[k] for r in results) for k in fields}
        totals.update(tensors=len(results), scale_bytes=2*totals['blocks'],
                      q8_bytes=sum(t['bytes'] for t in tensors),
                      complete_one_read_bytes=inv['one_token_weight_stream_bytes'])
        totals['free_scale_reuse_saved_bytes'] = 2*totals['duplicate_scale_words']
        totals['fixed_index_saved_bytes'] = totals['scale_bytes'] - totals['fixed_index_bytes']
        totals['ideal_entropy_scale_saved_bytes'] = totals['scale_bytes'] - totals['entropy_bits']/8
        if all('packed_bytes' in r for r in results):
            totals['packed_scale_bytes'] = sum(r['packed_bytes'] for r in results)
            totals['packed_saved_bytes'] = totals['scale_bytes'] - totals['packed_scale_bytes']
            totals['packed_fraction_complete'] = totals['packed_saved_bytes']/totals['complete_one_read_bytes']
        receipt = dict(summary=totals, source_sha256=source_hash,
                       inventory_sha256=inventory_hash, header_sha256=inv['header_sha256'],
                       model_sha256=inv.get('model_sha256'),
                       per_tensor=[dict(name=r['name'], receipt_sha256=sha(a.output / f'{i:03d}.json'),
                                        payload_sha256=r['payload_sha256'],
                                        packed_sha256=r.get('packed_sha256')) for i, r in enumerate(results)])
        (a.output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
        print(json.dumps(totals, indent=2))
        return
    image = np.memmap(a.model, dtype='u1', mode='r')
    for i in range(a.first, min(a.first + a.count, len(tensors))):
        t = tensors[i]
        destination = a.output / f'{i:03d}.json'
        packed_path = a.output / f'{i:03d}.scales'
        if destination.exists():
            old = json.loads(destination.read_text())
            if old['name'] == t['name'] and old['source_sha256'] == source_hash and old['inventory_sha256'] == inventory_hash and (not a.pack or packed_path.exists() and old.get('packed_sha256') == sha(packed_path)):
                continue
        kb = t['shape'][0] // 32
        assert t['shape'][0] % 32 == 0 and t['bytes'] % (34 * kb) == 0
        rows = t['bytes'] // (34 * kb)
        payload = image[base + t['offset']:base + t['offset'] + t['bytes']]
        scales = payload.reshape(-1, 34)[:, :2].copy().view('<u2').reshape(rows, kb)
        result = census(scales)
        if a.pack:
            result['packed_bytes'] = pack_and_check(scales, packed_path)
            result['packed_sha256'] = sha(packed_path)
            assert result['packed_bytes'] == result['fixed_index_bytes'] + 8*kb
        result.update(name=t['name'], shape=t['shape'], bytes=t['bytes'],
                      payload_sha256=hashlib.sha256(payload).hexdigest(),
                      source_sha256=source_hash, inventory_sha256=inventory_hash,
                      header_sha256=inv['header_sha256'])
        destination.write_text(json.dumps(result, indent=2) + '\n')
        print(i, t['name'], result['duplicate_scale_words'], flush=True)


if __name__ == '__main__':
    main()
