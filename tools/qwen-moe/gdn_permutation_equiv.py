#!/usr/bin/env python3
"""Replay the logical-ID-aligned permutation panel at full-head and recurrent-state granularity."""
import argparse
import hashlib
import json
import mmap
from pathlib import Path

import numpy as np
from gdn_state_diff import regions

VOCAB = 248320
ROWS = 32
STEPS = 3


def file_digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return {'bytes': path.stat().st_size, 'sha256': h.hexdigest()}


def heads(a, b):
    x = np.memmap(a, dtype='<u4', mode='r', shape=(STEPS, ROWS, VOCAB))
    y = np.memmap(b, dtype='<u4', mode='r', shape=(STEPS, ROWS, VOCAB))
    return [{'step': step, 'logical_sequence': row, 'differing_f32_bits': int(np.count_nonzero(x[step, row] != y[step, row])),
             'top_a': int(np.argmax(x[step, row].view('<f4'))), 'top_b': int(np.argmax(y[step, row].view('<f4')))}
            for step in range(STEPS) for row in range(ROWS) if np.any(x[step, row] != y[step, row])]


def meta(blob, record):
    _, start, _, count = record
    pos = start + 4
    result = []
    for row in range(count):
        p = int(np.frombuffer(blob, '<i4', 1, pos)[0]); pos += 4
        n = int(np.frombuffer(blob, '<u4', 1, pos)[0]); pos += 4
        ids = tuple(int(x) for x in np.frombuffer(blob, '<i4', n, pos)); pos += 4*n
        result.append({'row': row, 'position': p, 'ids': ids})
    return result


def states(a, b):
    with a.open('rb') as af, b.open('rb') as bf, mmap.mmap(af.fileno(), 0, access=mmap.ACCESS_READ) as x, mmap.mmap(bf.fileno(), 0, access=mmap.ACCESS_READ) as y:
        rx, ry = list(regions(x)), list(regions(y))
        if [(name, end-start) for name, start, end, _ in rx] != [(name, end-start) for name, start, end, _ in ry]:
            raise ValueError('incompatible serialized state regions')
        result = {'recurrent_meta_a': meta(x, next(r for r in rx if r[0] == 'recurrent/meta')),
                  'recurrent_meta_b': meta(y, next(r for r in ry if r[0] == 'recurrent/meta')),
                  'different_regions': []}
        for (name, start, end, info), (_, bs, be, bi) in zip(rx, ry):
            if x[start:end] == y[bs:be]:
                continue
            record = {'region': name, 'bytes': end-start}
            if name.startswith('recurrent/') and name != 'recurrent/meta':
                if info['count'] != 32 or info['row_bytes'] != bi['row_bytes']:
                    raise ValueError('incompatible recurrent row layout')
                row_bytes = info['row_bytes']
                differences = []
                for row in range(32):
                    p, q = info['payload']+row*row_bytes, bi['payload']+row*row_bytes
                    u, v = np.frombuffer(x, '<u4', row_bytes//4, p), np.frombuffer(y, '<u4', row_bytes//4, q)
                    n = int(np.count_nonzero(u != v))
                    del u, v
                    if n: differences.append({'physical_row': row, 'different_f32_bits': n})
                record['rows'] = differences
            result['different_regions'].append(record)
        return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('directory', type=Path)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    root = args.directory
    arms = ('normal', 'swap', 'normal-gather', 'swap-gather')
    pairs = (('normal', 'normal-gather'), ('swap', 'swap-gather'), ('normal-gather', 'swap-gather'))
    receipt = {'contract': 'same 32 logical sequences and next tokens, final decode row 0/1 permuted only in swap; FP32 head bits indexed by logical ID',
               'model': file_digest(Path('../../data/qwen-moe/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf')),
               'source': file_digest(Path(__file__).with_suffix('.cpp')),
               'analyzer': file_digest(Path(__file__)),
               'probe': file_digest(root / 'probe'),
               'arms': {arm: {ext: file_digest(root / (arm+ext)) for ext in ('.f32', '.state', '.wrapper.log')} for arm in arms},
               'pairs': {a+'--'+b: {'head_differences': heads(root/(a+'.f32'), root/(b+'.f32')),
                                  'state': states(root/(a+'.state'), root/(b+'.state'))}
                         for a,b in pairs}}
    args.output.write_text(json.dumps(receipt, indent=2) + '\n')
    for key, pair in receipt['pairs'].items():
        print(key, 'heads:', pair['head_differences'], 'regions:', [(r['region'], len(r.get('rows', []))) for r in pair['state']['different_regions']])
    print(args.output)


if __name__ == '__main__':
    main()
