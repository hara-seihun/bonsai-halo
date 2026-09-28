#!/usr/bin/env python3
"""Localize saved Qwen gathered permutation differences by producer layer and token."""
import argparse
import hashlib
import json
import mmap
from pathlib import Path

import numpy as np

from gdn_state_diff import regions


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def compare(root):
    a_path = root / 'normal-gather.state'
    b_path = root / 'swap-gather.state'
    with a_path.open('rb') as af, b_path.open('rb') as bf, \
            mmap.mmap(af.fileno(), 0, access=mmap.ACCESS_READ) as a, \
            mmap.mmap(bf.fileno(), 0, access=mmap.ACCESS_READ) as b:
        ra = {name: (start, end, info) for name, start, end, info in regions(a)}
        rb = {name: (start, end, info) for name, start, end, info in regions(b)}
        if list(ra) != list(rb):
            raise ValueError('different serialization layout')
        result = {'input_states_sha256': {p.name: digest(p) for p in (a_path, b_path)},
                  'sources': {'analyzer_sha256': digest(Path(__file__)),
                              'parser_sha256': digest(Path(__file__).with_name('gdn_state_diff.py')),
                              'four_arm_receipt_sha256': digest(root / 'receipt.json')},
                  'contract': 'normal vs swapped third batch row order, identical logical IDs/tokens; gathered GDN on both. Serialized state is post-third-step, not an intermediate forward trace.',
                  'kv': {}, 'recurrent': {}}
        for row in range(32):
            mkey = f'kv/{row}/meta'
            ma, me, count = ra[mkey]
            mb, be, bcount = rb[mkey]
            if count != bcount or a[ma:me] != b[mb:be]:
                raise ValueError(f'KV metadata differs for physical row {row}')
            krow = {}
            for kind in ('k', 'v'):
                changed = {}
                for layer in range(10):
                    key = f'kv/{row}/{kind}/{layer}'
                    start, end, n = ra[key]
                    bs, be, bn = rb[key]
                    # dtype i32, row bytes u64, then n contiguous token images.
                    if n != bn or end-start != be-bs or (end-start-12) % n:
                        raise ValueError(f'incompatible KV layout {key}')
                    if a[start:start+12] != b[bs:bs+12]:
                        raise ValueError(f'KV tensor header changed {key}')
                    stride = (end-start-12)//n
                    counts = [sum(x != y for x, y in zip(a[start+12+t*stride:start+12+(t+1)*stride],
                                                         b[bs+12+t*stride:bs+12+(t+1)*stride]))
                              for t in range(n)]
                    if any(counts):
                        entry = {'token_byte_differences': counts, 'bytes_per_token': stride}
                        if row == 4 and kind == 'k' and layer == 0:
                            if a[start:start+4] != (1).to_bytes(4, 'little') or stride % 2:
                                raise ValueError('expected FP16 key tensor')
                            x = np.frombuffer(a[start+12+9*stride:start+12+10*stride], dtype='<u2')
                            y = np.frombuffer(b[bs+12+9*stride:bs+12+10*stride], dtype='<u2')
                            entry['changed_half_indices_last_token'] = np.flatnonzero(x != y).tolist()
                            entry['maximum_absolute_half_value_difference'] = float(np.max(np.abs(x.astype('<f2').astype('f4') - y.astype('<f2').astype('f4'))))
                        changed[str(3+4*layer)] = entry
                if changed:
                    krow[kind] = changed
            if krow:
                result['kv'][str(row)] = krow
        for row in range(32):
            changed = {}
            for kind in ('r', 's'):
                layers = {}
                for layer in range(40):
                    key = f'recurrent/{kind}/{layer}'
                    if key not in ra:
                        continue
                    start, end, info = ra[key]
                    bs, be, other = rb[key]
                    if (info['count'] != 32 or info['row_bytes'] != other['row_bytes'] or
                            end-start != be-bs):
                        raise ValueError(f'incompatible recurrent layout {key}')
                    stride = info['row_bytes']
                    x = a[info['payload']+row*stride:info['payload']+(row+1)*stride]
                    y = b[other['payload']+row*stride:other['payload']+(row+1)*stride]
                    if x != y:
                        layers[str(layer)] = {'different_bytes': int(np.count_nonzero(np.frombuffer(x, 'u1') != np.frombuffer(y, 'u1'))),
                                              'row_bytes': stride}
                if layers:
                    changed[kind] = layers
            if changed:
                result['recurrent'][str(row)] = changed
        return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = compare(args.directory)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'output': str(args.output),
                      'first_kv': {row: {kind: next(iter(v)) for kind,v in kinds.items()} for row,kinds in result['kv'].items()},
                      'first_recurrent': {row: {kind: next(iter(v)) for kind,v in kinds.items()} for row,kinds in result['recurrent'].items()}}, indent=2))


if __name__ == '__main__':
    main()
