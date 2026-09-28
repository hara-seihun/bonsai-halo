#!/usr/bin/env python3
"""Identify the physical pre-swap source of a saved convolution history plane."""
import argparse
import hashlib
import json
import mmap
from pathlib import Path

import numpy as np

from gdn_state_diff import regions


def r_regions(data):
    return {name: info for name, _, _, info in regions(data) if name.startswith('recurrent/r/')}


def plane(data, info, row, time):
    if info['count'] != 32 or info['row_bytes'] != 3 * 8192 * 4:
        raise ValueError('unexpected Qwen convolution state layout')
    offset = info['payload'] + row * info['row_bytes'] + time * 4
    return np.ndarray((8192,), dtype='<u4', buffer=data, offset=offset, strides=(12,))


def identify(pre, control, direct):
    info = [r_regions(state) for state in (pre, control, direct)]
    if not (info[0].keys() == info[1].keys() == info[2].keys() and len(info[0]) == 30):
        raise ValueError('recurrent layer mismatch')
    result = {}
    for name in info[0]:
        counts = {}
        for t in range(2):
            sources = [plane(pre, info[0][name], row, t + 1) for row in range(32)]
            for arm, state, meta in [('gathered', control, info[1][name]), ('direct', direct, info[2][name])]:
                observed = plane(state, meta, 1, t)
                hits = [int(np.count_nonzero(observed == source)) for source in sources]
                counts[f'{arm}_old_{t}'] = {'matching_words_by_pre_row': hits,
                    'exact_pre_rows': [row for row, count in enumerate(hits) if count == 8192]}
        result[name] = counts
    return result


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pre', type=Path)
    parser.add_argument('gathered', type=Path)
    parser.add_argument('direct', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    files = [args.pre, args.gathered, args.direct]
    handles = [path.open('rb') for path in files]
    maps = [mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) for handle in handles]
    try:
        layers = identify(*maps)
    finally:
        for data in maps:
            data.close()
        for handle in handles:
            handle.close()
    r1 = layers['recurrent/r/1']
    for time in range(2):
        if r1[f'gathered_old_{time}']['exact_pre_rows'] != [0] or r1[f'direct_old_{time}']['exact_pre_rows'] != [1]:
            raise ValueError('layer-1 old-history source mapping did not reproduce')
    receipt = {'domain': 'pinned Qwen3.6, 32 stable generated rows then 31+1 two-ID swap; post-step row 1',
               'comparison': 'FP32 u32 bit equality of each saved R old plane with all 32 pre-swap physical rows',
               'files': {str(path): {'bytes': path.stat().st_size, 'sha256': sha256(path)} for path in files},
               'layers': layers}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + '\n')
    for layer in ['recurrent/r/0', 'recurrent/r/1', 'recurrent/r/2']:
        print(layer, {key: value['exact_pre_rows'] for key, value in layers[layer].items()})
    print(args.output)


if __name__ == '__main__':
    main()
