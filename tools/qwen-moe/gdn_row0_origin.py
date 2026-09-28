#!/usr/bin/env python3
"""Locate the first saved R0 change in matched gathered 31+1 Qwen row orders.

Compare each time coordinate against all 32 control rows as FP32 words. This
reads saved post-transition state only; it cannot identify an unsaved producer.
"""
import argparse
import hashlib
import json
import mmap
from pathlib import Path

import numpy as np

from gdn_state_diff import regions


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def report(normal, swapped):
    a = {name: info for name, _, _, info in regions(normal) if name.startswith('recurrent/r/')}
    b = {name: info for name, _, _, info in regions(swapped) if name.startswith('recurrent/r/')}
    if a != b or len(a) != 30:
        raise ValueError('recurrent R layout differs')
    result = {}
    for name in list(a)[:3]:
        info = a[name]
        if info['count'] != 32 or info['row_bytes'] != 3 * 8192 * 4:
            raise ValueError(f'unexpected R row layout {info}')
        off = info['payload']
        x = np.frombuffer(normal, '<u4', count=32 * 8192 * 3, offset=off).reshape(32,8192,3)
        y = np.frombuffer(swapped, '<u4', count=32 * 8192 * 3, offset=off).reshape(32,8192,3)
        same_rows = [int(row) for row in range(32) if np.array_equal(x[row], y[row])]
        rows = {}
        for row in (0,1,4):
            planes = []
            for plane in range(3):
                equal = np.count_nonzero(x[row,:,plane] == y[row,:,plane])
                match = np.count_nonzero(x[:,:,plane] == y[row,:,plane][None,:], axis=1)
                planes.append({'same_row_equal_words': int(equal),
                               'normal_row_matches': [int(i) for i in np.flatnonzero(match == 8192)],
                               'best_normal_row': int(np.argmax(match)),
                               'best_normal_equal_words': int(np.max(match))})
            rows[str(row)] = planes
        result[name] = {'identical_physical_rows': same_rows, 'inspected_rows': rows}
        if name == 'recurrent/r/0':
            u = x[0,:,2].view('<f4').astype('<f8')
            v = y[0,:,2].view('<f4').astype('<f8')
            if not np.isfinite(u).all() or not np.isfinite(v).all():
                raise ValueError('nonfinite R0 current plane')
            delta = np.abs(u-v)
            result[name]['row0_current_difference'] = {
                'changed_words': int(np.count_nonzero(x[0,:,2] != y[0,:,2])),
                'max_absolute': float(delta.max()),
                'rms': float(np.sqrt(np.mean(delta**2))),
                'normal_rms': float(np.sqrt(np.mean(u**2))),
                'zero_mask_identical': bool(np.array_equal(u == 0, v == 0)),
            }
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('normal', type=Path)
    p.add_argument('swapped', type=Path)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    with args.normal.open('rb') as fa, args.swapped.open('rb') as fb:
        with mmap.mmap(fa.fileno(), 0, access=mmap.ACCESS_READ) as a, mmap.mmap(fb.fileno(), 0, access=mmap.ACCESS_READ) as b:
            result = report(a,b)
    receipt = {'normal': str(args.normal), 'normal_sha256': sha(args.normal),
               'swapped': str(args.swapped), 'swapped_sha256': sha(args.swapped),
               'script_sha256': sha(Path(__file__)), 'comparison': result}
    text = json.dumps(receipt, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    else:
        print(text, end='')


if __name__ == '__main__':
    main()
