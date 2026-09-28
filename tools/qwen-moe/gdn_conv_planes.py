#!/usr/bin/env python3
"""Compare the three convolution-history coordinates of a saved swapped Qwen state.

For the pinned model, build_conv_state reshapes the old 3x8192 row, concatenates
one new projected input on axis 0, and stores the last three coordinates.
This observes saved *post-step* states, not graph intermediates or timing.
"""
import argparse
import json
import mmap
from pathlib import Path

import numpy as np

from gdn_state_diff import regions


def inspect(control, candidate):
    if len(control) != len(candidate):
        raise ValueError('different serialized state lengths')
    a = {name: info for name, _, _, info in regions(control) if name.startswith('recurrent/r/')}
    b = {name: info for name, _, _, info in regions(candidate) if name.startswith('recurrent/r/')}
    if a != b or len(a) != 30:
        raise ValueError('recurrent R layout differs or incomplete')
    result = {}
    for name, info in a.items():
        count = info['count']
        size = info['row_bytes']
        if count != 32 or size != 3 * 8192 * 4:
            raise ValueError(f'unexpected convolution layout: {name}: {info}')
        offset = info['payload']
        # uint32 equality tests exact FP32 bits, including signed zero and NaNs.
        x = np.frombuffer(control, '<u4', count=count * size // 4, offset=offset).reshape(count, 8192, 3)
        y = np.frombuffer(candidate, '<u4', count=count * size // 4, offset=offset).reshape(count, 8192, 3)
        eq = x == y
        plane_equal = eq.sum(axis=1).astype(int)
        if np.any(plane_equal[np.arange(count) != 1] != 8192):
            raise ValueError(f'other serialized rows changed in {name}')
        result[name] = {
            'row_1_equal_words_by_history_coordinate': plane_equal[1].tolist(),
            'other_31_rows_bit_identical': True,
        }
    return {'domain': 'selected Qwen3.6-35B-A3B, rejected broad 31+1 writer versus gathered, saved after swap',
            'coordinate': 'R row is [8192 channels, 3 time coordinates]; last = current qkv_mixed, first two = old history positions 1 and 2',
            'words_per_coordinate': 8192, 'recurrent_layers': result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('control', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    with args.control.open('rb') as ca, args.candidate.open('rb') as cb:
        with mmap.mmap(ca.fileno(), 0, access=mmap.ACCESS_READ) as a, mmap.mmap(cb.fileno(), 0, access=mmap.ACCESS_READ) as b:
            receipt = inspect(a, b)
    receipt['control_path'] = str(args.control)
    receipt['candidate_path'] = str(args.candidate)
    text = json.dumps(receipt, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    else:
        print(text, end='')


if __name__ == '__main__':
    main()
