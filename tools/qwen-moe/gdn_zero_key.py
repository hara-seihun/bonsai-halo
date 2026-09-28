#!/usr/bin/env python3
"""Census complete saved K/V cells of the matched Qwen row-permutation panel."""
import argparse
import hashlib
import json
import mmap
from pathlib import Path

import numpy as np

from gdn_state_diff import regions


def digest(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def inspect(root):
    arms = ('normal', 'swap', 'normal-gather', 'swap-gather')
    paths = {name: root / (name + '.state') for name in arms}
    files = [paths[name].open('rb') for name in arms]
    blobs = [mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) for f in files]
    try:
        tables = [{name: (start, end, info) for name, start, end, info in regions(blob)} for blob in blobs]
        if any(list(t) != list(tables[0]) for t in tables[1:]):
            raise ValueError('state region layouts disagree')
        result = {'contract': 'Saved post-third-step FP16 K/V images; four matched 32-sequence three-step arms, swapped rows 0/1 only at step 2. A zero cell diagnoses the saved destination, not its producer or graph operation.',
                  'state_sha256': {name: digest(path) for name, path in paths.items()},
                  'probe_sha256': digest(Path(__file__).with_name('gdn_permutation_equiv.cpp')),
                  'parser_sha256': digest(Path(__file__).with_name('gdn_state_diff.py')),
                  'panel_sha256': digest(root / 'receipt.json'),
                  'source_sha256': digest(Path(__file__)), 'arms': {}}
        for arm, blob, table in zip(arms, blobs, tables):
            zeros = []
            all_cells = 0
            nonzero_counts = {}
            for row in range(32):
                for kind in ('k', 'v'):
                    for layer in range(10):
                        start, end, count = table[f'kv/{row}/{kind}/{layer}']
                        if count != 10 or blob[start:start+4] != (1).to_bytes(4, 'little'):
                            raise ValueError('unexpected KV dtype or length')
                        stride = (end-start-12)//count
                        if stride != 1024 or end-start-12 != count*stride:
                            raise ValueError('unexpected KV cell stride')
                        for token in range(count):
                            data = blob[start+12+stride*token:start+12+stride*(token+1)]
                            codes = np.frombuffer(data, '<u2')
                            nonzero = int(np.count_nonzero(codes))
                            all_cells += 1
                            if nonzero == 0:
                                zeros.append([row, kind, layer, token])
                            if row in (0, 1, 4) and token == 9:
                                nonzero_counts[f'{row}/{kind}/{layer}'] = nonzero
            result['arms'][arm] = {'cells': all_cells, 'zero_cells': zeros,
                                   'last_cell_nonzero_halfwords': nonzero_counts}
        return result
    finally:
        for blob in blobs:
            blob.close()
        for f in files:
            f.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('../../data/qwen-moe/gdn-permutation-equiv'))
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    report = inspect(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({name: {'total_cells': arm['cells'], 'zero_cells': len(arm['zero_cells']),
                             'zero_rows': sorted(set(z[0] for z in arm['zero_cells']))}
                      for name, arm in report['arms'].items()}, indent=2))


if __name__ == '__main__':
    main()
