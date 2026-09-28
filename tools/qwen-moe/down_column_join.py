#!/usr/bin/env python3
"""Join exact down-column classes to complete paired gate/up dormant masks."""
import hashlib
import json
from pathlib import Path

import numpy as np

BASE = Path('../../data/qwen-moe')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mask(text):
    return np.unpackbits(np.frombuffer(bytes.fromhex(text), dtype='u1'), bitorder='little').astype(bool)


def main():
    down = BASE / 'down-column-identity'
    dormant = down / 'dormant-current'
    predecessor = BASE / 'all-layer-dormancy'
    result = {'contract': 'bitwise decoded down repetition mask subset of paired decoded gate/up all-zero rows; real finite-input SwiGLU contract',
              'source_sha256': sha(Path(__file__)),
              'down_receipt_sha256': sha(down / 'receipt.json'),
              'dormancy_aggregate_sha256': sha(dormant / 'aggregate.json'),
              'predecessor_dormancy_aggregate_sha256': sha(predecessor / 'aggregate.json'),
              'layers': [], 'repeated_coordinates': 0, 'paired_dormant_repeats': 0,
              'active_repeated_coordinates': {split: 0 for split in ('train', 'held')}}
    for layer in range(40):
        a_path = down / f'layer-{layer}.json'
        d_path = dormant / f'layer-{layer}.json'
        a, d = json.loads(a_path.read_text()), json.loads(d_path.read_text())
        old_path = predecessor / f'layer-{layer}.json'
        old = json.loads(old_path.read_text())
        assert old['masks_hex'] == d['masks_hex']
        assert a['layer'] == d['layer'] == layer and a['model_sha256'] == d['model_sha256']
        assert a['library_sha256'] == d['library_sha256']
        assert len(a['repeated_masks_hex']) == len(d['masks_hex']) == 256
        repeats = 0
        for expert in range(256):
            r, z = mask(a['repeated_masks_hex'][expert]), mask(d['masks_hex'][expert])
            assert len(r) == len(z) == 512
            assert not (r & ~z).any(), (layer, expert, np.flatnonzero(r & ~z).tolist())
            repeats += int(r.sum())
        result['repeated_coordinates'] += repeats
        result['paired_dormant_repeats'] += repeats
        result['layers'].append({'layer': layer, 'repeated_coordinates': repeats,
                                 'down_sha256': sha(a_path), 'dormancy_sha256': sha(d_path),
                                 'predecessor_dormancy_sha256': sha(old_path)})
        for split in ('train', 'held'):
            result['active_repeated_coordinates'][split] += a['splits'][split]['repeated_active_coordinates']
    assert result['active_repeated_coordinates'] == {'train': 0, 'held': 0}
    (down / 'join-receipt.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('repeated_coordinates', 'paired_dormant_repeats', 'active_repeated_coordinates')}, indent=2))


if __name__ == '__main__':
    main()
