#!/usr/bin/env python3
"""Certify the ideal-real post-MoE RMSNorm observation from the pinned GGUF."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path('../../data/qwen-moe')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, default=ROOT / 'Qwen3.6-35B-A3B-UD-Q4_K_M.gguf')
    p.add_argument('--inventory', type=Path, default=ROOT / 'traffic.json')
    p.add_argument('--out', type=Path, default=ROOT / 'rms-observation/receipt.json')
    a = p.parse_args()
    inventory = json.loads(a.inventory.read_text())
    assert inventory['metadata']['general.architecture'] == 'qwen35moe'
    assert inventory['header_sha256'] == hashlib.sha256(a.model.open('rb').read(inventory['header_bytes'])).hexdigest()
    acquisition_file = a.model.parent / 'acquisition.json'
    acquisition = json.loads(acquisition_file.read_text())
    assert acquisition['verified'] and acquisition['size'] == a.model.stat().st_size == 22_134_528_992
    assert acquisition['filename'] == a.model.name
    epsilon = inventory['metadata']['qwen35moe.attention.layer_norm_rms_epsilon']
    assert np.isfinite(epsilon) and epsilon > 0
    align = inventory['metadata'].get('general.alignment', 32)
    base = (inventory['header_bytes'] + align - 1) // align * align
    names = [f'blk.{i}.attn_norm.weight' for i in range(1, 40)] + ['output_norm.weight']
    tensors = {t['name']: t for t in inventory['tensors']}
    records = []
    for name in names:
        t = tensors[name]
        assert t['shape'] == [2048] and t['type'] == 'F32' and t['bytes'] == 8192
        with a.model.open('rb') as f:
            f.seek(base + t['offset'])
            payload = f.read(t['bytes'])
        assert len(payload) == t['bytes']
        gamma = np.frombuffer(payload, dtype='<f4')
        assert np.isfinite(gamma).all()
        nonzero = int(np.count_nonzero(gamma))
        records.append({'name': name, 'payload_sha256': hashlib.sha256(payload).hexdigest(),
                        'nonzero': nonzero, 'min_abs': float(np.min(np.abs(gamma))),
                        'max_abs': float(np.max(np.abs(gamma)))})
    result = {'source_sha256': digest(Path(__file__)), 'inventory_sha256': digest(a.inventory),
              'acquisition_sha256': digest(acquisition_file),
              'model_header_sha256': inventory['header_sha256'],
              'pinned_model_sha256': acquisition['sha256'],
              'model_bytes': a.model.stat().st_size, 'epsilon': epsilon,
              'observation': 'Ideal-real RMSNorm on the post-MoE residual entering the next layer (or final output). No model evaluation, native FP32 identity, or performance claim.',
              'norms': records, 'all_nonzero': all(r['nonzero'] == 2048 for r in records),
              'minimum_abs_gamma': min(r['min_abs'] for r in records)}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'epsilon': epsilon, 'norms': len(records),
                      'all_nonzero': result['all_nonzero'],
                      'minimum_abs_gamma': result['minimum_abs_gamma']}))


if __name__ == '__main__':
    main()
