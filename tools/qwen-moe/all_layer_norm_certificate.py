#!/usr/bin/env python3
"""Cost the free-exact-norm triangle stopping certificate on actual Qwen routes."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

TOLERANCES = (0.01, 0.05, 0.10)
FULL_BYTES = 2_626_187_904


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def evaluate(scores, down, image_bytes):
    v = down.astype(np.float64) * scores.astype(np.float64)[:, :, None]
    norms = np.linalg.norm(v, axis=2)
    target = v.sum(axis=1)
    target_norm = np.linalg.norm(target, axis=1)
    assert np.all(target_norm > 0)
    result = {}
    for name, order in (
        ('score', np.argsort(-scores, axis=1, kind='stable')),
        ('free_norm_order', np.argsort(-norms, axis=1, kind='stable')),
    ):
        ordered = np.take_along_axis(v, order[:, :, None], axis=1)
        ordered_norm = np.take_along_axis(norms, order, axis=1)
        prefixes = np.cumsum(ordered, axis=1)
        bounds = np.cumsum(ordered_norm[:, ::-1], axis=1)[:, ::-1]
        first = {str(t): np.full(len(scores), 8, dtype=np.int32) for t in TOLERANCES}
        for k in range(1, 8):
            p = np.linalg.norm(prefixes[:, k-1], axis=1)
            b = bounds[:, k]
            # b/(p-b) <= tolerance, assuming p > b, equivalently
            # b <= tolerance*p/(1+tolerance). No omitted direction is inspected.
            for tolerance in TOLERANCES:
                stop = first[str(tolerance)]
                stop[(stop == 8) & (b <= tolerance*p/(1+tolerance))] = k
        result[name] = {}
        for tolerance in TOLERANCES:
            k = first[str(tolerance)]
            omit = 8-k
            # FP64 triangle certificate must not endorse a violating output.
            residual = target - prefixes[np.arange(len(scores)), k-1]
            relative = np.linalg.norm(residual, axis=1)/target_norm
            assert np.all(relative <= tolerance + 1e-12)
            result[name][str(tolerance)] = {
                'skipped': int(omit.sum()),
                'stopped': int(np.count_nonzero(omit)),
                'histogram_skipped_0_to_7': np.bincount(omit, minlength=8).tolist(),
                'image_bytes_skipped': int(omit.sum()) * image_bytes,
                'max_certified_local_error': float(relative.max()),
            }
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('capture', type=Path)
    ap.add_argument('--out', required=True, type=Path)
    args = ap.parse_args()
    provenance_file = args.capture/'receipt.json'
    traffic_file = args.capture.parent/'traffic.json'
    provenance = json.loads(provenance_file.read_text())
    traffic = json.loads(traffic_file.read_text())
    image_bytes = [traffic['layers'][str(i)]['bank_bytes']//256 for i in range(40)]
    assert 8*sum(image_bytes) == traffic['active_routed_bytes_per_token']
    panels = {}
    for split in ('train', 'held'):
        n = provenance['splits'][split]['tokens']
        assert n == 64
        layers = []
        for i in range(40):
            arrays = []
            for field, shape in (('ffn_moe_weights_norm', (n, 8)), ('ffn_moe_down', (n, 8, 2048))):
                file = args.capture/f'{split}.layer-{i}.{field}.f32'
                assert file.stat().st_size == np.prod(shape)*4
                assert sha(file) == provenance['splits'][split]['files_sha256'][file.name]
                arrays.append(np.memmap(file, mode='r', dtype='<f4', shape=shape))
            scores, down = arrays
            assert np.isfinite(scores).all() and np.isfinite(down).all()
            assert np.all(scores > 0) and np.all(scores[:, :-1] >= scores[:, 1:])
            assert np.allclose(scores.sum(axis=1), 1, atol=1e-5)
            layers.append({'layer': i, 'expert_image_bytes': image_bytes[i],
                           'certificate': evaluate(scores, down, image_bytes[i])})
        aggregate = {}
        for order in ('score', 'free_norm_order'):
            aggregate[order] = {}
            for tolerance in TOLERANCES:
                key = str(tolerance)
                rows = [layer['certificate'][order][key] for layer in layers]
                total_skipped = sum(row['skipped'] for row in rows)
                skipped_bytes = sum(row['image_bytes_skipped'] for row in rows)
                aggregate[order][key] = {
                    'skipped_assignments': total_skipped,
                    'stopped_layer_tokens': sum(row['stopped'] for row in rows),
                    'skipped_bytes_per_token': skipped_bytes/n,
                    'conditional_complete_one_read_bytes_percent':
                        100*skipped_bytes/(n*FULL_BYTES),
                    'histogram_skipped_0_to_7': np.sum([row['histogram_skipped_0_to_7'] for row in rows], axis=0).tolist(),
                }
        panels[split] = {'tokens': n, 'layers': layers, 'aggregate': aggregate}
        print(split, json.dumps(aggregate, indent=2))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        'contract': 'FP64 triangle norm-only certificate of local Euclidean routed-sum error on callback-captured FP32 down vectors and normalized scores. Exact omitted norms and even norm ordering are granted free; not native FP32 identity, measured memory traffic, complete language quality, or executable speed.',
        'source_sha256': sha(Path(__file__)),
        'capture_receipt_sha256': sha(provenance_file),
        'traffic_sha256': sha(traffic_file),
        'complete_one_read_bytes_per_token': FULL_BYTES,
        'panels': panels,
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
