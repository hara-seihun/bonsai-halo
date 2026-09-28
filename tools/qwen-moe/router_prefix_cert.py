#!/usr/bin/env python3
"""Certify real-arithmetic Qwen top-eight using suffix norm intervals on actual producers."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

MODEL = Path('../../data/qwen-moe/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf')
DATA = Path('../../data/qwen-moe/all-producers')
INVENTORY = Path('../../data/qwen-moe/traffic.json')
OUTPUT = Path('../../data/qwen-moe/router-prefix-cert')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def analyze(weights, x, ids):
    # Prefix dots are recomputed independently at each checkpoint. Intervals are
    # statements about real sums of the stored F32 weights and captured F32 inputs.
    n = x.shape[0]
    points = list(range(128, 2049, 128))
    results = []
    w64 = np.asarray(weights, dtype=np.float64)
    suffix_gram = np.zeros((256, 256), dtype=np.float64)
    distances = {}
    for k in reversed(points[:-1]):
        block = w64[:, k:k + 128]
        suffix_gram += block @ block.T
        diag = np.diag(suffix_gram)
        distances[k] = np.sqrt(np.maximum(0, diag[:, None] + diag[None, :] - 2 * suffix_gram))
    for k in points:
        partial = np.asarray(x[:, :k], dtype=np.float64) @ np.asarray(weights[:, :k], dtype=np.float64).T
        suffix_w = np.linalg.norm(np.asarray(weights[:, k:], dtype=np.float64), axis=1)
        suffix_x = np.linalg.norm(np.asarray(x[:, k:], dtype=np.float64), axis=1)
        bound = suffix_x[:, None] * suffix_w[None, :]
        lower, upper = partial - bound, partial + bound
        # A candidate is certified iff its lower bound exceeds the upper bound
        # of every other row. Strict comparison handles all tie conventions.
        order = np.argsort(-lower, axis=1)[:, :8]
        l8 = np.take_along_axis(lower, order, axis=1).min(axis=1)
        mask = np.ones((n, 256), dtype=bool)
        np.put_along_axis(mask, order, False, axis=1)
        outsiders = np.max(np.where(mask, upper, -np.inf), axis=1)
        certified = l8 > outsiders
        observed = np.array([set(order[i]) == set(ids[i]) for i in range(n)])
        if np.any(certified & ~observed):
            raise AssertionError('certificate disagrees with native top-eight')
        # An optimistic single-rival oracle knows the actual route and can pay
        # none of the search cost. It still needs this suffix enclosure.
        true_lower = np.take_along_axis(lower, ids, axis=1).min(axis=1)
        true_mask = np.ones((n, 256), dtype=bool)
        np.put_along_axis(true_mask, ids, False, axis=1)
        hindsight = true_lower > np.max(np.where(true_mask, upper, -np.inf), axis=1)
        if k == 2048:
            pairwise = hindsight_pairwise = np.ones(n, dtype=bool)
        else:
            distance = distances[k]
            # Pairwise Cauchy bound encloses (w_i-w_j)·x_suffix; shared
            # suffix terms cancel. No individual-row enclosure is required.
            def pair_check(candidates):
                delta = (np.take_along_axis(partial, candidates, axis=1)[:, :, None]
                         - partial[:, None, :])
                rhs = suffix_x[:, None, None] * distance[candidates]
                comparisons = delta > rhs
                np.put_along_axis(comparisons, candidates[:, :, None], True, axis=2)
                return comparisons.all(axis=(1, 2))
            pair_candidates = np.argsort(-partial, axis=1)[:, :8]
            pairwise = pair_check(pair_candidates)
            hindsight_pairwise = pair_check(ids)
            if np.any(pairwise & np.array([set(pair_candidates[i]) != set(ids[i]) for i in range(n)])):
                raise AssertionError('pairwise certificate disagrees with native top-eight')
        results.append(dict(prefix=k, certified=int(certified.sum()), oracle=int(hindsight.sum()),
                            pairwise=int(pairwise.sum()), pairwise_oracle=int(hindsight_pairwise.sum()),
                            candidate_matches=int(observed.sum()),
                            min_gap=float(np.min(l8 - outsiders)),
                            median_gap=float(np.median(l8 - outsiders))))
    full = np.asarray(x, dtype=np.float64) @ np.asarray(weights, dtype=np.float64).T
    order = np.argsort(-full, axis=1)[:, :8]
    if any(set(order[i]) != set(ids[i]) for i in range(n)):
        raise AssertionError('full decoded router top-eight does not match captured route')
    gap = np.sort(full, axis=1)[:, -8] - np.sort(full, axis=1)[:, -9]
    return results, dict(min_margin=float(gap.min()), median_margin=float(np.median(gap)),
                         min_abs_input=float(np.min(np.abs(x))), max_abs_input=float(np.max(np.abs(x))))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--first', type=int, default=0)
    p.add_argument('--count', type=int, default=40)
    p.add_argument('--summarize', action='store_true')
    p.add_argument('--output', type=Path, default=OUTPUT)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    if a.summarize:
        layers = [a.output / f'layer-{i:02d}.json' for i in range(40)]
        rows = [json.loads(path.read_text()) for path in layers]
        assert all(row['layer'] == i and row['source_sha256'] == digest(Path(__file__))
                   and row['inventory_sha256'] == digest(INVENTORY) for i, row in enumerate(rows))
        summary = dict(source_sha256=digest(Path(__file__)), inventory_sha256=digest(INVENTORY),
                       acquisition_sha256=digest(Path('../../data/qwen-moe/acquisition.json')),
                       layer_receipts=[dict(path=path.name, sha256=digest(path),
                                            weight_sha256=row['weight_sha256'])
                                       for path, row in zip(layers, rows)],
                       splits={})
        for split in ('train', 'held'):
            summary['splits'][split] = [dict(prefix=k, **{
                key: sum(next(p for p in r['splits'][split]['points'] if p['prefix'] == k)[key] for r in rows)
                for key in ('certified', 'oracle', 'pairwise', 'pairwise_oracle', 'candidate_matches')})
                for k in range(128, 2049, 128)]
        (a.output / 'receipt.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(json.dumps(summary['splits'], indent=2))
        return
    inv = json.loads(INVENTORY.read_text())
    acquisition = json.loads(Path('../../data/qwen-moe/acquisition.json').read_text())
    assert acquisition['verified'] and acquisition['size'] == MODEL.stat().st_size
    assert inv['metadata']['general.architecture'] == 'qwen35moe'
    with MODEL.open('rb') as f:
        assert hashlib.sha256(f.read(inv['header_bytes'])).hexdigest() == inv['header_sha256']
    align = int(inv['metadata'].get('general.alignment', 32))
    base = (inv['header_bytes'] + align - 1) // align * align
    indexed = {t['name']: t for t in inv['tensors']}
    for layer in range(a.first, min(40, a.first + a.count)):
        t = indexed[f'blk.{layer}.ffn_gate_inp.weight']
        assert t['type'] == 'F32' and t['shape'] == [2048, 256] and t['bytes'] == 2048 * 256 * 4
        w = np.memmap(MODEL, dtype='<f4', offset=base + t['offset'], shape=(256, 2048), mode='r')
        assert np.isfinite(w).all()
        record = dict(layer=layer, weight_sha256=hashlib.sha256(np.asarray(w).tobytes()).hexdigest(), splits={})
        for split in ('train', 'held'):
            pre = DATA / f'{split}.layer-{layer}'
            xp = Path(str(pre) + '.attn_post_norm.f32')
            ip = Path(str(pre) + '.ffn_moe_topk.i32')
            x = np.fromfile(xp, dtype='<f4').reshape(-1, 2048)
            ids = np.fromfile(ip, dtype='<i4').reshape(-1, 8)
            assert x.shape == (64, 2048) and ids.shape == (64, 8)
            assert np.isfinite(x).all() and np.all((0 <= ids) & (ids < 256))
            results, margins = analyze(w, x, ids)
            record['splits'][split] = dict(input_sha256=digest(xp), ids_sha256=digest(ip),
                                          points=results, full=margins)
        record.update(source_sha256=digest(Path(__file__)), inventory_sha256=digest(INVENTORY),
                      model_sha256=acquisition['sha256'], acquisition_sha256=digest(Path('../../data/qwen-moe/acquisition.json')),
                      model_header_sha256=inv['header_sha256'])
        (a.output / f'layer-{layer:02d}.json').write_text(json.dumps(record, indent=2) + '\n')
        print(layer, [(q['prefix'], q['certified']) for q in record['splits']['held']['points']], flush=True)


if __name__ == '__main__':
    main()
