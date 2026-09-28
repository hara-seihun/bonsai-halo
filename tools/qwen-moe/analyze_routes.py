#!/usr/bin/env python3
"""Measure real-route score concentration and omitted expert output on captures."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

NAMES = {'input': ('attn_post_norm', np.float32, 2048),
         'ids': ('ffn_moe_topk', np.int32, 8),
         'scores': ('ffn_moe_weights_norm', np.float32, 8),
         'hidden': ('ffn_moe_swiglu', np.float32, 8 * 512),
         'down': ('ffn_moe_down', np.float32, 8 * 2048)}


def load(prefix):
    tok = np.fromfile(str(prefix) + '.tokens', dtype=np.int32)
    arrays, hashes = {}, {}
    for key, (name, dtype, width) in NAMES.items():
        path = Path(str(prefix) + '.0.' + name + '-0.bin')
        hashes[key] = hashlib.sha256(path.read_bytes()).hexdigest()
        arrays[key] = np.fromfile(path, dtype=dtype).reshape(len(tok), width)
    arrays['hidden'] = arrays['hidden'].reshape(len(tok), 8, 512)
    arrays['down'] = arrays['down'].reshape(len(tok), 8, 2048)
    assert ((arrays['ids'] >= 0) & (arrays['ids'] < 256)).all()
    assert np.all(np.diff(np.sort(arrays['ids'], axis=1), axis=1) > 0)
    assert np.isfinite(arrays['scores']).all() and np.isfinite(arrays['down']).all()
    assert np.allclose(arrays['scores'].sum(axis=1), 1, atol=1e-5)
    return tok, arrays, hashes


def measure(arrays):
    a = arrays['scores'].astype(np.float64)
    out = arrays['down'].astype(np.float64)
    weighted = out * a[:, :, None]
    full = weighted.sum(axis=1)
    denom = (full * full).sum()
    order = np.argsort(-a, axis=1, kind='stable')
    rms = {}
    renormalized_rms = {}
    token_errors = {}
    for k in (1, 2, 4, 6, 7):
        keep = np.take_along_axis(weighted, order[:, :k, None], axis=1).sum(axis=1)
        diff = full - keep
        rms[str(k)] = float(np.sqrt((diff * diff).sum() / denom))
        divisor = np.take_along_axis(a, order[:, :k], axis=1).sum(axis=1)
        adjusted = full - keep / divisor[:, None]
        renormalized_rms[str(k)] = float(np.sqrt((adjusted * adjusted).sum() / denom))
        token_errors[str(k)] = np.sqrt((diff * diff).sum(axis=1) / np.maximum((full * full).sum(axis=1), 1e-30)).tolist()
    # A free hindsight choice of the best individual output is a lower bound for
    # a one-expert selector restricted to returning exactly that weighted output.
    oracle = ((full[:, None, :] - weighted) ** 2).sum(axis=2).min(axis=1)
    rank_share = [np.sort(a, axis=1)[:, ::-1][:, :k].sum(axis=1) for k in (1, 2, 4)]
    return dict(n=len(a), score_share={str(k): dict(mean=float(v.mean()), p10=float(np.quantile(v, .1)),
                     p50=float(np.median(v)), p90=float(np.quantile(v, .9)))
                     for k, v in zip((1, 2, 4), rank_share)},
                energy_share_top1=float(np.mean(np.max(a*a, axis=1) / (a*a).sum(axis=1))),
                routed_output_relative_rms=rms,
                routed_output_relative_rms_renormalized=renormalized_rms,
                routed_output_relative_rms_best_one=float(np.sqrt(oracle.sum() / denom)),
                per_token_error_quantiles={k: dict(p10=float(np.quantile(v, .1)), p50=float(np.median(v)),
                    p90=float(np.quantile(v, .9))) for k, v in token_errors.items()},
                first_16_use_fraction=float((arrays['ids'] < 16).mean()),
                distinct_experts=int(np.unique(arrays['ids']).size),
                unique_route_count=int(np.unique(np.sort(arrays['ids'], axis=1), axis=0).shape[0]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = {}
    for split in ('train', 'held'):
        prefix = args.data / split
        tok, arr, hashes = load(prefix)
        result[split] = dict(token_sha256=hashlib.sha256(tok.tobytes()).hexdigest(), tensor_sha256=hashes,
                             text_sha256=hashlib.sha256((args.data / (split + '.txt')).read_bytes()).hexdigest(),
                             **measure(arr))
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({s: {k: v for k, v in r.items() if k not in ('per_token_error_quantiles', 'tensor_sha256')}
                      for s, r in result.items()}, indent=2))


if __name__ == '__main__':
    main()
