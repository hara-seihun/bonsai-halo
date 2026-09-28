#!/usr/bin/env python3
"""Finite FP64 oracle: can the installed shared expert absorb one missing routed output?"""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path

import numpy as np

BASE = Path('../../data/qwen-moe')
CAP = BASE / 'all-producers'
MODEL = BASE / 'Qwen3.6-35B-A3B-UD-Q4_K_M.gguf'
LIB = (BASE / 'runtime/current/bin/libggml-base.so').resolve()
FULL_BYTES = 2_626_187_904


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda: f.read(1 << 20), b''):
            h.update(part)
    return h.hexdigest()


def capture(split, layer, key, suffix, shape, provenance):
    path = CAP / f'{split}.layer-{layer}.{key}.{suffix}'
    assert path.stat().st_size == np.prod(shape) * 4
    assert sha(path) == provenance['splits'][split]['files_sha256'][path.name]
    x = np.memmap(path, mode='r', dtype='<f4' if suffix == 'f32' else '<i4', shape=shape)
    return x.astype(np.float64) if suffix == 'f32' else x, path.name


def weights(layer, metadata, header, decoder):
    matrices = {}
    for name, shape in (('gate', (512, 2048)), ('up', (512, 2048)), ('down', (2048, 512))):
        tensor = metadata[f'blk.{layer}.ffn_{name}_shexp.weight']
        assert tensor['type'] == 'Q8_0' and tensor['bytes'] == np.prod(shape) // 32 * 34
        raw = np.memmap(MODEL, dtype=np.uint8, mode='r', offset=header+tensor['offset'], shape=(tensor['bytes'],))
        out = np.empty(shape, dtype=np.float32)
        decoder(raw.ctypes.data_as(ctypes.c_void_p), out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), out.size)
        matrices[name] = out.astype(np.float64)
    tensor = metadata[f'blk.{layer}.ffn_gate_inp_shexp.weight']
    assert tensor['type'] == 'F32' and tensor['bytes'] == 8192
    gate = np.memmap(MODEL, dtype='<f4', mode='r', offset=header+tensor['offset'], shape=(2048,)).astype(np.float64)
    return matrices, gate


def measure(split, layer, m, gate_w, provenance):
    n = provenance['splits'][split]['tokens']
    assert n == 64
    x, xp = capture(split, layer, 'attn_post_norm', 'f32', (n, 2048), provenance)
    scores, sp = capture(split, layer, 'ffn_moe_weights_norm', 'f32', (n, 8), provenance)
    down, dp = capture(split, layer, 'ffn_moe_down', 'f32', (n, 8, 2048), provenance)
    ids, ip = capture(split, layer, 'ffn_moe_topk', 'i32', (n, 8), provenance)
    assert np.isfinite(x).all() and np.isfinite(scores).all() and np.isfinite(down).all()
    assert np.allclose(scores.sum(axis=1), 1, atol=1e-5)
    assert ((ids >= 0) & (ids < 256)).all()
    g = x @ m['gate'].T
    u = x @ m['up'].T
    h = (g / (1 + np.exp(-g))) * u
    shared = (h @ m['down'].T) * (1 / (1 + np.exp(-(x @ gate_w))))[:, None]
    v = scores[:, :, None] * down
    target = v.sum(axis=1) + shared
    norm2 = np.einsum('ij,ij->i', target, target)
    s2 = np.einsum('ij,ij->i', shared, shared)
    dot = np.einsum('ijk,ik->ij', v, shared)
    v2 = np.einsum('ijk,ijk->ij', v, v)
    assert np.all(norm2 > 0) and np.all(s2 > 0)
    alpha = dot / s2[:, None]
    error = np.sqrt(np.maximum(0, v2 - dot*alpha) / norm2[:, None])
    omitted = np.sqrt(v2 / norm2[:, None])
    index = np.argmin(error, axis=1)
    rows = np.arange(n)
    direct = np.linalg.norm(v[rows, index] - alpha[rows, index, None] * shared, axis=1) / np.sqrt(norm2)
    assert np.max(np.abs(error[rows, index] - direct)) < 1e-8
    nonnegative = np.maximum(-1, alpha)
    nonnegative_error = np.sqrt(np.maximum(0, v2 - 2*nonnegative*dot + nonnegative**2*s2[:, None]) / norm2[:, None])
    result = dict(layer=layer, split=split, input_sha256={name: provenance['splits'][split]['files_sha256'][name] for name in (xp,sp,dp,ip)},
                  norm2=norm2.tolist(),
                  omission_min=np.min(omitted, axis=1).tolist(),
                  free_shared_min=np.min(error, axis=1).tolist(),
                  nonnegative_shared_min=np.min(nonnegative_error, axis=1).tolist(),
                  chosen_ids=ids[rows, index].tolist(),
                  selected_alpha=alpha[rows, index].tolist(),
                  shared_norm_ratio_median=float(np.median(np.sqrt(s2/norm2))),
                  selected_abs_cosine_median=float(np.median(np.abs(dot[rows,index]/np.sqrt(v2[rows,index]*s2)))))
    return result


def summary(parts, costs):
    result = {}
    for split in ('train', 'held'):
        entries = [p[split] for p in parts]
        norm2 = np.array([e['norm2'] for e in entries])
        result[split] = {}
        for key in ('omission_min', 'free_shared_min', 'nonnegative_shared_min'):
            errors = np.array([e[key] for e in entries])
            assert errors.shape == (40, 64)
            count = {str(t): int(np.sum(errors <= t)) for t in (.01, .05, .10)}
            saving = {str(t): float(100*np.sum((errors <= t)*costs[:, None])/ (64*FULL_BYTES)) for t in (.01, .05, .10)}
            result[split][key] = dict(median=float(np.median(errors)), aggregate_rms=float(np.sqrt(np.sum(errors**2 * norm2)/np.sum(norm2))),
                                       count_le=count, conditional_complete_bytes_percent=saving)
        result[split]['median_selected_abs_cosine'] = float(np.median([e['selected_abs_cosine_median'] for e in entries]))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--stop', type=int, default=40)
    p.add_argument('--out', type=Path, default=BASE / 'all-layer-shared-compensation')
    p.add_argument('--aggregate', action='store_true')
    a = p.parse_args()
    assert 0 <= a.start < a.stop <= 40
    a.out.mkdir(parents=True, exist_ok=True)
    provenance_path = CAP / 'receipt.json'
    traffic_path = BASE / 'traffic.json'
    provenance = json.loads(provenance_path.read_text())
    traffic = json.loads(traffic_path.read_text())
    costs = np.array([traffic['layers'][str(i)]['bank_bytes']//256 for i in range(40)], dtype=np.int64)
    assert costs.sum()*8 == traffic['active_routed_bytes_per_token']
    source_hash = sha(Path(__file__))
    if a.aggregate:
        parts = [json.loads((a.out / f'layer-{i}.json').read_text()) for i in range(40)]
        assert [x['layer'] for x in parts] == list(range(40))
        assert all(x['source_sha256'] == source_hash for x in parts)
        assert all(x['capture_receipt_sha256'] == sha(provenance_path) for x in parts)
        receipt = dict(contract='FP64 projection of each unchanged captured score-weighted routed vector onto the offline Q8_0 shared branch; one omission, free per-token hindsight choice and scalar. Native FP32 bits, language quality, decision cost and physical traffic are not claimed.',
                       model_sha256=provenance['model_sha256'], library_sha256=sha(LIB), source_sha256=source_hash,
                       capture_receipt_sha256=sha(provenance_path), traffic_sha256=sha(traffic_path),
                       expert_image_bytes_by_layer=costs.tolist(), layer_receipts_sha256={f'layer-{i}.json':sha(a.out / f'layer-{i}.json') for i in range(40)},
                       panels=summary(parts, costs))
        (a.out / 'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
        print(json.dumps(receipt['panels'], indent=2))
        return
    decoder = ctypes.CDLL(str(LIB)).dequantize_row_q8_0
    decoder.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int64)
    decoder.restype = None
    metadata = {t['name']:t for t in traffic['tensors']}
    header = (traffic['header_bytes'] + 31)//32*32
    for layer in range(a.start, a.stop):
        matrices, gate = weights(layer, metadata, header, decoder)
        receipt = dict(layer=layer, source_sha256=source_hash, capture_receipt_sha256=sha(provenance_path),
                       train=measure('train', layer, matrices, gate, provenance),
                       held=measure('held', layer, matrices, gate, provenance))
        (a.out / f'layer-{layer}.json').write_text(json.dumps(receipt, indent=2)+'\n')
        print(layer, receipt['held']['free_shared_min'][:2], flush=True)


if __name__ == '__main__':
    main()
