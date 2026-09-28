#!/usr/bin/env python3
"""Attribute layer-0 installed Q4 dormant-channel loss on actual routed producers."""
import ctypes
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path('../../data/qwen-moe')
CAP = ROOT / 'all-producers'
OFFICIAL = ROOT / 'experts/layer-0-0-16'
MODEL = ROOT / 'Qwen3.6-35B-A3B-UD-Q4_K_M.gguf'
LIB = (ROOT / 'runtime/current/bin/libggml-base.so').resolve()
MASK = ROOT / 'all-layer-dormancy/layer-0.json'
OUTPUT = ROOT / 'dormant-original-impact/receipt.json'


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def bf16(path, shape):
    return (np.memmap(path, mode='r', dtype='<u2', shape=shape).astype(np.uint32) << 16).view(np.float32)


def norm(x):
    return float(np.linalg.norm(np.asarray(x, dtype=np.float64).ravel()))


def main():
    traffic_path, cap_receipt = ROOT / 'traffic.json', CAP / 'receipt.json'
    traffic = json.loads(traffic_path.read_text())
    captured = json.loads(cap_receipt.read_text())
    mask_receipt = json.loads(MASK.read_text())
    assert mask_receipt['layer'] == 0
    assert mask_receipt['model_sha256'] == captured['model_sha256'] == json.loads((ROOT / 'acquisition.json').read_text())['sha256']
    assert mask_receipt['traffic_sha256'] == sha(traffic_path)
    with MODEL.open('rb') as model:
        assert hashlib.sha256(model.read(traffic['header_bytes'])).hexdigest() == traffic['header_sha256']
    meta = {t['name']: t for t in traffic['tensors']}
    header = (traffic['header_bytes'] + 31) // 32 * 32
    lib = ctypes.CDLL(str(LIB))
    # The mask census used an earlier library build; check its rows against
    # the current decoder below rather than equating two unrelated binaries.
    banks, payload_hashes = {}, {}
    for name, typ, shape in (('gate', 'q4_K', (512, 2048)), ('up', 'q4_K', (512, 2048)),
                             ('down', 'q5_K', (2048, 512))):
        t = meta[f'blk.0.ffn_{name}_exps.weight']
        raw = np.memmap(MODEL, dtype='u1', mode='r', offset=header + t['offset'], shape=(256, t['bytes'] // 256))
        payload_hashes[name] = sha_bytes(raw)
        decode = getattr(lib, f'dequantize_row_{typ}')
        decode.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int64)
        decode.restype = None
        out = np.empty((16, *shape), dtype=np.float32)
        for expert in range(16):
            decode(raw[expert].ctypes.data_as(ctypes.c_void_p), out[expert].ctypes.data_as(ctypes.POINTER(ctypes.c_float)), out[expert].size)
        banks[name] = out
    original = bf16(OFFICIAL / 'gate_up_proj.bf16', (16, 1024, 2048))
    original_down = bf16(OFFICIAL / 'down_proj.bf16', (16, 2048, 512))
    original_hashes = {name: sha(OFFICIAL / (name + '.bf16')) for name in ('gate_up_proj', 'down_proj')}
    for name, digest in original_hashes.items():
        assert json.loads((OFFICIAL / (name + '.json')).read_text())['sha256'] == digest
    masks = [np.unpackbits(np.frombuffer(bytes.fromhex(mask_receipt['masks_hex'][e]), dtype=np.uint8),
                           bitorder='little').astype(bool) for e in range(16)]
    for e, mask in enumerate(masks):
        assert np.array_equal(mask, ~np.any(banks['gate'][e] != 0, axis=1) & ~np.any(banks['up'][e] != 0, axis=1))
    channel_stats = {}
    for e, mask in enumerate(masks):
        gmax = np.max(np.abs(original[e, :512]), axis=1)
        umax = np.max(np.abs(original[e, 512:]), axis=1)
        dmax = np.max(np.abs(original_down[e]), axis=0)
        tiny = (gmax <= 2.0**-120) & (umax <= 2.0**-120) & (dmax <= 2.0**-120)
        channel_stats[str(e)] = {'dormant': int(mask.sum()),
            'all_three_original_tiny': int(tiny.sum()),
            'q4_dormant_not_original_tiny': int(np.count_nonzero(mask & ~tiny)),
            'original_tiny_not_q4_dormant': int(np.count_nonzero(tiny & ~mask)),
            'original_gate_absmax_dormant': float(np.max(np.abs(original[e, :512][mask]))) if np.any(mask) else 0,
            'original_up_absmax_dormant': float(np.max(np.abs(original[e, 512:][mask]))) if np.any(mask) else 0,
            'original_down_absmax_dormant': float(np.max(np.abs(original_down[e][:, mask]))) if np.any(mask) else 0,
            'original_gate_absmax_active': float(np.max(np.abs(original[e, :512][~mask]))),
            'original_up_absmax_active': float(np.max(np.abs(original[e, 512:][~mask]))),
            'original_down_absmax_active': float(np.max(np.abs(original_down[e][:, ~mask])))}
    panels, input_hashes = {}, {}
    for split in ('train', 'held'):
        files = {name: CAP / f'{split}.layer-0.{tensor}.{suffix}' for name, tensor, suffix in (
            ('x', 'attn_post_norm', 'f32'), ('ids', 'ffn_moe_topk', 'i32'),
            ('scores', 'ffn_moe_weights_norm', 'f32'))}
        input_hashes[split] = {name: sha(path) for name, path in files.items()}
        for name, path in files.items():
            assert captured['splits'][split]['files_sha256'][path.name] == input_hashes[split][name]
        n = captured['splits'][split]['tokens']
        x = np.memmap(files['x'], dtype='<f4', mode='r', shape=(n, 2048))
        ids = np.memmap(files['ids'], dtype='<i4', mode='r', shape=(n, 8))
        scores = np.memmap(files['scores'], dtype='<f4', mode='r', shape=(n, 8))
        assert np.isfinite(x).all() and np.isfinite(scores).all()
        original_sum = np.zeros((n, 2048), dtype=np.float64)
        dormant = np.zeros_like(original_sum)
        gate_error = np.zeros_like(original_sum)
        restored_error = np.zeros_like(original_sum)
        dormant_selected = 0
        per_slot = []
        counts = {}
        for e in sorted(set(ids.flat) & set(range(16))):
            rows, slots = np.where(ids == e)
            counts[str(e)] = len(rows)
            inp = np.asarray(x[rows], dtype=np.float32)
            def hidden(gate, up):
                g, u = inp @ gate.T, inp @ up.T
                return ((g / (1 + np.exp(-g))) * u).astype(np.float32)
            h0 = hidden(original[e, :512], original[e, 512:])
            hq = hidden(banks['gate'][e], banks['up'][e])
            mask = masks[e]
            assert np.all(hq[:, mask] == 0)
            assert np.all(h0[:, mask] == 0)
            restored = hq.copy()
            restored[:, mask] = h0[:, mask]
            d0 = h0 @ original_down[e].T
            dq = hq @ original_down[e].T
            dr = restored @ original_down[e].T
            # Evaluate the missing channel as its own output map; unlike dr-dq,
            # this retains the cancellation geometry before the full FP32 GEMM fold.
            dm = h0[:, mask] @ original_down[e][:, mask].T if np.any(mask) else np.zeros_like(d0)
            scale = scores[rows, slots].astype(np.float64)[:, None]
            np.add.at(original_sum, rows, scale * d0)
            np.add.at(dormant, rows, scale * dm)
            np.add.at(gate_error, rows, scale * (dq.astype(np.float64) - d0.astype(np.float64)))
            np.add.at(restored_error, rows, scale * (dr.astype(np.float64) - d0.astype(np.float64)))
            dormant_selected += int(mask.sum()) * len(rows)
            per_slot.extend((np.linalg.norm(dm.astype(np.float64), axis=1) / np.linalg.norm(d0.astype(np.float64), axis=1)).tolist())
        denom = norm(original_sum)
        dot = float(np.sum(dormant * gate_error))
        panels[split] = {'tokens': n, 'input_l2_max': float(np.max(np.linalg.norm(np.asarray(x, dtype=np.float64), axis=1))),
                         'matched_assignments': sum(counts.values()), 'matched_expert_counts': counts,
                         'dormant_selected_channels': dormant_selected,
                         'matched_selected_channels': 512 * sum(counts.values()),
                         'original_matched_sum_norm': denom,
                         'missing_original_channel_norm': norm(dormant),
                         'missing_original_channel_relative': norm(dormant) / denom,
                         'gate_up_error_relative': norm(gate_error) / denom,
                         'restored_gate_up_error_relative': norm(restored_error) / denom,
                         'dormant_error_projection_cosine': dot / (norm(dormant) * norm(gate_error)) if norm(dormant) else None,
                         'dormant_only_slot_relative_median': float(np.median(per_slot)),
                         'dormant_only_slot_relative_rms': float(np.sqrt(np.mean(np.square(per_slot))))}
    result = {'contract': 'First 16 layer-0 experts, actual GGUF producers/route scores, official BF16 reference. FP32 BLAS products and SiLU, FP64 score sums. Missing-channel output uses original BF16 down. Restoration is an oracle hybrid replacing Q4 dormant hidden coordinates only; it is neither paid nor native, and is not complete-model quality.',
              'source_sha256': sha(__file__), 'model_sha256': captured['model_sha256'],
              'traffic_sha256': sha(traffic_path), 'capture_receipt_sha256': sha(cap_receipt),
              'mask_sha256': sha(MASK), 'decoder_sha256': sha(LIB),
              'original_sha256': original_hashes, 'gguf_first16_payload_sha256': payload_hashes,
              'original_channel_stats': channel_stats,
              'capture_files_sha256': input_hashes, 'panels': panels}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(panels, indent=2))
    print(OUTPUT)


def sha_bytes(array):
    return hashlib.sha256(array).hexdigest()


if __name__ == '__main__':
    main()
