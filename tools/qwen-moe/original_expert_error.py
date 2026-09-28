#!/usr/bin/env python3
"""Decompose installed-vs-official expert error on actual layer-0 routed producers."""
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
OUTPUT = ROOT / 'original-expert-error/receipt.json'


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def bf16(path, shape):
    return (np.memmap(path, mode='r', dtype='<u2', shape=shape).astype(np.uint32) << 16).view(np.float32)


def ratio(delta, reference):
    return float(np.linalg.norm(delta.astype(np.float64).ravel()) / np.linalg.norm(reference.astype(np.float64).ravel()))


def main():
    traffic_path, cap_receipt = ROOT / 'traffic.json', CAP / 'receipt.json'
    inventory = json.loads(traffic_path.read_text())
    captured = json.loads(cap_receipt.read_text())
    model_hash = json.loads((ROOT / 'acquisition.json').read_text())['sha256']
    assert model_hash == captured['model_sha256']
    with MODEL.open('rb') as model:
        assert hashlib.sha256(model.read(inventory['header_bytes'])).hexdigest() == inventory['header_sha256']
    meta = {t['name']: t for t in inventory['tensors']}
    header = (inventory['header_bytes'] + 31) // 32 * 32
    lib = ctypes.CDLL(str(LIB))
    specs = (('gate', 'Q4_K', 144, (512, 2048)), ('up', 'Q4_K', 144, (512, 2048)),
             ('down', 'Q5_K', 176, (2048, 512)))
    bank = {}
    payload_hash = {}
    for name, typ, block, shape in specs:
        t = meta[f'blk.0.ffn_{name}_exps.weight']
        assert t['type'] == typ and t['bytes'] == 256 * 2048 * 512 // 256 * block
        raw = np.memmap(MODEL, dtype='u1', mode='r', offset=header + t['offset'], shape=(16, t['bytes']//256))
        payload_hash[name] = hashlib.sha256(raw).hexdigest()
        fun = getattr(lib, 'dequantize_row_' + typ[:2].lower() + '_K')
        fun.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int64)
        fun.restype = None
        out = np.empty((16, *shape), dtype=np.float32)
        for expert in range(16):
            fun(raw[expert].ctypes.data_as(ctypes.c_void_p), out[expert].ctypes.data_as(ctypes.POINTER(ctypes.c_float)), out[expert].size)
        bank[name] = out
    official = bf16(OFFICIAL / 'gate_up_proj.bf16', (16, 1024, 2048))
    official_down = bf16(OFFICIAL / 'down_proj.bf16', (16, 2048, 512))
    assert all(json.loads((OFFICIAL / (kind + '.json')).read_text())['sha256'] == sha(OFFICIAL / (kind + '.bf16'))
               for kind in ('gate_up_proj', 'down_proj'))
    panels = {}
    file_hash = {}
    for split in ('train', 'held'):
        files = {key: CAP / f'{split}.layer-0.{node}.{suffix}' for key, node, suffix in (
            ('x', 'attn_post_norm', 'f32'), ('ids', 'ffn_moe_topk', 'i32'),
            ('scores', 'ffn_moe_weights_norm', 'f32'), ('native_down', 'ffn_moe_down', 'f32'))}
        file_hash[split] = {key: sha(path) for key, path in files.items()}
        assert all(captured['splits'][split]['files_sha256'][path.name] == file_hash[split][key]
                   for key, path in files.items())
        n = captured['splits'][split]['tokens']
        x = np.memmap(files['x'], dtype='<f4', mode='r', shape=(n, 2048))
        ids = np.memmap(files['ids'], dtype='<i4', mode='r', shape=(n, 8))
        scores = np.memmap(files['scores'], dtype='<f4', mode='r', shape=(n, 8))
        native_down = np.memmap(files['native_down'], dtype='<f4', mode='r', shape=(n, 8, 2048))
        assert np.allclose(scores.sum(axis=1), 1, atol=1e-5) and np.isfinite(x).all()
        selected = np.zeros((n, 2048), dtype=np.float64)
        delta = {k: np.zeros_like(selected) for k in ('both', 'gate_up', 'down')}
        counts = {}
        weight_error = {}
        slot_error = {k: [] for k in delta}
        for expert in sorted(set(ids.flat) & set(range(16))):
            rows, slots = np.where(ids == expert)
            counts[str(expert)] = len(rows)
            inp = np.asarray(x[rows], dtype=np.float32)
            def hidden(gate, up):
                g = inp @ gate.T
                u = inp @ up.T
                return ((g / (1 + np.exp(-g))) * u).astype(np.float32)
            h0 = hidden(official[expert, :512], official[expert, 512:])
            hq = hidden(bank['gate'][expert], bank['up'][expert])
            base = h0 @ official_down[expert].T
            arms = {'both': hq @ bank['down'][expert].T,
                    'gate_up': hq @ official_down[expert].T,
                    'down': h0 @ bank['down'][expert].T}
            scale = scores[rows, slots].astype(np.float64)[:, None]
            np.add.at(selected, rows, scale * base.astype(np.float64))
            for name, value in arms.items():
                d = value.astype(np.float64) - base.astype(np.float64)
                np.add.at(delta[name], rows, scale * d)
                slot_error[name].extend((np.linalg.norm(d, axis=1) / np.linalg.norm(base.astype(np.float64), axis=1)).tolist())
            weight_error[str(expert)] = {'gate': ratio(bank['gate'][expert] - official[expert, :512], official[expert, :512]),
                                         'up': ratio(bank['up'][expert] - official[expert, 512:], official[expert, 512:]),
                                         'down': ratio(bank['down'][expert] - official_down[expert], official_down[expert]),
                                         'hidden': ratio(hq - h0, h0),
                                         'native_vs_offline_q_down': ratio(np.asarray(native_down[rows, slots]) - arms['both'], arms['both'])}
        full_native = np.einsum('te,ted->td', scores.astype(np.float64), native_down.astype(np.float64))
        norm = np.linalg.norm(full_native)
        panels[split] = {'tokens': n, 'matched_assignments': sum(counts.values()), 'matched_tokens': int(np.count_nonzero(np.any(ids < 16, axis=1))),
                         'counts': counts, 'weight_error_by_expert': weight_error,
                         'selected_original_norm': float(np.linalg.norm(selected)), 'native_complete_sum_norm': float(norm),
                         'arms': {name: {'delta_norm': float(np.linalg.norm(d)),
                                        'relative_matched_original_sum': ratio(d, selected),
                                        'relative_complete_native_sum': float(np.linalg.norm(d)/norm),
                                        'slot_relative_rms': float(np.sqrt(np.mean(np.square(slot_error[name])))),
                                        'slot_relative_median': float(np.median(slot_error[name]))} for name, d in delta.items()},
                         'gate_down_error_interaction': ratio(delta['both'] - delta['gate_up'] - delta['down'], selected)}
    result = {'contract': 'Layer-0 actual GGUF producer/router; for selected experts 0..15 compare official BF16 real-valued matmul/SwiGLU with decoded installed Q4_K gate/up and Q5_K down. FP32 BLAS products, FP64 weighted accumulation. Other routed branches held fixed; not native FP32 identity, a full BF16 model, complete-model NLL or inference speed.',
              'source_sha256': sha(Path(__file__)), 'model_sha256': model_hash,
              'traffic_sha256': sha(traffic_path), 'capture_receipt_sha256': sha(cap_receipt),
              'capture_files_sha256': file_hash, 'decoder_sha256': sha(LIB),
              'original_sha256': {kind: sha(OFFICIAL / (kind + '.bf16')) for kind in ('gate_up_proj', 'down_proj')},
              'gguf_first16_payload_sha256': payload_hash, 'panels': panels}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({split: {k: p[k] for k in ('matched_assignments','matched_tokens','arms','gate_down_error_interaction')} for split,p in panels.items()}, indent=2))
    print(OUTPUT)

if __name__ == '__main__':
    main()
