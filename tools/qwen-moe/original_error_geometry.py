#!/usr/bin/env python3
"""Measure actual-route expert quantization error geometry against original BF16."""
import ctypes
import hashlib
import json
from pathlib import Path

import numpy as np
from original_expert_error import ROOT, CAP, OFFICIAL, MODEL, LIB, bf16, sha

OUT = ROOT / 'original-expert-error/geometry.json'


def norm(x):
    return float(np.linalg.norm(x.ravel()))


def geometry(terms):
    a, b = terms
    aa, bb = np.vdot(a.ravel(), a.ravel()).real, np.vdot(b.ravel(), b.ravel()).real
    ab = np.vdot(a.ravel(), b.ravel()).real
    return {'a_norm': float(np.sqrt(aa)), 'b_norm': float(np.sqrt(bb)),
            'sum_norm': float(np.sqrt(aa + bb + 2*ab)),
            'cosine': float(ab / np.sqrt(aa*bb))}


def main():
    traffic = ROOT / 'traffic.json'
    capture = CAP / 'receipt.json'
    model_sha = json.loads((ROOT / 'acquisition.json').read_text())['sha256']
    cap = json.loads(capture.read_text())
    assert cap['model_sha256'] == model_sha
    inventory = json.loads(traffic.read_text())
    with MODEL.open('rb') as f:
        assert hashlib.sha256(f.read(inventory['header_bytes'])).hexdigest() == inventory['header_sha256']
    tensors = {t['name']: t for t in inventory['tensors']}
    lib = ctypes.CDLL(str(LIB))
    bank = {}
    payload_hashes = {}
    offset = (inventory['header_bytes'] + 31) // 32 * 32
    for name, typ, block, shape in (('gate','Q4_K',144,(512,2048)),
                                     ('up','Q4_K',144,(512,2048)),
                                     ('down','Q5_K',176,(2048,512))):
        t = tensors[f'blk.0.ffn_{name}_exps.weight']
        assert t['type'] == typ and t['bytes'] == 256*2048*512//256*block
        raw = np.memmap(MODEL, mode='r', dtype='u1', offset=offset+t['offset'], shape=(16,t['bytes']//256))
        payload_hashes[name] = hashlib.sha256(raw).hexdigest()
        fun = getattr(lib, 'dequantize_row_' + typ[:2].lower() + '_K')
        fun.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int64)
        fun.restype = None
        decoded = np.empty((16,*shape), dtype=np.float32)
        for e in range(16):
            fun(raw[e].ctypes.data_as(ctypes.c_void_p), decoded[e].ctypes.data_as(ctypes.POINTER(ctypes.c_float)), decoded[e].size)
        bank[name] = decoded
    original = bf16(OFFICIAL/'gate_up_proj.bf16', (16,1024,2048))
    original_down = bf16(OFFICIAL/'down_proj.bf16', (16,2048,512))
    assert all(json.loads((OFFICIAL/(k+'.json')).read_text())['sha256'] == sha(OFFICIAL/(k+'.bf16'))
               for k in ('gate_up_proj','down_proj'))
    panels, input_hashes = {}, {}
    for split in ('train','held'):
        files = {kind: CAP/f'{split}.layer-0.{node}.{ext}' for kind,node,ext in (
            ('x','attn_post_norm','f32'),('ids','ffn_moe_topk','i32'),
            ('scores','ffn_moe_weights_norm','f32'))}
        input_hashes[split] = {k:sha(v) for k,v in files.items()}
        assert all(cap['splits'][split]['files_sha256'][p.name] == input_hashes[split][k]
                   for k,p in files.items())
        n = cap['splits'][split]['tokens']
        x = np.memmap(files['x'],mode='r',dtype='<f4',shape=(n,2048))
        ids = np.memmap(files['ids'],mode='r',dtype='<i4',shape=(n,8))
        scores = np.memmap(files['scores'],mode='r',dtype='<f4',shape=(n,8))
        assert np.isfinite(x).all() and np.isfinite(scores).all()
        assert np.allclose(scores.sum(axis=1),1,atol=1e-5)
        gate = np.zeros((n,2048), dtype=np.float64)
        down = np.zeros_like(gate)
        interaction = np.zeros_like(gate)
        exact = np.zeros_like(gate)
        solo_energy = 0.
        matched = 0
        matched_per_token = np.count_nonzero(ids < 16, axis=1)
        for e in sorted(set(ids.flat) & set(range(16))):
            rows, slots = np.where(ids == e)
            matched += len(rows)
            inp = np.asarray(x[rows],dtype=np.float32)
            def hidden(g,u):
                a, b = inp @ g.T, inp @ u.T
                return ((a/(1+np.exp(-a)))*b).astype(np.float32)
            h0 = hidden(original[e,:512],original[e,512:])
            hq = hidden(bank['gate'][e],bank['up'][e])
            base = h0 @ original_down[e].T
            g = hq @ original_down[e].T - base
            d = h0 @ bank['down'][e].T - base
            both = hq @ bank['down'][e].T - base
            s = scores[rows,slots].astype(np.float64)[:,None]
            np.add.at(exact, rows, s*base)
            np.add.at(gate, rows, s*g)
            np.add.at(down, rows, s*d)
            np.add.at(interaction, rows, s*(both-g-d))
            solo_energy += float(np.sum((s*both.astype(np.float64))**2))
        pair = geometry((gate,down))
        combined = gate+down+interaction
        panels[split] = {'tokens':n,'matched_assignments':matched,
                         'matched_tokens':int(np.count_nonzero(matched_per_token)),
                         'tokens_with_multiple_matched_experts':int(np.count_nonzero(matched_per_token >= 2)),
                         'matched_reference_norm':norm(exact),
                         'gate_down':pair,
                         'interaction_norm':norm(interaction),
                         'interaction_to_combined_cosine':float(np.vdot(interaction.ravel(),combined.ravel()).real/(norm(interaction)*norm(combined))),
                         'combined_norm':norm(combined),
                         'combined_relative_matched_original':norm(combined)/norm(exact),
                         'sum_solo_slot_error_squared':solo_energy,
                         'cross_expert_energy_over_solo':norm(combined)**2/solo_energy,
                         'sum_without_interaction_relative':pair['sum_norm']/norm(exact)}
    result = {'contract':'Actual layer0 real-text selected 16-expert subset, installed GGUF decoded to FP32 vs official BF16, offline BLAS and FP64 scored accumulation; no native logit identity or complete-model loss. Cross-expert statistic counts matched selected slots only.',
              'source_sha256':sha(Path(__file__)), 'baseline_source_sha256':sha(Path(__file__).with_name('original_expert_error.py')),
              'model_sha256':model_sha,'traffic_sha256':sha(traffic),'capture_receipt_sha256':sha(capture),
              'decoder_sha256':sha(LIB),'gguf_first16_payload_sha256':payload_hashes,
              'original_sha256':{k:sha(OFFICIAL/(k+'.bf16')) for k in ('gate_up_proj','down_proj')},
              'inputs_sha256':input_hashes,'panels':panels}
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(panels,indent=2))
    print(OUT)


if __name__ == '__main__':
    main()
