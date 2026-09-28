#!/usr/bin/env python3
"""Finite actual-route ceiling for static Q4_K -> Q3_K gate/up selection.

Counts logical one-read image bytes, not a quantized model's quality or time.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, default=Path('../../data/qwen-moe/all-producers'))
    parser.add_argument('--traffic', type=Path, default=Path('../../data/qwen-moe/traffic.json'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    inventory = json.loads(args.traffic.read_text())
    assert inventory['expert_count'] == 256 and inventory['experts_per_token'] == 8
    old, new = 144, 110  # GGML Q4_K and Q3_K bytes per 256 weights
    weights_per_pair = 2 * 2048 * 512
    saving = weights_per_pair // 256 * (old - new)
    assert saving == 278528
    counts, hashes = {}, {}
    for split in ('train', 'held'):
        rows = []
        for layer in range(40):
            layer_info = inventory['layers'][str(layer)]
            banks = [x for x in layer_info['tensors'] if 'gate_exps' in x['name'] or 'up_exps' in x['name']]
            assert len(banks) == 2 and all(x['type'] == 'Q4_K' and x['bytes'] == weights_per_pair // 2 * 256 // 256 * old for x in banks)
            path = args.capture / f'{split}.layer-{layer}.ffn_moe_topk.i32'
            array = np.fromfile(path, dtype='<i4')
            assert array.shape == (64 * 8,) and np.all((0 <= array) & (array < 256))
            for token in array.reshape(-1, 8):
                assert len(set(map(int, token))) == 8
            rows.append(np.bincount(array, minlength=256).astype(np.int64))
            hashes[path.name] = sha(path)
        counts[split] = np.stack(rows)
        assert counts[split].sum() == 40 * 64 * 8
    # Stable ID tie-breaking is stipulated, not fitted to held outcomes.
    def select(c, k, global_budget):
        if global_budget:
            flat = c.reshape(-1)
            order = sorted(range(flat.size), key=lambda i: (-int(flat[i]), i))
            chosen = np.zeros(flat.size, dtype=bool)
            chosen[order[:40*k]] = True
            return chosen.reshape(c.shape)
        chosen = np.zeros(c.shape, dtype=bool)
        for layer in range(40):
            order = sorted(range(256), key=lambda i: (-int(c[layer, i]), i))
            chosen[layer, order[:k]] = True
        return chosen
    results = []
    for k in (0, 8, 16, 32, 64, 128, 256):
        for allocation in ('per_layer', 'global'):
            train = select(counts['train'], k, allocation == 'global')
            oracle = select(counts['held'], k, allocation == 'global')
            assert train.sum() == oracle.sum() == 40*k
            held_hits = int((train * counts['held']).sum())
            oracle_hits = int((oracle * counts['held']).sum())
            train_hits = int((train * counts['train']).sum())
            assert held_hits <= oracle_hits
            results.append(dict(experts_per_layer=k, allocation=allocation,
                                train_hits=train_hits, held_hits=held_hits, held_oracle_hits=oracle_hits,
                                held_fraction=held_hits / int(counts['held'].sum()),
                                held_oracle_fraction=oracle_hits / int(counts['held'].sum()),
                                held_logical_bytes_saved_per_token=held_hits * saving // 64,
                                held_oracle_bytes_saved_per_token=oracle_hits * saving // 64,
                                complete_bank_bytes_saved=40*k*saving,
                                held_train_unseen_selected=int((train * (counts['train'] == 0) * counts['held']).sum()),
                                held_unseen_assignments=int(((counts['train'] == 0)*counts['held']).sum()),
                                selected_per_layer=[int(x) for x in train.sum(axis=1)] if allocation == 'global' else None))
    receipt = dict(contract='64 disjoint tokens/split, 40 layers, eight selected IDs; static Q4_K gate/up -> Q3_K at 144 -> 110 bytes/256; free per-image selection and reader; one full expert image per assignment',
                   code_sha256=sha(Path(__file__)), traffic_sha256=sha(args.traffic),
                   capture_receipt_sha256=sha(args.capture/'receipt.json'), capture_array_sha256=hashes,
                   one_token_weight_stream_bytes=inventory['one_token_weight_stream_bytes'],
                   q4_q3_pair_saving_bytes=saving, total_assignments=int(counts['held'].sum()),
                   train_seen_pairs=int((counts['train']>0).sum()), held_seen_pairs=int((counts['held']>0).sum()),
                   held_train_unseen_assignments=int(((counts['train']==0)*counts['held']).sum()),
                   rows=results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True)+'\n')
    for r in results:
        print(r['experts_per_layer'], r['allocation'], r['held_hits'], r['held_oracle_hits'], r['held_logical_bytes_saved_per_token'])


if __name__ == '__main__':
    main()
