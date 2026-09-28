#!/usr/bin/env python3
"""Actual routed-output omission geometry across the complete Qwen MoE layer stack."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

NAMES = ("attn_post_norm", "ffn_moe_topk", "ffn_moe_weights_norm", "ffn_moe_down")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def scan(root, split):
    prefix = root / split
    tokens_path = root / (split + ".tokens")
    n = tokens_path.stat().st_size // 4
    hashes = {tokens_path.name: digest(tokens_path)}
    layers = []
    for layer in range(40):
        paths = [root / f"{split}.layer-{layer}.{name}.{'i32' if name == 'ffn_moe_topk' else 'f32'}" for name in NAMES]
        for path in paths:
            hashes[path.name] = digest(path)
        producer = np.fromfile(paths[0], dtype='<f4').reshape(n, 2048)
        ids = np.fromfile(paths[1], dtype='<i4').reshape(n, 8)
        scores = np.fromfile(paths[2], dtype='<f4').reshape(n, 8).astype(np.float64)
        down = np.fromfile(paths[3], dtype='<f4').reshape(n, 8, 2048).astype(np.float64)
        assert np.isfinite(producer).all() and np.isfinite(down).all() and np.isfinite(scores).all()
        assert np.all((ids >= 0) & (ids < 256)) and np.all(np.sort(ids, axis=1)[:, 1:] != np.sort(ids, axis=1)[:, :-1])
        assert np.all(scores >= 0) and np.all(scores[:, :-1] >= scores[:, 1:])
        weighted = down * scores[:, :, None]
        reference = weighted.sum(axis=1)
        denominator = np.square(reference).sum()
        errors = {}
        for count in (1, 2, 4):
            missing = weighted[:, -count:, :].sum(axis=1)
            errors[str(count)] = float(np.sqrt(np.square(missing).sum() / denominator))
        candidate = weighted[:, :, :]
        omissions = np.square(candidate).sum(axis=2)
        oracle = np.sqrt(np.min(omissions, axis=1).sum() / denominator)
        relative = np.sqrt(omissions / np.square(reference).sum(axis=1)[:, None])
        layers.append({"layer": layer, "tokens": n, "distinct_experts": int(np.unique(ids).size),
                       "best_one_below_1pct": int(np.sum(np.min(relative, axis=1) < .01)),
                       "best_one_below_5pct": int(np.sum(np.min(relative, axis=1) < .05)),
                       "lowest_one_below_1pct": int(np.sum(relative[:, -1] < .01)),
                       "lowest_one_below_5pct": int(np.sum(relative[:, -1] < .05)),
                       "relative_rms_omit_lowest": errors, "relative_rms_free_best_one": float(oracle),
                       "score_lowest_mean": float(scores[:, -1].mean()),
                       "score_highest_mean": float(scores[:, 0].mean()),
                       "producer_rms": float(np.sqrt(np.square(producer.astype(np.float64)).mean())),
                       "reference_norm2": float(denominator)})
    return {"tokens": n, "token_sha256": hashes[tokens_path.name], "layers": layers, "files_sha256": hashes}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = {name: scan(args.directory, name) for name in ("train", "held")}
    for name, data in results.items():
        print(name, data['tokens'], "median one-skip RMS", np.median([r['relative_rms_omit_lowest']['1'] for r in data['layers']]),
              "layers <1%", sum(r['relative_rms_omit_lowest']['1'] < .01 for r in data['layers']))
    receipt = {"contract": "native callback producer, topk, normalized scores and per-slot down; FP64 offline sum/omission, not full-model logits",
               "source_sha256": digest(args.source), "analysis_sha256": digest(Path(__file__)), "binary_sha256": digest(args.binary),
               "model_sha256": digest(args.model), "splits": results}
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
