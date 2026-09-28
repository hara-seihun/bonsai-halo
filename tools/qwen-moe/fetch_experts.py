#!/usr/bin/env python3
"""Fetch contiguous BF16 expert samples from the pinned official safetensors."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import struct
import urllib.request

from acquire import OFFICIAL, OFFICIAL_REV, atomic_json


def ranged(shard, start, end):
    url = f"https://huggingface.co/{OFFICIAL}/resolve/{OFFICIAL_REV}/{shard}?range={start}-{end}"
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"})
    with urllib.request.urlopen(req, timeout=60) as response:
        cr = response.headers.get("Content-Range", "")
        if response.status != 206 or not cr.startswith(f"bytes {start}-{end}/"):
            raise RuntimeError(f"Range refused: {response.status} {cr}")
        data = response.read(end - start + 2)
        if len(data) != end - start + 1:
            raise RuntimeError(f"Range length mismatch: {len(data)}")
        return data, {"url": url, "content_range": cr, "sha256": hashlib.sha256(data).hexdigest()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("../../data/qwen-moe"))
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--first", type=int, default=0)
    p.add_argument("--count", type=int, default=16)
    args = p.parse_args()
    index = json.loads((args.root / "official/model.safetensors.index.json").read_text())
    target = args.root / "experts" / f"layer-{args.layer}-{args.first}-{args.count}"
    target.mkdir(parents=True, exist_ok=True)
    for projection in ("gate_up_proj", "down_proj"):
        name = f"model.language_model.layers.{args.layer}.mlp.experts.{projection}"
        shard = index["weight_map"][name]
        length, lr = ranged(shard, 0, 7)
        n = struct.unpack("<Q", length)[0]
        if n > 100_000_000:
            raise RuntimeError("Unreasonable safetensors header length")
        raw, hr = ranged(shard, 8, 7 + n)
        header = json.loads(raw)
        entry = header[name]
        shape = entry["shape"]
        if entry["dtype"] != "BF16" or len(shape) != 3 or args.first < 0 or args.count < 1 or args.first + args.count > shape[0]:
            raise RuntimeError(f"Unsupported tensor slice {entry}")
        per_expert = math.prod(shape[1:]) * 2
        lo, hi = entry["data_offsets"]
        if hi - lo != per_expert * shape[0]:
            raise RuntimeError("Tensor payload disagrees with dimensions")
        start = 8 + n + lo + args.first * per_expert
        size = args.count * per_expert
        ranges = [(offset, min(start + size, offset + 8 * 1024 * 1024) - 1)
                  for offset in range(start, start + size, 8 * 1024 * 1024)]
        final = target / f"{projection}.bf16"
        temp = final.with_suffix(".partial")
        h = hashlib.sha256()
        receipts = []
        with ThreadPoolExecutor(max_workers=4) as pool, temp.open("wb") as output:
            for data, receipt in pool.map(lambda pair: ranged(shard, *pair), ranges):
                output.write(data)
                h.update(data)
                receipts.append(receipt)
        temp.replace(final)
        (target / f"{projection}.header.json").write_bytes(raw)
        atomic_json(target / f"{projection}.json", dict(
            repository=OFFICIAL, revision=OFFICIAL_REV, tensor=name, shard=shard,
            dtype=entry["dtype"], original_shape=shape,
            shape=[args.count, *shape[1:]], first_expert=args.first,
            bytes=size, sha256=h.hexdigest(), path=str(final),
            header_receipts=[lr, hr], payload_ranges=receipts,
            verification="Exact pinned HTTP byte ranges and local payload hashes; not full-shard hash verification"))
        print(f"{final}: {[args.count, *shape[1:]]} BF16, {size} bytes, {h.hexdigest()}", flush=True)


if __name__ == "__main__":
    main()
