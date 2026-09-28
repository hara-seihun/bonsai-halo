#!/usr/bin/env python3
"""Fetch pinned Bonsai AWQ tensor ranges without downloading a safetensors shard.

Usage: python tools/fetch_awq_fixture.py metadata|attention|mlp|report [--output DIR]
The attention selection is layer 0 linear_attn.in_proj_qkv; the MLP selection is
layer 0 mlp.gate_proj. All files are independently reproducible from the pinned
revision; the report counts raw four-bit codes, not dequantized BF16 values.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import struct

import requests

REPO = "prism-ml/Ternary-Bonsai-27B-AWQ-4bit"
REVISION = "7f49f5d23a09087131cd50627366967f013b9f9e"
SHARD = "model-00003-of-00009.safetensors"
LAYER = "model.language_model.layers.0."
SELECTED = {
    "attention": LAYER + "linear_attn.in_proj_qkv",
    "mlp": LAYER + "mlp.gate_proj",
}
CHUNK = 8 * 1024 * 1024
OUT = Path("../../data/sglang-bonsai/fixture")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def save_json(path, obj):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def resolve(filename):
    return f"https://huggingface.co/{REPO}/resolve/{REVISION}/{filename}"


def ranged(session, url, start, end, total=None):
    with session.get(url, headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"}, timeout=(10, 35), stream=True) as response:
        response.raise_for_status()
        observed = response.headers.get("Content-Range", "")
        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", observed)
        if response.status_code != 206 or not match or (int(match[1]), int(match[2])) != (start, end):
            raise RuntimeError(f"server did not honor range {start}-{end}: HTTP {response.status_code} {observed}")
        size = int(match[3])
        if total is not None and size != total:
            raise RuntimeError(f"shard size changed: {size} != {total}")
        data = bytearray()
        for chunk in response.iter_content(1024 * 1024):
            data.extend(chunk)
            if len(data) > end - start + 1:
                raise RuntimeError("range response exceeded requested length")
        if len(data) != end - start + 1:
            raise RuntimeError(f"short range: {len(data)} != {end - start + 1}")
        return bytes(data), {"status": response.status_code, "content_range": observed, "etag": response.headers.get("ETag"), "sha256": digest(data)}


def metadata(session, out):
    api = f"https://huggingface.co/api/models/{REPO}/revision/{REVISION}?blobs=true"
    api_response = session.get(api, timeout=25)
    api_response.raise_for_status()
    info = api_response.json()
    if info["sha"] != REVISION:
        raise RuntimeError(f"revision mismatch: {info['sha']}")
    siblings = {item["rfilename"]: item for item in info["siblings"]}
    shard_info = siblings[SHARD]
    shard_size = shard_info["size"]
    for filename in ("config.json", "model.safetensors.index.json"):
        response = session.get(resolve(filename), timeout=25)
        response.raise_for_status()
        path = out / filename
        path.write_bytes(response.content)
    index = json.loads((out / "model.safetensors.index.json").read_bytes())
    config = json.loads((out / "config.json").read_bytes())
    length, receipt = ranged(session, resolve(SHARD), 0, 7, shard_size)
    header_length = struct.unpack("<Q", length)[0]
    if header_length > 100_000_000:
        raise RuntimeError(f"implausibly large safetensors header: {header_length}")
    header, header_receipt = ranged(session, resolve(SHARD), 8, 7 + header_length, shard_size)
    entries = json.loads(header)
    if any(index["weight_map"].get(name) != SHARD for name in entries if name != "__metadata__"):
        raise RuntimeError("header and index disagree on shard")
    save_json(out / "header.json", entries)
    save_json(out / "source.json", {
        "repo": REPO, "revision": REVISION, "shard": SHARD,
        "shard_size": shard_size, "shard_sha256": shard_info["lfs"]["sha256"],
        "config_sha256": digest((out / "config.json").read_bytes()),
        "index_sha256": digest((out / "model.safetensors.index.json").read_bytes()),
        "header_length": header_length, "header_sha256": digest(header),
        "data_start": 8 + header_length,
        "quantization_config": {k: config["quantization_config"][k] for k in ("quant_method", "bits", "group_size", "zero_point", "version")},
        "range_receipts": [receipt, header_receipt], "api_url": api,
    })
    print(f"Pinned {REVISION}; {SHARD}: {shard_size} bytes; header {header_length} bytes")
    for group, prefix in SELECTED.items():
        print(group, [(suffix, entries[prefix + '.' + suffix]["shape"], entries[prefix + '.' + suffix]["dtype"]) for suffix in ("qweight", "qzeros", "scales")])


def fetch(session, out, group):
    source = json.loads((out / "source.json").read_text())
    header = json.loads((out / "header.json").read_text())
    for suffix in ("qweight", "qzeros", "scales"):
        name = SELECTED[group] + "." + suffix
        entry = header[name]
        start, end = entry["data_offsets"]
        size = end - start
        if size <= 0:
            raise RuntimeError(f"bad tensor size: {name}")
        target = out / (group + "." + suffix + ".bin")
        partial = target.with_name(target.name + ".part")
        progress = target.with_name(target.name + ".ranges.json")
        if target.exists() and progress.exists():
            recorded = json.loads(progress.read_text())
            if target.stat().st_size == size and recorded.get("sha256") == digest(target.read_bytes()):
                print(f"Reused {name}: {size} bytes")
                continue
        chunks = json.loads(progress.read_text()).get("chunks", []) if progress.exists() else []
        cursor = partial.stat().st_size if partial.exists() else 0
        if cursor != sum(c["length"] for c in chunks) or cursor > size:
            raise RuntimeError(f"inconsistent partial download for {name}")
        with partial.open("ab") as file:
            while cursor < size:
                lo = source["data_start"] + start + cursor
                hi = lo + min(CHUNK, size - cursor) - 1
                data, receipt = ranged(session, resolve(SHARD), lo, hi, source["shard_size"])
                file.write(data)
                file.flush()
                chunks.append({**receipt, "length": len(data), "url": resolve(SHARD)})
                cursor += len(data)
                save_json(progress, {"name": name, "dtype": entry["dtype"], "shape": entry["shape"], "data_offsets": [start, end], "chunks": chunks})
        partial.replace(target)
        complete = json.loads(progress.read_text())
        complete["sha256"] = digest(target.read_bytes())
        complete["bytes"] = size
        save_json(progress, complete)
        print(f"Fetched {name}: {size} bytes sha256={complete['sha256']}")


def nibble_histogram(path):
    counts = [0] * 16
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            for byte, times in Counter(block).items():
                counts[byte & 15] += times
                counts[byte >> 4] += times
    return counts


def report(out):
    source = json.loads((out / "source.json").read_text())
    header = json.loads((out / "header.json").read_text())
    report_data = {"repo": REPO, "revision": REVISION, "shard": SHARD, "matrices": {}}
    for group, prefix in SELECTED.items():
        matrix = {}
        total = 0
        for suffix in ("qweight", "qzeros", "scales"):
            name = prefix + "." + suffix
            entry = header[name]
            path = out / (group + "." + suffix + ".bin")
            receipt = json.loads((out / (group + "." + suffix + ".bin.ranges.json")).read_text())
            nbytes = entry["data_offsets"][1] - entry["data_offsets"][0]
            if path.stat().st_size != nbytes or digest(path.read_bytes()) != receipt["sha256"]:
                raise RuntimeError(f"bad fixture hash/length: {path}")
            item = {"name": name, "dtype": entry["dtype"], "shape": entry["shape"], "bytes": nbytes, "sha256": receipt["sha256"]}
            if suffix != "scales":
                histogram = nibble_histogram(path)
                item["raw_nibble_histogram"] = histogram
                item["distinct_raw_codes"] = [i for i, count in enumerate(histogram) if count]
            else:
                with path.open("rb") as file:
                    bits = set()
                    while block := file.read(1024 * 1024):
                        bits.update(memoryview(block).cast("H"))
                item["distinct_bf16_bit_patterns"] = len(bits)
            total += nbytes
            matrix[suffix] = item
        q = matrix["qweight"]
        z = matrix["qzeros"]
        s = matrix["scales"]
        matrix["paid_bytes"] = total
        matrix["paid_bits_per_weight"] = 8 * total / (q["shape"][0] * q["shape"][1] * 8)
        matrix["raw_nibble_counts"] = {"qweight": sum(q["raw_nibble_histogram"]), "qzeros": sum(z["raw_nibble_histogram"])}
        matrix["scales_bytes"] = s["bytes"]
        report_data["matrices"][group] = matrix
    report_data["paid_bytes_total"] = sum(m["paid_bytes"] for m in report_data["matrices"].values())
    save_json(out / "report.json", report_data)
    lines = ["# Official Bonsai AWQ fixture", "", f"Repository: `{REPO}`, revision `{REVISION}`.",
             f"Shard `{SHARD}`: {source['shard_size']:,} bytes, published SHA-256 `{source['shard_sha256']}`.",
             "Fetched only the pinned safetensors header and six range-selected tensor payloads; see `source.json` and `*.ranges.json` for HTTP range receipts and per-tensor SHA-256.",
             "Raw nibbles are stored AWQ codes. They are not signed ternary weights or BF16 activation values. The quantization config is AWQ GEMM, four bits, group size 128, with zero points.", ""]
    for group, matrix in report_data["matrices"].items():
        lines += [f"## {group}", "", f"Paid storage: {matrix['paid_bytes']:,} bytes, {matrix['paid_bits_per_weight']:.5f} bits per weight including zeros and BF16 scales."]
        for suffix in ("qweight", "qzeros", "scales"):
            item = matrix[suffix]
            detail = f"raw codes {item['distinct_raw_codes']}, counts {item['raw_nibble_histogram']}" if suffix != "scales" else f"{item['distinct_bf16_bit_patterns']} distinct BF16 bit patterns"
            lines.append(f"- `{suffix}` `{item['shape']}` `{item['dtype']}`, {item['bytes']:,} bytes, SHA-256 `{item['sha256']}`; {detail}.")
        lines.append("")
    lines.append("These two projections are a fixture, not evidence for the distribution across the whole checkpoint or a throughput comparator. The published shard SHA-256 identifies the source object; range receipts and payload hashes identify acquired fragments, not a full-shard SHA-256 verification.")
    (out / "README.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {out / 'report.json'} and README.md; total {report_data['paid_bytes_total']:,} payload bytes")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("metadata", "attention", "mlp", "report"))
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with requests.Session() as session:
        if args.action == "metadata":
            metadata(session, args.output)
        elif args.action == "report":
            report(args.output)
        else:
            fetch(session, args.output, args.action)


if __name__ == "__main__":
    main()
