#!/usr/bin/env python3
"""Reconcile Qwen GL2C zero-valued dispatches by profiler selection, not inferred DRAM bytes."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import q8_native


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    rows = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            ident = int(row["Dispatch_Id"])
            entry = rows.setdefault(ident, {
                "name": row["Kernel_Name"], "grid": int(row["Grid_Size"]),
                "start": int(row["Start_Timestamp"]), "end": int(row["End_Timestamp"]),
                "values": {},
            })
            metric = row["Counter_Name"]
            if metric in entry["values"]:
                raise ValueError(f"duplicate {ident} {metric}")
            entry["values"][metric] = float(row["Counter_Value"])
    for entry in rows.values():
        entry["positive"] = any(entry["values"].values())
        entry["duration_us"] = (entry["end"] - entry["start"]) / 1000
    return rows


def runs(rows):
    result = []
    for ident, row in sorted(rows.items(), key=lambda item: item[1]["start"]):
        positive = row["positive"]
        if result and result[-1]["positive"] == positive:
            result[-1]["last_dispatch"] = ident
            result[-1]["count"] += 1
        else:
            result.append({"positive": positive, "first_dispatch": ident,
                           "last_dispatch": ident, "count": 1})
    return result


def summarize(path, rows):
    q = {ident: row for ident, row in rows.items() if "mul_mat_vec_q<" in row["name"]}
    head = [ident for ident, row in q.items() if row["grid"] == 7946240]
    return {
        "path": str(path), "sha256": digest(path), "dispatches": len(rows),
        "quantized_calls": len(q), "positive_quantized": sum(r["positive"] for r in q.values()),
        "counter_names": sorted(set().union(*(r["values"].keys() for r in rows.values()))),
        "runs": runs(rows), "head_dispatches": head,
        "head_positive": [q[i]["positive"] for i in head],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--graph-off", type=Path, required=True)
    parser.add_argument("--one-broad", type=Path, required=True)
    parser.add_argument("--one-qonly", type=Path, required=True)
    parser.add_argument("--four-qonly", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--original-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = {name: getattr(args, name.replace("-", "_")) for name in
             ("original", "graph-off", "one-broad", "one-qonly", "four-qonly")}
    samples = {name: read(path) for name, path in paths.items()}
    original_receipt = json.loads(args.original_receipt.read_text())
    if digest(args.binary) != original_receipt["binary_hashes"]["llama-bench"]:
        raise ValueError("control binary does not match original profile receipt")
    original_q = {i: r for i, r in samples["original"].items() if "mul_mat_vec_q<" in r["name"]}
    comparisons = {}
    for name in ("graph-off", "one-broad", "one-qonly", "four-qonly"):
        quantized = {i: r for i, r in samples[name].items() if "mul_mat_vec_q<" in r["name"]}
        if quantized.keys() != original_q.keys():
            raise ValueError(f"{name}: quantized dispatch IDs do not match original")
        if any((r["grid"], r["name"]) != (original_q[i]["grid"], original_q[i]["name"])
               for i, r in quantized.items()):
            raise ValueError(f"{name}: quantized dispatch geometry differs")
        comparisons[name] = {
            "matched_quantized_dispatches": len(quantized),
            "recovered_original_zero_quantized": sum(
                not original_q[i]["positive"] and r["positive"] for i, r in quantized.items()),
            "lost_original_positive_quantized": sum(
                original_q[i]["positive"] and not r["positive"] for i, r in quantized.items()),
        }
    narrow = q8_native.analyze(paths["four-qonly"], args.inventory, 2)
    report = {
        "source_sha256": digest(Path(__file__)),
        "inventory_sha256": digest(args.inventory),
        "model_header_sha256": json.loads(args.inventory.read_text())["header_sha256"],
        "binary_sha256": digest(args.binary),
        "original_receipt_sha256": digest(args.original_receipt),
        "wrapper_log_sha256": {
            name: digest(paths[name].parent.parent / "wrapper.log")
            for name in paths if name != "original"
        },
        "inputs": {name: summarize(paths[name], samples[name]) for name in paths},
        "comparisons": comparisons,
        "narrow_family_read_counter_to_one_image": [
            {"family": x["family"], "grid_or_width": x["width_or_grid"],
             "calls": x["calls"], "positive": x["nonzero_calls"],
             "counter_ratio": x["counter_to_image_ratio"]} for x in narrow["quant_groups"]
        ],
        "interpretation": "Narrow quantized-kernel profiler selection recovers all omitted dispatch counters; graph disable and reducing to one counter do not. A GL2C read-request ratio is not independently calibrated DRAM bytes."
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
