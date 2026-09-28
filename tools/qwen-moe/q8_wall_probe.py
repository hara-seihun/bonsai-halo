#!/usr/bin/env python3
"""Price ordinary native Q8 calls by installed tensor bytes, not profiler gap time.

This is an *in-trace* cost comparison; it cannot measure unprofiled wall time.
"""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import median


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze(trace, inventory, depth_zero=False):
    rows = list(csv.DictReader(trace.open(newline="")))
    head = [i for i, r in enumerate(rows) if "mul_mat_vec_q<(ggml_type)14," in r["Kernel_Name"]
            and int(r["Grid_Size_X"]) == 7946240]
    if len(head) != (8 if depth_zero else 10):
        raise ValueError(f"unexpected vocabulary head count {len(head)}")
    # At depth 1024, two setup heads precede eight timed heads. At depth
    # zero, discard the first incomplete warmup span and use seven full spans.
    first = 0 if depth_zero else 1
    spans = [(head[first + i] + 1, head[first + i + 1] + 1)
             for i in range(len(head) - first - 1)]
    if any(b - a != 1565 for a, b in spans):
        raise ValueError("unexpected decode graph or missing dispatches")
    model = json.loads(inventory.read_text())
    tensors = [r for r in model["tensors"] if r["type"] == "Q8_0" and r["name"] != "token_embd.weight"]
    shape_counts = defaultdict(lambda: defaultdict(int))
    for r in tensors:
        shape_counts[r["shape"][-1]][r["bytes"]] += 1
    expected = {512: {1114112: 100}, 2048: {1114112: 40, 8912896: 40},
                4096: {8912896: 30}, 8192: {17825792: 40}}
    if dict(shape_counts) != expected:
        raise ValueError("Q8 model inventory has changed; audit graph attribution")
    calls = []
    for tok, (lo, hi) in enumerate(spans):
        previous_end, last_gap = None, None
        q8_2048 = 0
        q8_count = defaultdict(int)
        routers = 0
        for ordinal, row in enumerate(rows[lo:hi]):
            start, end = int(row["Start_Timestamp"]), int(row["End_Timestamp"])
            if previous_end is not None and start - previous_end >= 500_000:
                last_gap = ordinal
            previous_end = end
            name = row["Kernel_Name"]
            if "topk_moe_cuda<" in name:
                routers += 1
            if "mul_mat_vec_q<(ggml_type)8," not in name:
                continue
            grid = int(row["Grid_Size_X"])
            if grid not in (16384, 65536, 131072, 262144):
                raise ValueError(f"unexpected Q8 grid {grid}")
            if grid == 65536:
                # Each layer has one large pre-expert projection and one small
                # shared-expert down projection, paired in execution order.
                # Attention layers may route before the first projection;
                # router count alone therefore does not classify the pair.
                width = 8192 if q8_2048 % 2 == 0 else 1024
                q8_2048 += 1
            elif grid == 16384:
                fused = ", true, true," in name
                width = 4096 if fused else 2048
                q8_count["fused512" if fused else "single512"] += 1
            else:
                width = {131072: 8192, 262144: 16384}[grid]
            # Installed Q8_0 physical image sizes, from the GGUF inventory.
            if grid == 16384:
                size = 2228224 if width == 4096 else 1114112
            elif grid == 65536:
                size = 8912896 if width == 8192 else 1114112
            elif grid == 131072:
                size = 8912896
            else:
                size = 17825792
            distance = ordinal - last_gap if last_gap is not None else None
            calls.append({"token": tok, "ordinal": ordinal, "router_count_before": routers,
                          "group": str(size), "grid": grid, "bytes": size,
                          "us": (end - start) / 1000, "distance_after_gap": distance,
                          "ordinary": distance is None or distance > 64})
        large = [c["us"] for c in calls if c["token"] == tok and c["grid"] == 65536 and c["bytes"] == 8912896]
        small = [c["us"] for c in calls if c["token"] == tok and c["grid"] == 65536 and c["bytes"] == 1114112]
        if len(large) != 40 or len(small) != 40 or median(large) <= median(small):
            raise ValueError(f"token {tok}: paired 2048 projections disagree with image inventory")
        if routers != 40 or q8_2048 != 80 or q8_count != {"fused512": 40, "single512": 20}:
            raise ValueError(f"token {tok} incomplete graph: routers={routers}, 2048={q8_2048}, 512={dict(q8_count)}")
        if sum(c["token"] == tok for c in calls) != 210:
            raise ValueError("incomplete Q8 token")
    summary = {}
    split_at = 3 if depth_zero else 4
    for size in (1114112, 2228224, 8912896, 17825792):
        for split, tokset in (("inspection", range(split_at)), ("held", range(split_at, len(spans)))):
            samples = [c["us"] for c in calls if c["bytes"] == size and c["token"] in tokset and c["ordinary"]]
            med = median(samples)
            summary[f"{size}/{split}"] = {"calls": len(samples), "median_us": med,
                                            "effective_image_GBps": size / (med * 1000),
                                            "mean_us": sum(samples) / len(samples)}
    # An affine byte-stream model trained on the ordinary small and medium
    # populations. The largest population is an honest held-shape prediction.
    s1 = summary["1114112/inspection"]["median_us"]
    s8 = summary["8912896/inspection"]["median_us"]
    slope = (s8 - s1) / (8912896 - 1114112)
    predicted = s1 + slope * (17825792 - 1114112)
    held = summary["17825792/held"]["median_us"]
    held_token_excess = [median(c["us"] for c in calls if c["token"] == tok and
                           c["bytes"] == 17825792 and c["ordinary"]) - predicted
                         for tok in range(split_at, len(spans))]
    if not all(x > 0 for x in held_token_excess):
        raise ValueError("the held wide-shape direction is not stable")
    return {"source_sha256": sha(Path(__file__)), "trace_sha256": sha(trace),
            "inventory_sha256": sha(inventory), "model_header_sha256": model["header_sha256"],
            "traced_tokens": len(spans), "trace_depth_zero": depth_zero,
            "ordinary_gate": "more than 64 dispatches after 0.5ms gap",
            "samples": summary, "fit_inspection_us_per_byte": slope,
            "fit_inspection_intercept_us": s1 - slope * 1114112,
            "predicted_held_wide_median_us": predicted, "actual_held_wide_median_us": held,
            "held_wide_excess_us": held - predicted,
            "held_token_wide_excess_us": held_token_excess,
            "calls": calls,
            "interpretation": "Within counter-free rocprof trace only; not an unprofiled wall-time or DRAM-byte result."}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("trace", type=Path)
    p.add_argument("inventory", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--depth-zero", action="store_true", help="independent seven-warm-token trace")
    a = p.parse_args()
    result = analyze(a.trace, a.inventory, a.depth_zero)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "calls"}, indent=2))
