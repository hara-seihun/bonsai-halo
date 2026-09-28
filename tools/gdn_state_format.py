#!/usr/bin/env python3
"""Search storage coordinates for the gated-delta state against a real dumped state.

    tools/batch_compare --only state-dump --modes 19 --slots 1 ...   # writes state-gdn*.f32
    tools/gdn_state_format.py ../../data/bonsai2/batch-comparison/<tag> [--json out.json]

Why this is offline. A coordinate for the recurrent state is a choice of block and a choice of code,
and both are decided by what the values look like. One dump answers the whole family at once, where
every candidate measured on the device costs a kernel, a build and a GPU panel.

What it measures, and why that and not RMS error. Both consumers of a state row are dot products
against an L2-normalised vector: `S_{t-1} k_t` inside the recurrence and `S_t q_t` on the way out.
For a row `s` with elementwise error `e` and a unit vector `u` the readout error is `e . u`, whose
mean square over directions is `||e||^2 / 128` against a signal of `||s||^2 / 128`. So the number
that ranks coordinates is the **per-row relative L2 error** `||e|| / ||s||`, pooled over rows - not
the absolute RMS error, and not a relative error per element, which would weight a near-zero element
of a row as heavily as the element that carries it.

Rows are reported separately from the pool because a coordinate that is excellent on the median row
and bad on the tail is a different risk from one that is uniformly mediocre: the recurrence feeds
its own output back, so a row that is wrong is wrong for every later token of that sequence.

Everything here reproduces `kernels/gdn_state_codec.hpp` exactly: a shared exponent is
`a = max|v|` over the block, `scale = a / qmax`, `code = clamp(rint(v * (qmax / a)))`, and the value
a reader sees is `scale * code`. The kernel's block is one whole row, because a wave owns a row;
the finer blocks here are the 32- and 16-column groups that the same lane layout can also reduce
over, so they are implementable in that kernel without moving a single state element.
"""
import argparse
import json
import math
import os
import sys

import numpy as np

HV, SS = 48, 128


# ---------------------------------------------------------------- coordinates

def blocked_int(x, bits, group):
    """Shared-exponent integer code, one scale per `group` consecutive columns of a row."""
    qmax = float(2 ** (bits - 1) - 1)
    h, r, c = x.shape
    v = x.reshape(h, r, c // group, group)
    a = np.abs(v).max(axis=-1, keepdims=True)
    scale = (a / qmax).astype(np.float32)
    inv = np.where(a > 0, qmax / np.where(a > 0, a, 1.0), 0.0).astype(np.float32)
    code = np.clip(np.rint(v * inv), -qmax, qmax).astype(np.float32)
    return (scale * code).reshape(h, r, c)


def fp16(x):
    return x.astype(np.float16).astype(np.float32)


def bf16(x):
    u = x.astype(np.float32).view(np.uint32)
    # round to nearest even on the low 16 bits
    r = ((u >> 16) & 1) + 0x7FFF
    return ((u + r) & 0xFFFF0000).view(np.float32)


def column_int(x, bits):
    """Diagnostic only: one scale per state *column*, shared over the head's rows.

    Not implementable in the deployed lane layout without a cross-wave reduction - a wave owns a
    row, so a column's maximum lives in 32 different waves. It is here to say where the spread is.
    A row block and a column block cost the same 128 scales per head; if the column block wins, the
    state's dynamic range is a property of the key channel rather than of the value channel, and a
    coordinate that tracked it would have to be built somewhere other than the store path.
    """
    qmax = float(2 ** (bits - 1) - 1)
    a = np.abs(x).max(axis=1, keepdims=True)          # [head, 1, col]
    scale = (a / qmax).astype(np.float32)
    inv = np.where(a > 0, qmax / np.where(a > 0, a, 1.0), 0.0).astype(np.float32)
    return scale * np.clip(np.rint(x * inv), -qmax, qmax).astype(np.float32)


def coordinates():
    """name -> (function, bits per element including the block scales)."""
    out = {"fp16": (fp16, 16.0), "bf16": (bf16, 16.0)}
    for bits in (8, 6):
        out[f"i{bits}/col"] = ((lambda b: lambda x: column_int(x, b))(bits), bits + 32.0 / SS)
    for bits in (16, 12, 8, 6, 4):
        for group in (128, 32, 16, 8):
            # one fp32 scale per block, amortised over the block's elements
            per = bits + 32.0 / group
            out[f"i{bits}/{group}"] = ((lambda b, g: lambda x: blocked_int(x, b, g))(bits, group), per)
    return out


# ---------------------------------------------------------------- statistics

def row_error(ref, arm):
    """Per-row relative L2 error, and the pooled value over all rows with signal."""
    e = (arm.astype(np.float64) - ref.astype(np.float64)).reshape(-1, SS)
    s = ref.astype(np.float64).reshape(-1, SS)
    en = np.sqrt((e * e).sum(axis=1))
    sn = np.sqrt((s * s).sum(axis=1))
    live = sn > 0
    rel = np.zeros_like(en)
    rel[live] = en[live] / sn[live]
    pooled = math.sqrt((en[live] ** 2).sum() / (sn[live] ** 2).sum()) if live.any() else 0.0
    return rel[live], pooled


def describe(x):
    a = np.abs(x.astype(np.float64))
    rows = a.reshape(-1, SS)
    rmax = rows.max(axis=1)
    rrms = np.sqrt((rows ** 2).mean(axis=1))
    live = rmax > 0
    crest = np.zeros_like(rmax)
    crest[live] = rmax[live] / rrms[live]
    g32 = rows.reshape(-1, SS // 32, 32).max(axis=2)
    # what a per-32-column scale buys: the row scale divided by the group's own scale, per group
    spread = np.zeros_like(g32)
    ok = g32 > 0
    spread[ok] = np.repeat(rmax[:, None], SS // 32, axis=1)[ok] / g32[ok]
    return {
        "rows": int(rows.shape[0]),
        "live_rows": int(live.sum()),
        "row_max_median": float(np.median(rmax[live])) if live.any() else 0.0,
        "row_max_p99": float(np.percentile(rmax[live], 99)) if live.any() else 0.0,
        "crest_median": float(np.median(crest[live])) if live.any() else 0.0,
        "crest_p90": float(np.percentile(crest[live], 90)) if live.any() else 0.0,
        "crest_p99": float(np.percentile(crest[live], 99)) if live.any() else 0.0,
        "crest_max": float(crest.max()),
        "group32_headroom_median": float(np.median(spread[ok])) if ok.any() else 0.0,
        "group32_headroom_p90": float(np.percentile(spread[ok], 90)) if ok.any() else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="results directory holding run.json and state-gdn*.f32")
    ap.add_argument("--json", help="also write the report here")
    ap.add_argument("--heads", type=int, default=HV, help="heads to read from each dump")
    args = ap.parse_args()

    path = args.run
    if os.path.isdir(path):
        path = os.path.join(path, "run.json")
    base = os.path.dirname(os.path.abspath(path))
    run = json.loads(open(path).read())
    dumps = []
    for d in run.get("state_dump", []):
        for f in d["files"]:
            dumps.append((f["gdn_layer"], os.path.join(base, f["file"]), d.get("context_tokens")))
    if not dumps:
        print("no state dump in this run", file=sys.stderr)
        return 1

    cands = coordinates()
    report = {"run": path, "revision": run.get("git_revision"), "layers": [], "pooled": {}}
    pool_num = {k: 0.0 for k in cands}
    pool_den = 0.0
    all_rel = {k: [] for k in cands}

    print(f"state dump: {len(dumps)} layers, {args.heads} heads each, "
          f"{dumps[0][2]} teacher-forced tokens of context\n")
    for gi, f, _ctx in dumps:
        x = np.fromfile(f, dtype=np.float32, count=args.heads * SS * SS).reshape(args.heads, SS, SS)
        st = describe(x)
        entry = {"gdn_layer": gi, **st, "coordinates": {}}
        for name, (fn, bits) in cands.items():
            rel, pooled = row_error(x, fn(x))
            entry["coordinates"][name] = {
                "bits_per_element": bits,
                "rel_l2_pooled": pooled,
                "rel_l2_median": float(np.median(rel)) if len(rel) else 0.0,
                "rel_l2_p99": float(np.percentile(rel, 99)) if len(rel) else 0.0,
                "snr_db": float(-20 * math.log10(pooled)) if pooled > 0 else float("inf"),
            }
            all_rel[name].append(rel)
        report["layers"].append(entry)
        print(f"gdn layer {gi:2d}: live rows {st['live_rows']}/{st['rows']}, "
              f"row |max| median {st['row_max_median']:.4g}, crest median {st['crest_median']:.2f} "
              f"p99 {st['crest_p99']:.2f}, 32-column headroom median {st['group32_headroom_median']:.2f}")

    # Pool across layers: one number per coordinate, and the frontier ordered by bytes.
    print("\ncoordinate        bits/elt   rel L2 (pooled)   median      p99      SNR dB")
    rows = []
    for name, (_fn, bits) in cands.items():
        rel = np.concatenate(all_rel[name])
        pooled = float(np.sqrt((rel ** 2).mean()))
        rows.append((bits, name, pooled, float(np.median(rel)), float(np.percentile(rel, 99))))
        report["pooled"][name] = {"bits_per_element": bits, "rel_l2": pooled,
                                  "rel_l2_median": float(np.median(rel)),
                                  "rel_l2_p99": float(np.percentile(rel, 99)),
                                  "snr_db": float(-20 * math.log10(pooled)) if pooled > 0 else float("inf")}
    for bits, name, pooled, med, p99 in sorted(rows, key=lambda r: (-r[0], r[1])):
        snr = -20 * math.log10(pooled) if pooled > 0 else float("inf")
        print(f"{name:<16} {bits:8.2f}   {pooled:15.3e} {med:10.3e} {p99:8.3e} {snr:9.1f}")

    # The frontier: for each byte cost, the coordinate that wins it.
    print("\nfrontier (best coordinate at each byte cost, cheapest first):")
    best = {}
    for bits, name, pooled, _m, _p in rows:
        if bits not in best or pooled < best[bits][1]:
            best[bits] = (name, pooled)
    for bits in sorted(best):
        name, pooled = best[bits]
        print(f"  {bits:6.2f} bits/element  {name:<12} rel L2 {pooled:.3e}")
    report["frontier"] = {str(b): {"coordinate": n, "rel_l2": p} for b, (n, p) in best.items()}

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w") as fh:
            json.dump(report, fh, indent=1)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
